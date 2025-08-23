# Importing required Keras components
from tensorflow.keras.layers import Dense, Activation  # Importing dense layers and activation functions
from tensorflow.keras.models import Sequential, load_model  # Importing model types
from tensorflow.keras.optimizers import Adam  # Importing Adam optimizer
import numpy as np  # Importing numpy for numerical operations
import tensorflow as tf  # Importing TensorFlow
import os  # Importing os for file system operations

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        # Initializing replay buffer with specified capacity
        self.mem_size = max_size  # Setting maximum memory size
        self.mem_cntr = 0  # Initializing memory counter
        self.discrete = discrete  # Setting whether actions are discrete
        # Creating numpy arrays to store experiences
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)  # Storing states
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)  # Storing next states
        dtype = np.int8 if self.discrete else np.float32  # Setting dtype based on action type
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)  # Storing actions
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)  # Storing rewards
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)  # Storing done flags

    def store_transition(self, state, action, reward, state_, done):
        # Storing a single experience in memory
        index = self.mem_cntr % self.mem_size  # Calculating storage index
        self.state_memory[index] = state  # Storing current state
        self.new_state_memory[index] = state_  # Storing next state
        if self.discrete:  # Handling discrete actions
            actions = np.zeros(self.action_memory.shape[1])  # Creating one-hot vector
            actions[action] = 1.0  # Setting action index
            self.action_memory[index] = actions  # Storing action
        else:  # Handling continuous actions
            self.action_memory[index] = action  # Storing action directly
        self.reward_memory[index] = reward  # Storing reward
        self.terminal_memory[index] = 1 - int(done)  # Storing terminal flag (inverted)
        self.mem_cntr += 1  # Incrementing memory counter

    def sample_buffer(self, batch_size):
        # Sampling a batch of experiences from memory
        max_mem = min(self.mem_cntr, self.mem_size)  # Determining available memory
        batch = np.random.choice(max_mem, batch_size)  # Randomly selecting indices

        # Retrieving experiences from memory
        states = self.state_memory[batch]  # Getting batch of states
        actions = self.action_memory[batch]  # Getting batch of actions
        rewards = self.reward_memory[batch]  # Getting batch of rewards
        states_ = self.new_state_memory[batch]  # Getting batch of next states
        terminal = self.terminal_memory[batch]  # Getting batch of terminal flags

        return states, actions, rewards, states_, terminal  # Returning sampled batch


class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.999995, epsilon_end=0.10,
                 mem_size=25000, fname='ddqn_model.h5', replace_target=25):
        # Initializing Double DQN agent with parameters
        self.action_space = [i for i in range(n_actions)]  # Creating action space
        self.n_actions = n_actions  # Storing number of actions
        self.gamma = gamma  # Setting discount factor
        self.epsilon = epsilon  # Setting initial exploration rate
        self.epsilon_dec = epsilon_dec  # Setting exploration decay rate
        self.epsilon_min = epsilon_end  # Setting minimum exploration rate
        self.batch_size = batch_size  # Setting batch size for training

        # Configuring model file path
        self.model_file = os.path.join(os.path.dirname(__file__), 'IMT2022083', fname)  # Setting model path

        self.replace_target = replace_target  # Setting target network update frequency
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)  # Creating replay buffer

        # Creating evaluation and target networks
        self.brain_eval = Brain(input_dims, n_actions, batch_size)  # Main network for action selection
        self.brain_target = Brain(input_dims, n_actions, batch_size)  # Target network for stability

    def remember(self, state, action, reward, new_state, done):
        # Storing experience in replay buffer
        self.memory.store_transition(state, action, reward, new_state, done)  # Delegating to replay buffer

    def choose_action(self, state):
        # Selecting action using ε-greedy policy
        state = np.array(state)  # Converting state to numpy array
        state = state[np.newaxis, :]  # Adding batch dimension

        rand = np.random.random()  # Generating random number
        if rand < self.epsilon:  # Exploring with probability ε
            action = np.random.choice(self.action_space)  # Choosing random action
        else:  # Exploiting with probability 1-ε
            actions = self.brain_eval.predict(state)  # Getting Q-values from network
            action = np.argmax(actions)  # Choosing action with highest Q-value

        return action  # Returning selected action

    def learn(self):
        # Performing learning step if enough experiences are stored
        if self.memory.mem_cntr > self.batch_size:  # Checking if buffer has enough samples
            # Sampling batch from replay buffer
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

            # Converting one-hot actions to indices
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            # Calculating target Q-values using Double DQN approach
            q_next = self.brain_target.predict(new_state)  # Getting Q-values from target network
            q_eval = self.brain_eval.predict(new_state)  # Getting Q-values from eval network
            q_pred = self.brain_eval.predict(state)  # Getting current Q-value predictions

            max_actions = np.argmax(q_eval, axis=1)  # Selecting best actions from eval network

            q_target = np.copy(q_pred)  # Creating copy of predictions as target

            batch_index = np.arange(self.batch_size, dtype=np.int32)  # Creating batch indices

            # Calculating target Q-values using Bellman equation
            q_target[batch_index, action_indices] = reward + self.gamma * q_next[batch_index, max_actions.astype(int)] * done

            # Training evaluation network
            _ = self.brain_eval.train(state, q_target)

            # Decaying exploration rate
            self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def update_network_parameters(self):
        # Updating target network weights
        self.brain_target.copy_weights(self.brain_eval)  # Copying weights from evaluation network

    def save_model(self):
        # Saving model to file
        self.brain_eval.model.save(self.model_file)  # Saving evaluation network

    def load_model(self):
        # Loading model from file
        self.brain_eval.model = load_model(self.model_file, compile = False)  # Loading evaluation network
        self.brain_target.model = load_model(self.model_file, compile = False)  # Loading target network

        if self.epsilon == 0.0:  # If in pure exploitation mode
            self.update_network_parameters()  # Syncing target network


class Brain:
    def __init__(self, NbrStates, NbrActions, batch_size=256):
        # Initializing neural network brain
        self.NbrStates = NbrStates  # Storing state dimension
        self.NbrActions = NbrActions  # Storing action dimension
        self.batch_size = batch_size  # Storing batch size
        self.model = self.createModel()  # Creating neural network model

    def createModel(self):
        # Creating neural network architecture
        model = Sequential()  # Creating sequential model
        model.add(Dense(256, input_dim=self.NbrStates, activation='relu'))  # Adding hidden layer
        model.add(Dense(self.NbrActions, activation='softmax'))  # Adding output layer
        model.compile(loss='mse', optimizer=Adam())  # Compiling model with MSE loss and Adam optimizer
        return model  # Returning compiled model

    def train(self, x, y, epoch=1, verbose=0):
        # Training the neural network
        self.model.fit(x, y, batch_size=self.batch_size, epochs=epoch, verbose=verbose)  # Performing training step

    def predict(self, s):
        # Making batch predictions
        return self.model.predict(s, verbose=0)  # Returning Q-values for batch of states

    def predictOne(self, s):
        # Making single prediction
        return self.model.predict(tf.reshape(s, [1, self.NbrStates]), verbose=0).flatten()  # Returning Q-values for single state

    def copy_weights(self, TrainNet):
        # Copying weights from another network
        self.model.set_weights(TrainNet.model.get_weights())  # Setting weights to match source network