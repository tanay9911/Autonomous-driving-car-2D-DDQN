# Importing required libraries
import pygame  # Importing pygame for game visualization
import numpy as np  # Importing numpy for numerical operations
import random  # Importing random for random number generation
import math  # Importing math for mathematical functions
from collections import deque  # Importing deque for efficient memory storage
import os  # Importing os for file system operations

# Importing custom implementations
from ddqn_keras import DDQNAgent  # Importing Double DQN agent implementation
import GameEnv  # Importing custom racing game environment

# Defining training constants
TOTAL_GAMETIME = 1000  # Setting maximum timesteps per episode
N_EPISODES = 10000  # Setting total number of training episodes
REPLACE_TARGET = 50  # Setting frequency for target network updates

# Configuring model save path
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'IMT2022083')  # Creating model directory path
MODEL_FILENAME = 'ddqn_model.h5'  # Defining model filename
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)  # Creating full model path

# Ensuring model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)  # Creating directory if it doesn't exist

# Initializing game environment
game = GameEnv.RacingEnv()  # Creating instance of racing environment
game.fps = 60  # Setting frames per second for rendering

# Creating DDQN agent with training parameters
ddqn_agent = DDQNAgent(
    alpha=0.0005,  # Setting learning rate
    gamma=0.99,  # Setting discount factor for future rewards
    n_actions=5,  # Setting number of possible actions
    epsilon=1.00,  # Setting initial exploration rate
    epsilon_end=0.10,  # Setting minimum exploration rate
    epsilon_dec=0.9995,  # Setting exploration rate decay
    replace_target=REPLACE_TARGET,  # Setting target network update frequency
    batch_size=512,  # Setting batch size for training
    input_dims=19,  # Setting input dimensions (state size)
    fname=MODEL_FILENAME  # Setting model filename
)

# Updating model path in agent
ddqn_agent.model_file = MODEL_PATH  # Setting full model path

# Uncomment to load existing model
# ddqn_agent.load_model()  # Loading pre-trained weights if available

# Initializing performance tracking lists
ddqn_scores = []  # Creating list to store episode scores
eps_history = []  # Creating list to store epsilon values


def run():
    # Main training loop
    for e in range(N_EPISODES):
        # Resetting environment for new episode
        game.reset()  # Resetting game state to initial conditions

        # Initializing episode variables
        done = False  # Flag for episode completion
        score = 0  # Accumulating total reward
        counter = 0  # Counting steps without reward

        # Getting initial observation
        observation_, reward, done = game.step(0)  # Taking no action initially
        observation = np.array(observation_)  # Converting observation to numpy array

        gtime = 0  # Initializing timestep counter
        renderFlag = (e % 10 == 0 and e > 0)  # Deciding whether to render this episode

        # Running episode until completion
        while not done:
            # Handling pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # Checking for window close event
                    pygame.quit()  # Closing pygame
                    return  # Exiting the function

            # Selecting and executing action
            action = ddqn_agent.choose_action(observation)  # Choosing action using ε-greedy policy
            observation_, reward, done = game.step(action)  # Executing action in environment
            observation_ = np.array(observation_)  # Converting new observation to numpy array

            # Handling time penalty for inaction
            if reward == 0:  # Checking if no reward was received
                counter += 1  # Incrementing inaction counter
                if counter > 100:  # Checking if too many steps without reward
                    done = True  # Ending episode early
            else:
                counter = 0  # Resetting inaction counter if reward received

            # Updating total score
            score += reward  # Accumulating rewards

            # Storing experience in replay memory
            ddqn_agent.remember(
                observation,  # Storing current state
                action,  # Storing chosen action
                reward,  # Storing received reward
                observation_,  # Storing next state
                int(done),  # Storing done flag
            )
            
            # Updating observation
            observation = observation_  # Setting current observation to new observation
            
            # Training the agent
            ddqn_agent.learn()  # Performing learning step with sampled batch

            # Incrementing timestep counter
            gtime += 1  # Counting current timestep
            if gtime >= TOTAL_GAMETIME:  # Checking if maximum timesteps reached
                done = True  # Ending episode

            # Rendering game if enabled
            if renderFlag:
                game.render(action)  # Displaying game state with current action

        # Storing episode statistics
        eps_history.append(ddqn_agent.epsilon)  # Recording current epsilon value
        ddqn_scores.append(score)  # Recording episode score
        
        # Calculating average score
        avg_score = np.mean(ddqn_scores[max(0, e - 100):(e + 1)])  # Computing 100-episode average

        # Updating target network periodically
        if e % REPLACE_TARGET == 0 and e > REPLACE_TARGET:
            ddqn_agent.update_network_parameters()  # Syncing target network with main network

        # Saving model periodically
        if e % 10 == 0 and e > 10:
            ddqn_agent.save_model()  # Saving model weights to file
            print("Model saved.")  # Printing save confirmation

        # Printing training progress
        print(
            f"Episode: {e}, Score: {score:.2f}, "  # Printing current episode and score
            f"Average Score: {avg_score:.2f}, "  # Printing average score
            f"Epsilon: {ddqn_agent.epsilon:.3f}, "  # Printing current exploration rate
            f"Memory Size: {ddqn_agent.memory.mem_cntr % ddqn_agent.memory.mem_size}"  # Printing memory usage
        )


# Main execution block
if __name__ == "__main__":
    run()  # Starting the training process