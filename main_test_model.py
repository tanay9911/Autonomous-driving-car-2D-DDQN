# Importing necessary libraries
import pygame  # Importing pygame for game rendering and sound
import numpy as np  # Importing numpy for numerical operations
from collections import deque  # Importing deque for efficient queue operations
import random  # Importing random for random number generation
import math  # Importing math for mathematical operations
import os  # Importing os for operating system interactions

# Importing custom game environment and DDQN agent
import GameEnv  # Importing the custom racing game environment
from ddqn_keras import DDQNAgent  # Importing the Double DQN agent implementation

# Defining constants for the training process
TOTAL_GAMETIME = 10000  # Setting maximum timesteps per episode
N_EPISODES = 10000  # Setting total number of training episodes
REPLACE_TARGET = 10  # Setting frequency for target network updates

# Setting up model file paths
MODEL_DIR = os.path.join(os.path.dirname(__file__))  # Getting directory of current file
MODEL_FILENAME = 'ddqn_model.h5'  # Defining model filename
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)  # Creating full model path

# Ensuring the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)  # Creating directory if it doesn't exist

# Setting base directory for sound file
BASE_DIR = os.path.dirname(__file__)  # Getting base directory path
SOUND_FILENAME = 'car_engine.wav'  # Defining sound filename
SOUND_PATH = os.path.join(BASE_DIR, SOUND_FILENAME)  # Creating full sound path

# Initializing pygame and sound systems
pygame.init()  # Initializing all pygame modules
pygame.mixer.init()  # Initializing pygame mixer for sound
engine_sound = pygame.mixer.Sound(SOUND_PATH)  # Loading engine sound file
engine_sound.set_volume(0.3)  # Setting sound volume to 30%

# Creating game environment instance
game = GameEnv.RacingEnv()  # Initializing racing game environment
game.fps = 60  # Setting frames per second for the game

# Creating DDQN agent with specific parameters
ddqn_agent = DDQNAgent(
    alpha=0.0005,  # Setting learning rate
    gamma=0.99,  # Setting discount factor
    n_actions=5,  # Setting number of possible actions
    epsilon=0.02,  # Setting initial exploration rate
    epsilon_end=0.01,  # Setting minimum exploration rate
    epsilon_dec=0.999,  # Setting exploration decay rate
    replace_target=REPLACE_TARGET,  # Setting target network update frequency
    batch_size=64,  # Setting batch size for training
    input_dims=19,  # Setting input dimensions (state size)
    fname=MODEL_FILENAME  # Setting model filename
)

# Updating model file path in agent
ddqn_agent.model_file = MODEL_PATH  # Setting full model path in agent

# Loading pre-trained model weights
ddqn_agent.load_model()  # Loading saved model weights if available
ddqn_agent.update_network_parameters()  # Syncing target network with main network

# Initializing lists for tracking performance
ddqn_scores = []  # Creating list to store episode scores
eps_history = []  # Creating list to store epsilon values

# Setting rendering flag
renderFlag = True  # Controlling whether to render the game visually


def run():
    # Running episodes loop
    for e in range(N_EPISODES):
        # Resetting game environment for new episode
        game.reset()  # Resetting all game elements to starting positions
        engine_sound.play(loops=-1)  # Starting engine sound with infinite looping

        # Initializing episode variables
        done = False  # Flag for episode completion
        score = 0  # Accumulating total reward
        counter = 0  # Counting steps without reward
        gtime = 0  # Counting timesteps in current episode

        # Taking initial observation
        observation_, reward, done = game.step(0)  # Getting initial state with no action
        observation = np.array(observation_)  # Converting observation to numpy array

        # Running episode until completion
        while not done:
            # Handling pygame events (like window close)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # Checking for window close event
                    engine_sound.stop()  # Stopping engine sound
                    pygame.quit()  # Closing pygame
                    return  # Exiting the function

            # Selecting action using trained policy
            action = ddqn_agent.choose_action(observation)  # Choosing action based on current state
            observation_, reward, done = game.step(action)  # Executing action in environment
            observation_ = np.array(observation_)  # Converting new observation to numpy array

            # Handling time penalty for inaction
            if reward == 0:  # Checking if no reward was received
                counter += 1  # Incrementing inaction counter
                if counter > 100:  # Checking if too many steps without reward
                    done = True  # Ending episode due to inaction
            else:
                counter = 0  # Resetting inaction counter if reward received

            # Updating episode statistics
            score += reward  # Accumulating total reward
            observation = observation_  # Updating current observation

            # Incrementing timestep counter
            gtime += 1  # Counting current timestep
            if gtime >= TOTAL_GAMETIME:  # Checking if maximum timesteps reached
                done = True  # Ending episode

            # Rendering game if enabled
            if renderFlag:
                game.render(action)  # Displaying game state with current action

        # Ending episode cleanup
        engine_sound.stop()  # Stopping engine sound after episode
        print(f"Episode {e}: Score = {score:.2f}")  # Printing episode results


# Main execution block
if __name__ == "__main__":
    run()  # Starting the training/evaluation process