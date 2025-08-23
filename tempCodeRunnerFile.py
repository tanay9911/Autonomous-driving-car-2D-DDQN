import pygame
import numpy as np
from collections import deque
import random
import math
import os

import GameEnv
from ddqn_keras import DDQNAgent

# Constants
TOTAL_GAMETIME = 10000
N_EPISODES = 10000
REPLACE_TARGET = 10

# Set model path relative to this file
MODEL_DIR = os.path.join(os.path.dirname(__file__))
MODEL_FILENAME = 'ddqn_model.h5'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize game
game = GameEnv.RacingEnv()
game.fps = 60

# Agent setup (low epsilon for evaluation)
ddqn_agent = DDQNAgent(
    alpha=0.0005,
    gamma=0.99,
    n_actions=5,
    epsilon=0.02,
    epsilon_end=0.01,
    epsilon_dec=0.999,
    replace_target=REPLACE_TARGET,
    batch_size=64,
    input_dims=19,
    fname=MODEL_FILENAME  # just the filename
)

# Override model file path to full relative path
ddqn_agent.model_file = MODEL_PATH

# Load pre-trained weights and sync target network
ddqn_agent.load_model()
ddqn_agent.update_network_parameters()

# Score tracking
ddqn_scores = []
eps_history = []

# Toggle rendering
renderFlag = True


def run():
    for e in range(N_EPISODES):
        game.reset()

        done = False
        score = 0
        counter = 0
        gtime = 0

        # Initial step with no action (passive start)
        observation_, reward, done = game.step(0)
        observation = np.array(observation_)

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Select action using the trained agent
            action = ddqn_agent.choose_action(observation)
            observation_, reward, done = game.step(action)
            observation_ = np.array(observation_)

            # Time penalty for inaction
            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward
            observation = observation_

            gtime += 1
            if gtime >= TOTAL_GAMETIME:
                done = True

            if renderFlag:
                game.render(action)

        print(f"Episode {e}: Score = {score:.2f}")


if __name__ == "__main__":
    run()
