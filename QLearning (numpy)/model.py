import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class QModel:
    
    def __init__(self):
        # Memory
        self.memory = []
        # Q-table
        self.q_table = {}
        # Hyperparameters
        self.alpha = 0.05
        self.gamma = 0.9
        self.epsilon = 0.1
        # Epochs
        self.NUM_EPOCHS = 0
        # For visualization
        self.scores = []
        self.max_scores  = []
        self.df = pd.DataFrame()

    # states are rounded to the tens place to make it more likely to avert undertraining
    def get_player_state(self, player_y,player_x, nearestPipeDict, yVelocity):
        x_dist_from_lower_pipes = round((nearestPipeDict['x'] - player_x) / 5, -1)
        y_dist_from_lower_pipes = round((nearestPipeDict['y'] - player_y) / 5, -1)
        y_dist_from_upper_pipes = round((nearestPipeDict['y'] - 100 - player_y) / 10, -1) # dist between upper and lower is always 100

        player_state_data = (x_dist_from_lower_pipes,
                            y_dist_from_lower_pipes,
                            y_dist_from_upper_pipes,
                            yVelocity)

        # create a placeholder in the Q-Table if does not exist
        try:
            self.q_table[player_state_data]
        except:
            self.q_table[player_state_data] = [0,0]

        self.state = player_state_data 
        return player_state_data

    def choose_best_action(self):
        state = self.state
        if random.uniform(0,1) < self.epsilon:
            action = random.choice([0,1]) # flap or don't flap
            self.epsilon -= 0.001
        else:
            action = np.argmax(self.q_table[state])
            print(self.q_table[state])
        return action
    
    def get_reward(self, crash, playery):
        if crash:
            reward = -1000
        else:
            reward = 1
        return reward

    def remember_SARS(self, player_initial_state, action, reward, player_final_state):
        self.memory.append((player_initial_state, action, reward, player_final_state))

    def replay_memories(self):
        print('mem len: ', len(self.memory))
        if len(self.memory) > 1000:
            batch = random.sample(self.memory, 1000)
        else:
            batch = self.memory
        print(batch[0])
        print('len q_table: ', len(self.q_table))
        for player_initial_state, action, reward, player_final_state in batch:
            # existing q_table value for this state
            initial_score = self.q_table[player_initial_state][action]
            next_max = np.max(self.q_table[player_final_state])
            
            new_value = (1 - self.alpha) * initial_score + self.alpha * (reward + self.gamma * next_max)
            # append new value to q_table
            self.q_table[player_initial_state][action] = new_value

    def plot_scores(self):
        """Upon pressing the ESC key in-game, this function will trigger"""
        self.df['max_scores'] = self.max_scores
        self.df.to_csv('max_scores.csv')
        plot_df = self.df.reset_index()
        plot_df.plot(kind='scatter', x='index', y='max_scores')
        plt.savefig('scores_x.png',bbox_inches = 'tight')
        plt.show()

        