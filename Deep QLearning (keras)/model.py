import numpy as np
import random
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LeakyReLU
import matplotlib.pyplot as plt
import pandas as pd

class QModel:
    
    def __init__(self):
        # Memory
        self.memory = []
        # num state dimensions
        self.NUM_DIM = 4
        # num_epochs
        self.NUM_EPOCHS = 0
        # Hyperparameters
        self.alpha = 0.005
        self.gamma = 0.9
        self.epsilon = 0.05
        # For visualization
        self.scores = []
        self.max_scores = []
        self.df = pd.DataFrame()
        # Model
        self.model = self.network() # loads weights if desired

    def get_player_state(self, player_y, player_x, nearestPipeDict, yVelocity):
        # delta x to the lower pipe
        x_dist_from_lower_pipes = (nearestPipeDict['x'] - player_x)
        # delta y to the lower pipe
        y_dist_from_lower_pipes = (nearestPipeDict['y'] - player_y)
        # delta y to the upper pipe (gap size is 100)
        y_dist_from_upper_pipes = (nearestPipeDict['y'] - player_y - 100) # dist between upper and lower is always 100

        player_state_data = [x_dist_from_lower_pipes,
            y_dist_from_lower_pipes,
            y_dist_from_upper_pipes,
            yVelocity]
        return np.asarray(player_state_data)

    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(128, input_dim=self.NUM_DIM))
        model.add(LeakyReLU())
        model.add(Dense(128))
        model.add(LeakyReLU())
        model.add(Dense(128))
        model.add(LeakyReLU())
        model.add(Dense(output_dim=2, activation='linear'))
        opt = Adam(self.alpha, beta_1 = .8)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def get_reward(self, crash, playery, score):
        if crash:
            reward = -1000
        else:
            reward = 1
        return reward

    def remember_SARS(self, player_initial_state, action, reward, player_final_state, crash):
        self.memory.append((player_initial_state, action, reward, player_final_state, crash))

    def replay_memories(self):
        print('mem len: ', len(self.memory))
        if len(self.memory) > 1000:
            batch = random.sample(self.memory, 1000)
        else:
            batch = self.memory
        # save model weights
        self.model.save('saved_models/flappy_model_weights.h5')
        states = []
        targets = []
        for player_initial_state, action, reward, player_final_state, crash in batch:
            target = reward
            if not crash:
                target = reward + self.gamma * np.max(self.model.predict(np.array([player_final_state]))[0])
            target_f = self.model.predict(np.array([player_initial_state]))
            target_f[0][action] = target
            # Appending for batch model fitting
            states.append(player_initial_state)
            targets.append(target_f[0])

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
    
    def plot_scores(self):
        self.df['max_scores'] = self.max_scores
        self.df.to_csv('max_scores.csv') # commented out to not overwrite
        plot_df = self.df.reset_index()
        # breakpoint()
        plot_df.plot(kind='scatter', x='index', y='max_scores')
        plt.savefig('scores_x.png',bbox_inches = 'tight')
        plt.show()
