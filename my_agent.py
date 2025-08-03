import numpy as np
import pygame
from pytorch_mlp import MLPRegression
import argparse
from console import FlappyBirdEnv
import random
import torch
from torch import nn



class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode=None):
        self.show_screen = show_screen
        if mode is None:
            self.mode = 'train'  # mode is either 'train' or 'eval', we will set the mode of your agent to eval mode
        else:
            self.mode = mode

        self.storage = []  # storage that stores (state, action, reward, q-value)
        # A neural network MLP model which can be used as Q
        self.network = MLPRegression(input_dim=10, output_dim=2, learning_rate=0.001)
        # network2 has identical structure to network1, network2 is the Q_f
        self.network2 = MLPRegression(input_dim=10, output_dim=2, learning_rate=0.001)
        # initialise Q_f's parameter by Q's, here is an example
        MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

        self.epsilon = 0.01  # ε 
        self.n = 64  # batch size
        self.discount_factor = 1  # γ

        if load_model_path:
            self.load_model(load_model_path)

    def build_state(self, state: dict) -> np.ndarray:
        
        def extract_next_pipe_features(bird_x, pipes):
            for pipe in pipes:
                if pipe['x'] + pipe['width'] > bird_x:
                    return pipe['x'], pipe['top'], pipe['bottom'], pipe['width']
            return 0, 0, 0, 0  

        bird_x = state['bird_x']
        bird_y = state['bird_y']
        bird_velocity = state['bird_velocity']

        pipe_x, pipe_top_y, pipe_bottom_y, pipe_width = extract_next_pipe_features(bird_x, state['pipes'])

        phi = np.array([
            bird_x, 
            bird_y, 
            bird_velocity,
            pipe_x, 
            pipe_top_y, 
            pipe_bottom_y,
            pipe_width,
            pipe_top_y - bird_y,
            bird_y - pipe_bottom_y,
            pipe_x - bird_x   
        ], dtype=np.float32)

        return phi

    def compute_reward(self, next_state: dict) -> float:
        """
        Computes the reward given the current and next game states.

        Args:
            next_state: state after taking the action

        Returns:
            reward: a float reward value
        """
        # Terminal case (game over)
        if next_state['done']:
            if (next_state['done_type'] == 'hit pipe'):
                return -100.0
            elif next_state['done_type'] == 'off_screen':
                return -10
            # reward of succesfully finish the game
            elif next_state['done_type'] == 'well_done':
                return 10.0  

        # Reward for staying alive
        reward = 1.0
        phi = self.build_state(next_state)

        # Reward for flying in the center area       
        if (phi[7] > 0 and phi[8] > 0):
            reward += 10.0

        center_y = (phi[4] - phi[5]) / 2 + phi[5]
        dist_to_center = abs(next_state['bird_y'] - center_y)
        reward += max(0, 5 - 0.05 * dist_to_center)
        
        return reward


    def choose_action(self, state: dict, action_table: dict) -> int:
        """
        This function should be called when the agent action is requested.
        Args:
            state: input state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            action: the action code as specified by the action_table
        """
        phi_t = self.build_state(state)  

        if self.mode == "train":
            if np.random.rand() <= self.epsilon:
                a_t = list(action_table.values())[random.randint(0, 1)]
            else:
                q_values = self.network2.predict(phi_t)
                a_t = np.argmax(q_values)

            # Store partial transition in memory
            self.last_phi = phi_t
            self.last_action = a_t
            
            # rt, q_t+1 will be filled in later
            self.storage.append([phi_t, a_t, 0, 0])  
        else:
            q_values = self.network.predict(phi_t)
            a_t = np.argmax(q_values)
        
        return a_t

    def receive_after_action_observation(self, state: dict, action_table: dict) -> None:
        """
        This function should be called to notify the agent of the post-action observation.
        Args:
            state: post-action state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            None
        """
        if self.mode != "train":
            return

        # Get the essential states from state
        phi_next = self.build_state(state)
        # Get reward
        r_t = self.compute_reward(state)  
        # Compute q_t+1
        # Terminal state
        if state['done']:  
            q_next = 0
        else:
            q_next = np.max(self.network2.predict(phi_next))  
        # Update last transition with (r_t, q_next)
        self.storage[-1][2] = r_t
        self.storage[-1][3] = q_next

        # Sample minibatch from D and train
        if len(self.storage) >= self.n:
            batch = random.sample(self.storage, self.n)
            X, Y, W = [], [], []
            for phi_j, a_j, r_j, q_j1 in batch:
                phi_j = np.array(phi_j).flatten()  
                w_j = np.zeros(2)
                w_j[a_j] = 1
                y_j = r_j + self.discount_factor * q_j1

                X.append(phi_j)
                Y.append([y_j])       
                W.append(w_j)

            self.network.fit_step(np.array(X), np.array(Y), np.array(W))
            self.epsilon = max(0.01, self.epsilon * 0.995)

    def save_model(self, path: str = 'my_model.ckpt'):
        """
        Save the MLP model. Unless you decide to implement the MLP model yourself, do not modify this function.

        Args:
            path: the full path to save the model weights, ending with the file name and extension

        Returns:

        """
        self.network.save_model(path=path)

    def load_model(self, path: str = 'my_model.ckpt'):
        """
        Load the MLP model weights.  Unless you decide to implement the MLP model yourself, do not modify this function.
        Args:
            path: the full path to load the model weights, ending with the file name and extension

        Returns:

        """
        self.network.load_model(path=path)

    @staticmethod
    def update_network_model(net_to_update: MLPRegression, net_as_source: MLPRegression):
        """
        Update one MLP model's model parameter by the parameter of another MLP model.
        Args:
            net_to_update: the MLP to be updated
            net_as_source: the MLP to supply the model parameters

        Returns:
            None
        """
        net_to_update.load_state_dict(net_as_source.state_dict())
