import numpy as np
import pygame
from pytorch_mlp import MLPRegression
import argparse
from console import FlappyBirdEnv
import random
import torch
from torch import nn
from my_agent import MyAgent
from matplotlib import pyplot as plt

if __name__ == '__main__':
    game_length = 50
    episodes = 10
    model_path = 'my_model.ckpt'
    show_screen = False
    scores = list()

    # evaluatint the agent
    env = FlappyBirdEnv(config_file_path='config.yml', show_screen=show_screen, game_length=game_length)
    agent = MyAgent(show_screen=show_screen, load_model_path=model_path, mode='eval')

    success_count = 0
    for episode in range(episodes):
        env.play(player=agent)
        scores.append(env.score)
        if env.score > 1:
            success_count += 1
        print(f'{episode}: {env.score}')
        
    print(f'Highest score: {np.max(scores)}')
    print(f'Average score: {np.mean(scores)}')
    print(f'Success rate: {(success_count/episodes) * 100}%')