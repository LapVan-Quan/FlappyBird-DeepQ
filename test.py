import numpy as np
import pygame
from pytorch_mlp import MLPRegression
import argparse
from console import FlappyBirdEnv
import random
import torch
from torch import nn
from my_agent import MyAgent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=1)

    args = parser.parse_args()

    # evaluatint the agent
    env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level, game_length=50)
    agent2 = MyAgent(show_screen=False, load_model_path='./pretrained-model/my_model-level7.ckpt', mode='eval')

    episodes = 10
    scores = list()
    for episode in range(episodes):
        env2.play(player=agent2)
        scores.append(env2.score)
        print(f'{episode}: {env2.score}')
        

    print(np.max(scores))
    print(np.mean(scores))