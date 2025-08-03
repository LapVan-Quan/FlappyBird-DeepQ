from console import FlappyBirdEnv
from human_agent import HumanAgent
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=6)

    args = parser.parse_args()

    # Game environment
    env = FlappyBirdEnv(config_file_path='config.yml', show_screen=True, level=args.level, game_length=100)
    # a human agent (yourself) playing the game using keyboard
    human = HumanAgent(show_screen=True)

    while True:
        env.play(player=human)
        print('Game Over')
        if env.replay_game():
            print(f"Game restart.")
        else:
            break
