from my_agent import MyAgent
from console import FlappyBirdEnv
import numpy as np

if __name__ == '__main__':
    game_length = 50
    train_episodes = 20000 
    best_score = float('-inf')
    model_path = 'my_model.ckpt'
    show_screen = False

    # train the model with 20000 epidsodes
    env = FlappyBirdEnv(config_file_path='config.yml', show_screen=show_screen, game_length=game_length)
    agent = MyAgent(show_screen=show_screen)
    
    for episode in range(train_episodes):
        env.play(player=agent)

        # env.score has the score value from the last play
        # env.mileage has the mileage value from the last play
        print(env.score)
        print(env.mileage)

        # store the best model based on your judgement
        if env.score > best_score:
            best_score = env.score
            agent.save_model(path=model_path)

        # clear the memory when the storage length reaches 50000
        if len(agent.storage) > 50000:
            agent.storage = agent.storage[-50000:]

        # update the fixed Q-target network (Q_f) with Q's model parameter after a few episodes
        if episode % 5:
            agent.update_network_model(net_to_update=agent.network2, net_as_source=agent.network)
            

    # evaluating the agent
    env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=show_screen, game_length=game_length)
    agent2 = MyAgent(show_screen=show_screen, load_model_path=model_path, mode='eval')

    eval_episodes = 10
    scores = list()
    for episode in range(eval_episodes):
        env2.play(player=agent2)
        scores.append(env2.score)
        print(f'{episode}: {env2.score}')

    print(np.max(scores))
    print(np.mean(scores))
