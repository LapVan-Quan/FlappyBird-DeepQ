# Flappy Bird AI Agent using Deep Q-Learning (DQN)
## Project Overview
This project implements a reinforcement learning (RL) agent to play **Flappy Bird** using the **Deep Q-Network (DQN)** algorithm. The agent learns to flap and survive as long as possible by interacting with the game environment and optimizing its behavior based on custom rewards.

> Note: This version of Flappy Bird is **more challenging than usual**, with **pipes appearing more frequently**, making learning and survival more difficult.


## Features

- Built with **PyTorch**
- Uses **Deep Q-Learning** with experience replay and target networks
- Game environment based on [`pygame`](https://www.pygame.org/)
- Trains an agent to learn the optimal flapping policy

## Algorithm

The agent uses the **Deep Q-Network (DQN)** algorithm:
- Q-value function is approximated by a deep neural network
- An **ε-greedy policy** balances exploration and exploitation
- A **replay buffer** stores past experiences for stable learning
- A **target network** improves stability during training

The pseudo-code for the **Deep Q-Network (DQN)** algorithm:
```
for episode = 1,E do
    Initialize state s_1
    while game is not finished do
        With probability ε select random action a_t ('do nothing' or 'jump'), 
        otherwise select a_t = argmax(network.predict(s_t))
        Execute action a_t and observe r_t and q_t (q_t will be approximated by neural network or 0 if it is terminal state)
        Store (phi_t, a_t, r_t, q_t_next) in D
        Sample a random minibatch of transitions from D
        For each (phi_j, a_j, r_j, q_j_next) in minibatch do
            Append phi_j to X
            Set w_j = onehot(a_j) and append to W
            Set y_j = r_j + self.discount_factor * q_j_next and append to Y
        Fit Q network for one step using X, Y, W
        Decay ε
```

## Game Environment

The environment simulates the **Flappy Bird** game:
- Observation:
  - Distance from bird to next pipe
  - Pipe position (coordinates)
  - Distance from bird to top and bottom pipes
  - Pipe width
  - Bird position (y-coordinate)
  - Bird velocity
- Action space: `1` (do nothing), `0` (jump)
- Reward:
## Reward Function
| Event                              | Reward                                 |
|-----------------------------------|----------------------------------------|
| Surviving each frame              | `+1`                                   |
| Flying in the center of pipe gap  | `+10`                                  |
| Proximity to center               | `+max(0, 5 - 0.05 * dist_to_center)`   |
| Finishing with 'well done'        | `+10`                                  |
| Crashing with 'hit pipe'          | `-100`                                 |
| Crashing with 'off screen'        | `-10`                                  |

## Performance

After training the agent for **20,000 episodes** with a maximum game length of **50 frames**, it was evaluated over **10 test episodes**. The agent achieved the following results:

- **Highest score:** 50
- **Average score:** 28
- **Success rate :** 80%


## How to Run?
### 1. Clone the project
```
git clone https://github.com/LapVan-Quan/FlappyBird-DeepQ.git
cd FlappyBird-DeepQ
```
### 2. Install dependencies
```
pip install -r requirements.txt
```
### 3. Run training code
```
python3 train.py
```
### 4. Run testing code
```
python3 test.py
```
### 5. Run my pretrained model (Optional)
```
python3 test_pretrained_model.py
```