# Corrigé TP Q-learning

import gym
from gym import spaces
import numpy as np

import sys
sys.path.append("..")
from resources import discrete_vec_to_state, show_rewards
from tqdm import tqdm

# 1.1/1
aire_de_jeu = spaces.Box(low=np.array([0, 0]), high=np.array([5, 3]))

# 1.1/2
print("low :", aire_de_jeu.low)
print("high :", aire_de_jeu.high)

# 1.1/3
print("position aléatoire :", aire_de_jeu.sample())

# 1.1/4
espace_action = spaces.Discrete(4)

# 1.1/5
print("action aléatoire :", espace_action.sample())

# 1.2.2/1.a
environment_name = 'CartPole-v1'
env = gym.make(environment_name)

# 1.2.2/1.b
print("observation_space :", env.observation_space)
print("action_space :", env.action_space)

# 1.2.2/2.a
observation = env.reset()

# 1.2.2/2.b
a = env.action_space.sample()

# 1.2.2/2.c
print(env.step(a))

# 1.2.2/3
def do_episode(env):
    # initialisation de l'environnement
    env.reset()
    # création de deux variables (terminated et truncated) pour savoir
    # lorsque l'épisode se termine ou est tronqué
    terminated, truncated = False, False
    # création d'une variable pour accumuler la récompense totale
    total_reward = 0
    while not (terminated or truncated):
        # choix d'une action aléatoire
        action = env.action_space.sample()
        # réalision de l'action (pas besoin de la nouvelle observation ni des info dans cette fonction)
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
    return total_reward

# 1.2.2/4
env = gym.make(environment_name, render_mode="human")
# bouble de 10 épisodes
episodes = 10
for episode in range(1, episodes+1):
    total_reward = do_episode(env)
    print(f'Episode:{episode}, Gain total :{total_reward}')
env.close()

# 2.1/1.a
cardinals = [10, 10, 10, 10] # on souhaite découper chaque variable en 10 segments

# 2.1/1.b
bounds = [(-2.4, 2.4), (0, 0), (-.2095, .2095), (0, 0)]
env = gym.make(environment_name)
episodes = 10000
for episode in range(1, episodes+1):
    env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)
        bounds[1] = (min(bounds[1][0], observation[1]), max(bounds[1][1], observation[1]))
        bounds[3] = (min(bounds[3][0], observation[3]), max(bounds[3][1], observation[3]))
print(bounds)

# 2.1/2.a
def discretize_value(x, bound, n):
    inf, sup = bound
    if x <= inf: return 0
    if x >= sup: return n - 1
    else:
        interval = (sup - inf)/n
        return (x-inf)//interval

# 2.1/2.b
def discretize_observation(observation, bounds, cardinals):
    return np.array([discretize_value(x, bounds[i], cardinals[i]) for i, x in enumerate(observation)])

# 2.1/2.c
def observation_to_state(observation, bounds, cardinals):
    discrete_vec = discretize_observation(observation, bounds, cardinals)
    return discrete_vec_to_state(discrete_vec, cardinals)

# 2.1/3
n_states = np.prod(cardinals)
n_actions = 2
Q = np.zeros((n_states, n_actions))

# 2.2/1.a
def explore(num, l):
    p = np.exp(-l*num)
    return np.random.rand() < p

# 2.2/1.b
def select_random_action(n):
    return np.random.randint(n)

# 2.2/1.c
def select_best_action(state, Q):
    return np.argmax(Q[state])

# 2.2/1.d
def select_action(state, Q, num, l=1e-3):
    return select_random_action(Q.shape[1]) if explore(num, l) else select_best_action(state, Q)

# 2.2/2
env = gym.make(environment_name)
alpha = 0.1
gamma = 0.99
episodes = 10000
l = 1e-3
total_rewards = []
for num_episode in tqdm(range(1, episodes+1)):
    observation, _ = env.reset()
    state = observation_to_state(observation, bounds, cardinals)
    terminated = False
    truncated = False
    total_reward = 0
    while not (terminated or truncated):
        action = select_action(state, Q, num_episode, l)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        new_state = observation_to_state(observation, bounds, cardinals)
        Q[state][action] = (1-alpha)*Q[state][action] + \
            alpha*(reward + gamma*max(Q[new_state]))
        state = new_state
    total_rewards.append(total_reward)

# 2.2/3
show_rewards(total_rewards)

# 2.3/1
def do_episode(env, Q):
    observation, _ = env.reset()
    terminated, truncated = False, False
    total_reward = 0
    while not (terminated or truncated):
        state = observation_to_state(observation, bounds, cardinals)
        action = np.argmax(Q[state])
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
    return total_reward

# 2.3/2
env = gym.make(environment_name, render_mode="human")
episodes = 5
for episode in range(1, episodes+1):
    score = do_episode(env, Q)
    print(f'Episode:{episode} Score:{score}')
env.close()

import pandas as pd
pd.DataFrame(data=Q).to_csv("data/qtable.csv")