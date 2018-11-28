#!/usr/bin/env python
import random
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gym_bridge.envs.bridge_env import BridgeEnv, check_game_status,\
    after_action_state

PASS = 0
MAX_BID = 52

class nLayerNet(nn.Module):
    '''
    A neural-net with ReLU activations and softmax output.
    
        pi(a | s) = softmax(bn + Wn * ReLU(... ReLU(b1 + W1 * o)))
        
    Notation:
        s = state
        o = observation
        a = action
        r = reward
        pi = agent's action distribution
        
    Input shape:    (batch_size, D_in)
    Output shape:   (batch_size, D_out)
    '''
    
    def __init__(self, params):
        # params (tuple) - n, D_in, D_h, D_out
        super(nLayerNet, self).__init__()
        n, D_in, D_h, D_out = params
        
        self.l_in = nn.Linear(D_in, D_h)
        self.l_hs = [nn.Linear(D_h, D_h) for _ in range(n - 2)]
        self.l_out = nn.Linear(D_h, D_out)
        
    def forward(self, x):
        
        x = self.l_in(x)
        x = F.relu(x)
        for l_h in self.l_hs:
            x = l_h(x)
            x = F.relu(x)
        x = self.l_out(x)
        x = F.softmax(x, dim=-1)
        
        return x

def torch_to_numpy(tensor):
    return tensor.data.numpy()
    
def numpy_to_torch(array):
    return torch.tensor(array).float()
    
class PolicyGradientNNAgent:
    '''
    Neural-net agent that trains using the policy gradient algorithm REINFORCE.
    Before updates, we standardize (mean 0, variance 1) the discounted rewards
    in each batch of episodes.
    '''
    
    def __init__(self, new_network, params, obs_to_input, lr=1e-3, df=0.8):
    
        # model and parameters
        self.model = new_network(params)
        if isinstance(self.model, torch.nn.Module):
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.df = df # discount factor
        
        # initialize replay history
        self.replay = []
        
        # function that converts observation into input of dimension D_in
        self.obs_to_input = obs_to_input
        
        # if trainable is changed to false, the model won't be updated
        self.trainable = True
        
    def move(self, o, invalid_moves=[], display=False):
        
        # feed observation as input to net to get distribution as output
        x = self.obs_to_input(o)
        x = numpy_to_torch([x])
        y = self.model(x)
        pi = torch_to_numpy(y).flatten()
        
        if np.isnan(pi).any():
            raise Exception("nan value in move distribution")
        
        # sample action from distribution with invalid moves excluded
        pi[invalid_moves] = 0
        pi = pi / np.sum(pi)
        a = np.random.choice(np.arange(len(pi)), p=pi)
        
        if display:
            print("pi(a | s) =", pi.reshape((3, 3)), sep='\n')
        
        # update current episode in replay with observation and chosen action
        if self.trainable:
            self.replay[-1]['observations'].append(o)
            self.replay[-1]['actions'].append(a)
        
        return a
        
    def new_episode(self):
        # start a new episode in replay
        self.replay.append({'observations': [], 'actions': [], 'rewards': []})
        
    def store_reward(self, r):
        # insert 0s for actions that received no reward; end with reward r
        episode = self.replay[-1]
        T_no_reward = len(episode['actions']) - len(episode['rewards']) - 1
        episode['rewards'] += [0.0] * T_no_reward + [r]
        
    def _calculate_discounted_rewards(self):
        # calculate and store discounted rewards per episode
        # returns mean and std of disc. rewards across all episodes in replay
        
        R_all = []
        for episode in self.replay:
            
            R = episode['rewards']
            R_disc = []
            R_sum = 0
            for r in R[::-1]:
                R_sum = r + self.df * R_sum
                R_disc.insert(0, R_sum)
                
            episode['rewards_disc'] = R_disc
            
            R_all += R_disc
            
        return np.mean(R_all), np.std(R_all)
        
    def update(self):
        
        assert(self.trainable)
        
        episode_losses = torch.tensor(0.0)
        N = len(self.replay)
        R_mean, R_std = self._calculate_discounted_rewards()
        
        for episode in self.replay:

            O = episode['observations']
            A = episode['actions']
            R_disc = episode['rewards_disc']
            T = len(R_disc)
            
            # forward pass
            X = numpy_to_torch([self.obs_to_input(o) for o in O])
            Y = self.model(X)
            
            # log probabilities of selected actions
            log_prob = Y[np.arange(T), A].log()
            
            # discounted rewards per timestep, standardized
            R_disc = (numpy_to_torch(R_disc) - R_mean) / R_std
            
            # loss is negative of the reward-weighted sum of log likelihood
            episode_loss = -torch.dot(log_prob, R_disc)
            episode_losses += episode_loss
            
        # backward pass
        self.optimizer.zero_grad()
        loss = episode_losses / N
        loss.backward()
        self.optimizer.step()
        
        # reset the replay history
        self.replay = []

    def copy(self):
        
        # create a copy of this agent with frozen weights
        agent = PolicyGradientNNAgent(lambda x: 0, 0, self.obs_to_input)
        agent.model = copy.deepcopy(self.model)
        agent.trainable = False
        for param in agent.model.parameters():
            param.requires_grad = False
            
        return agent

def train(iterations, episodes, verbose=False):
    
    def obs_to_input(o):
        # takes observation vector o and converts it into input for the network
        # TODO: standardize bids and info to [0, 1]
        third_last, second_last, last, WE_bid, NS_bid, points = obs
        bids = np.array([third_last, second_last, last])
        passes = (bids == 0)
        info = np.array([WE_bid, NS_bid, points])
        return np.hstack((bids, passes, info))
        
    n_layers = 2
    D_in = 9
    D_h = 25
    D_out = MAX_BID + 1
    params = n_layers, D_in, D_h, D_out
        
    new_model = nLayerNet
    new_agent = lambda : PolicyGradientNNAgent(new_model, params, obs_to_input, \
                                               lr=1e-3, df=0.9)
    
    env = BridgeEnv()
    start_name = 'West'
    env = BridgeEnv()
    agents = {
        'East': new_agent(),
        'West': new_agent(),
        'North': new_agent(),
        'South': new_agent()
    }
    teams = [['East', 'West'], ['North', 'South']]
    
    def copy_agents():
        # create a copy of the most recently updated agents with frozen weights
        return {name: agent.copy() for name, agent in agents.items()}
    
    agent_history = [copy_agents()]
    
    delta = 0.5
    def sample_teams(train_id, iter):
        # train_id is the current team being trained (0 or 1)
        # the opposing team is chosen from iteration i ~ Uniform[dv, v] where
        # v = iter (the current iteration)
        current_agents = {}
        i = np.random.randint(int(delta * iter), iter + 1)
        for name in agents:
            if name in teams[train_id]:
                current_agents[name] = agents[name]
            else:
                current_agents[name] = agent_history[i][name]
                
        return current_agents
    
    # main training loop
    for iter in range(iterations):
    
        # TODO: write eval function to print training progress to terminal
        # eval(agent_history[-1], iter)
        
        # train each team on its own rollout
        for train_id in np.random.permutation(len(teams)):
            
            # play out each episode
            for ep in range(episodes):
                
                current_agents = sample_teams(train_id, iter)
                observation = env.reset(start_player_name=start_name)
                for name in teams[train_id]:
                    current_agents[name].new_episode()
                    
                done = False
                while not done:
                    done, obs, info = check_game_status(env)

                    curr_agent = current_agents[info['cur_player']]
                    if verbose: env.show_turn(True)
                        
                    action = curr_agent.move(obs, env.invalid_moves())
                    obs, reward, done, info = env.step(action)
                    if verbose: env.render()
                    
                    if done and verbose: env.show_result(True)
                
                # TODO: once reward is fixed we should get rid of this line
                reward = {name: 0 for name in agents}
                
                # tell agents their reward at the end of the episode
                for name in teams[train_id]:
                    current_agents[name].store_reward(reward[name])
                
            # adjust agent parameters based on played episodes
            for name in teams[train_id]:
                current_agents[name].update()
            
        # update completed history
        agent_history.append(copy_agents())
    
    return agents

if __name__ == '__main__':
    
    iterations = 1
    episodes = 3
    trained_agents = train(iterations, episodes, verbose=True)
