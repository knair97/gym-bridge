#!/usr/bin/env python
import random
import numpy as np

from gym_bridge.envs.bridge_env import BridgeEnv, check_game_status,\
    after_action_state
    
PASS = 0
MAX_BID = 52

class BaseAgent(object):
    def __init__(self, name):
        self.name = name

    def act(self, obs, env):
        third_last, second_last, last, WE_bid, NS_bid, points = obs

        team_bid = None
        if self.name in ['West', 'East']:
            team_bid = WE_bid
        else:
            team_bid = NS_bid

        curr_player_points = points

        if team_bid >= curr_player_points + 30:
            return PASS
            
        return env.sample_action()


def play(max_episode=10):
    episode = 0
    start_name = 'West'
    env = BridgeEnv()
    agents = {
        'East': BaseAgent('East'),
        'West': BaseAgent('West'),
        'North': BaseAgent('North'),
        'South': BaseAgent('South')
    }

    WE_point_total = 0
    NS_point_total = 0
    while episode < max_episode:
        obs = env.reset(start_player_name = start_name)
        
        done = False
        while not done:
            done, obs, info = check_game_status(env)

            curr_agent = agents[info['cur_player']]
            env.show_turn(True)

            action = curr_agent.act(obs, env)
            obs, reward, done, info = env.step(action)
            env.render()

            if done:
                done, obs, info = check_game_status(env)
                winning_team, bid = env.show_result(True)
                
                third_last, second_last, last, WE_bid, NS_bid, points = obs
                if winning_team == 'West/East':
                    WE_point_total += WE_bid
                elif winning_team == 'North/South':
                    NS_point_total += NS_bid
                    
                break
        # rotate start
        start_name, _ = random.choice(list(agents.items()))
        episode += 1
    print('WE point total:', WE_point_total, 'NS point total:', NS_point_total)


if __name__ == '__main__':
    play()
