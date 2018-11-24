#!/usr/bin/env python
import random
import numpy as np

from gym_bridge.envs.bridge_env import BridgeEnv, check_game_status,\
    after_action_state

MAX_BID = 52

class BaseAgent(object):
    def __init__(self, name):
        self.name = name

    def act(self, env):
        state = env.state
        ava_actions = env.available_actions() # this function needs to be fixed

        (gstatus, WE, NS) = check_game_status(state)

        team_bid = None
        if self.name in ['West', 'East']:
            team_bid = WE
        else:
            team_bid = NS

        curr_player_points = env.state.get_points(self.name)

        if team_bid >= curr_player_points + 30:
            return np.array([1, MAX_BID])
            
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
        last_3_bids = env.reset(start_player_name = start_name)
        
        done = False
        while not done:
            print(check_game_status(env.state))
            curr_agent = agents[env.get_player_name()]
            env.show_turn(True)

            action = curr_agent.act(env)
            print(action)
            last_3_bids, reward, done, info = env.step(action)
            env.render()

            if done:
                state_tuple = check_game_status(env.state)
                winning_team, bid = env.show_result(True)
                _, WE_score, NS_score = state_tuple
                WE_bid, NS_bid = 0, 0
                if winning_team == 'West/East' and WE_score >= 0:
                    WE_point_total += WE_bid
                elif winning_team == 'North/South' and NS_score >= 0:
                    NS_point_total += NS_score
                    
                break
        # rotate start
        start_name, _ = random.choice(list(agents.items()))
        episode += 1
    print('WE point total:', WE_point_total, 'NS point total:', NS_point_total)


if __name__ == '__main__':
    play()
