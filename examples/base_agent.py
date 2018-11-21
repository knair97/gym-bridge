#!/usr/bin/env python
import random

from bridge_game.env import BridgeGameEnv, agent_by_mark, check_game_status,\
    after_action_state, tomark, next_mark


class BaseAgent(object):
    def __init__(self, name):
        self.name = name

    def act(self, state, ava_actions, env):
        for action in ava_actions:
            nstate = after_action_state(state, action)
            # gstatus is going to be 1 (game in progress)
            (gstatus, WE, NS) = check_game_status(nstate)
            assert(gstatus == 1)

            team_bid = None
            if self.name in ['West', 'East']:
                team_bid = WE
            else:
                team_bid = NS

            curr_player_points = None
            for player in env.state.players:
                if player.name == self.name:
                    curr_player_points = player.points

            if team_bid >= curr_player_points + 20:
                return 'pass'
        return random.choice(ava_actions)


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
        state = env.reset(start_player_name = start_name)
        
        done = False
        while not done:
            curr_agent = agents[state.next_player.name]
            env.show_turn(True)

            ava_actions = env.available_actions()
            action = agent.act(state, ava_actions)
            state_tuple, reward, done, info = env.step(action)
            state = env.state
            env.render()

            if done:
                env.show_result(True)
                _, WE_score, NS_score = state_tuple
                WE_bid, NS_bid = 0, 0
                if state.get_last_player.name in ['East', 'West']:
                    WE_bid, NS_bid = env.state.bid_history[-1][1], env.state.bid_history[-2][1]
                else:
                    NS_bid, WE_bid = env.state.bid_history[-1][1], env.state.bid_history[-2][1]

                if WE_bid > NS_bid and WE_score >= 0:
                    WE_point_total += WE_bid
                elif WE_bid < NS_bid and NS_score >= 0:
                    NS_point_total += NS_score
                break
        # rotate start
        start_name, _ = random.choice(list(agents.items()))
        episode += 1
    print('WE point total:', WE_point_total, 'NS point total:', NS_point_total)


if __name__ == '__main__':
    play()
