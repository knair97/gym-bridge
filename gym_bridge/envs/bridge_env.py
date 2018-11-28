import random
import numpy as np
import gym
from gym import spaces
import logging

MAX_BID = 52
MAX_POINTS = 52
PASS = 0

LOG_FMT = logging.Formatter('%(levelname)s '
                            '[%(filename)s:%(lineno)d] %(message)s',
                            '%Y-%m-%d %H:%M:%S')

# Should put these functions in the state class. I'm keeping them here
# just so that perhaps this code is slightly more compatible with the
# example code
def after_action_state(state, action):
    """
    Execute an action and returns resulted state.

    Args:
        state: (GameState): State of the game
        action (int): Action to run (a bid to make)

    Returns:
        GameState: New state of the game
    """

    state.make_bid(action)
    return state

def check_game_status(env):
    '''
    returns done: bool, True if game is don
            obs: current observation (6-tuple)
            additional_info: dict with name of current player
    '''
    info = {
        'cur_player': env.state.next_player.name,
        'throw_out': env.done and len(env.state.bid_history) == 4
    }
    return (env.done, env._get_obs(), info)

def reward_calc_helper(state):
    """
    Return game status by current game state. A game finishes when the last
    last bids are all passes.
    TODO: game status is represented in a way for an omniscient person. If 
    this status is supposed to be visible for the players, you have to figure
    out a better way of doing it

    Args:
        state (GameState): State of the game

    Returns:
        WE = sum of their points - last bid of West or East (whichever came last)
        NS = sum of their points - last bid of North or South (whichever came last)
        tuple<int, int, int>:
            (0, 0, 0): game thrown out because starting four bids were all pass
            (1, WE, NS): game in progress
            (2, WE, NS): West/East's final bid was higher
            (3, WE, NS): North/South's final bid was higher
    """
    we_final_bid, ns_final_bid = 0, 0
    
    hist = state.bid_history
    for i in range(len(hist)):
        if hist[i][1] == PASS:
            continue
        if hist[i][0].get_name() in ["West", "East"]:
            we_final_bid = hist[i][1]
        else:
            ns_final_bid = hist[i][1]

    WE = state.we_sum_points - we_final_bid
    NS = state.ns_sum_points - ns_final_bid

    if len(hist) < 4:
        return (1, WE, NS)
    
    if len(hist) == 4 and hist[0][1] == PASS and hist[1][1] == PASS \
        and hist[2][1] == PASS and hist[3][1] == PASS:
            return (0, 0, 0)

    if hist[-1][1] == PASS and hist[-2][1] == PASS and hist[-3][1] == PASS:
        if hist[-4][0].get_name() in ["West", "East"]:
            return (2, WE, NS)
        return (3, WE, NS)

    return (1, WE, NS)

class Card:
    def __init__(self, suit, value):
        """
        Suit - H(earts), D(iamonds), C(lubs), S(pades)
        Value - 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A
        """
        assert suit in ['H', 'D', 'C', 'S']
        assert str(value).upper() in \
            ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.suit = str(suit)
        self.value = str(value).upper()

    def get_suit(self):
        return self.suit

    def get_value(self):
        return self.value

    def __str__(self):
        """
        Print the card in a super pretty way
        """
        return self.get_value() + self.get_suit()


class Player:
    def __init__(self, name, cards):
        """
        Name - West, North, East, South
        cards - List of Cards
        """
        assert name in ["West", "North", "East", "South"]
        self.name = name
        sorted_cards = sorted(cards, \
            key = lambda card : (card.suit, card.value))
        self.cards = sorted_cards
        self.points = self.calc_points()

    def calc_points(self):
        """
        Calculate the number of points a player has based on their cards
        Assigns 1 point for a Jack, 2 for a Queen, 3 for a King, 4 for an Ace
        Assigns 3 points for each suit that has no cards,
                2 points for each suit that has 1 card, and
                1 point for each suit that has 2 Cards
        """
        suit_counts = {'H': 0, 'D': 0, 'C': 0, 'S': 0}
        high_card_counts = {'A': 0, 'K': 0, 'Q': 0, 'J': 0}
        for card in self.cards:
            suit, value = card.get_suit(), card.get_value()
            suit_counts[suit] += 1
            if value in high_card_counts.keys():
                high_card_counts[value] += 1
        points = 0
        for suit_count in suit_counts.values():
            if suit_count == 0:
                points += 3
            elif suit_count == 1:
                points += 2
            elif suit_count == 2:
                points += 1
        points += high_card_counts['A'] * 4 + high_card_counts['K'] * 3 \
                + high_card_counts['Q'] * 2 + high_card_counts['J'] * 1
        return points

    def get_name(self):
        return self.name

    def get_cards(self):
        return self.cards

    def get_points(self):
        return self.points

    def __str__(self):
        """
        Print the player in a dope way
        Note: str(card) implicitly calls __str__
        """
        return self.get_name() + ": " + \
            ", ".join([str(card) for card in self.get_cards()])


class GameState:
    def __init__(self, start_player_name="West"):
        """
        Players: {Player name: Player} e.g. {"West": Player_west, ...}
        Bid history: List of (Player, bid) e.g. [(Player, bid), ...]
        """
        self.reset(start_player_name)

    def get_points(self, player_name):
        return self.players[player_name].get_points()

    def reset(self, start_player_name):
        self.players = self.assign_cards_to_players()
        self.bid_history = []
        assert start_player_name in ["West", "North", "East", "South"]
        self.next_player = self.players[start_player_name]
        self.we_sum_points = self.players["West"].get_points() \
            + self.players["East"].get_points()
        self.ns_sum_points = self.players["North"].get_points() \
            + self.players["South"].get_points()

    def get_last_player(self):
        """
        Returns the player who played last or returns "Ye" if nobody
        played yet
        """
        if len(self.bid_history) == 0:
            return "Ye"
        return self.bid_history[-1][0]

    def assign_cards_to_players(self):
        """
        Randomly assign all cards to players
        """
        cards = []
        for suit in ['H', 'D', 'C', 'S']:
            for value in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']:
                cards.append(Card(suit, value))
        # seed the random generator earlier if you want to repeat trials
        # I think random seeds persist (so you can just seed in the main function)
        random.shuffle(cards)
        player_west = Player("West", cards[:13])
        player_north = Player("North", cards[13:26])
        player_east = Player("East", cards[26:39])
        player_south = Player("South", cards[39:])
        return {"West": player_west, "North": player_north, \
                "East": player_east, "South": player_south}
                
    def leading_bid(self):
        '''
        Returns the maximum bid made so far.
        If no (non-pass) bids have been made, returns -1.
        '''
        bids = [turn[1] for turn in self.bid_history if turn[1] != PASS]
        return max(bids) if bids else -1

    def make_bid(self, bid):
        """
        Given a bid, adds the bid to the bid history.
        Bid must be a possible sum of points in a team and so must be
        in the range (0, ..., MAX_BID). A player can also pass.
        Changes the next_player appropriately based on players moving
        clockwise
        """
        assert ((bid == PASS) or (bid <= MAX_BID and bid > self.leading_bid()))
        cur_player = self.next_player
        self.bid_history.append((cur_player, bid))
        # Cards are played clockwise (W -> N -> E -> S -> ...)
        next_player_dict = {"West": "North", "North": "East", \
                            "East": "South", "South": "West"}
        self.next_player = self.players[next_player_dict[cur_player.get_name()]]
        
    def invalid_bids(self):
        '''
        Return a list of bids that cannot be played.
        '''
        ret = list(range(self.leading_bid() + 1))
        if len(ret) != 0 and ret[0] == PASS:
            ret = ret[1:]

        return ret

    def __str__(self):
        if self.bid_history:
            bid = self.bid_history[-1]
            string = bid[0].get_name() + " bid: " + str(bid[1])
        else:
            names = ["West", "North", "East", "South"]
            string = '\n'.join([str(self.players[name]) for name in names])
        
        return string


class BridgeEnv(gym.Env):
    """
    TODO: This is likely very wrong - definitely look over the logic and format of it
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, start_player_name='West', seed=None):
        # See https://github.com/openai/gym/blob/master/gym/spaces/
        # for different types of spaces
        
        # In the action space:
        # 1 - 52: bid that number
        # 0: pass
        self.action_space = spaces.Discrete(MAX_BID + 1)
        
        # In the observation space:
        #   there are 3 tuples corresponding to the actions of the 3
        #   previous players, represented the same way as the action space
        #   The next two tuples correspond to the max bid of West/East, and
        #   North/South respectively
        #   The last tuple is the number of points of the current player
        # TODO:
        #   player also needs to observe their own point total
        #       -- but do we need to include this on every turn?
        self.observation_space = spaces.Tuple((   \
            spaces.Discrete(MAX_BID + 1),   \
            spaces.Discrete(MAX_BID + 1),   \
            spaces.Discrete(MAX_BID + 1),   \
            spaces.Discrete(MAX_BID + 1),   \
            spaces.Discrete(MAX_BID + 1),   \
            spaces.Discrete(MAX_POINTS + 1) \
        ))
        
        if seed:
            # TODO: somehow seed the card shuffling,
            # maybe pass as argument to GameState() constructor
            pass
        self.state = GameState(start_player_name=start_player_name)
        self.done = False
        
    def invalid_moves(self):
    
        return np.array(self.state.invalid_bids()).astype(np.int64)

    def sample_action(self):
        # Sample an action according to distribution the uniform distribution. 
        # Invalid moves are set to probability 0 before sampling.
        #
        # Returns:
        #   An action a in the action space
        
        pi = np.ones(MAX_BID + 1)
        pi[self.invalid_moves()] = 0
        
        pi = pi / np.sum(pi)
        a = np.random.choice(np.arange(MAX_BID + 1), p=pi)
        
        return a

    def reset(self, start_player_name='West'):
        self.state.reset(start_player_name)
        self.done = False

        return self._get_obs()

    def step(self, action):
        """Step environment by action.

        Args:
            action int: Bid (or 0 for pass)

        Returns:
            list: Observation
            int: Reward
            bool: Done
            dict: Additional information
        """
        assert self.action_space.contains(action)

        # not sure if this is the correct behavior...
        # how will the players other than the final bidder know their reward?
        # maybe it makes more sense to return reward as a tuple
        if self.done:
            return self._get_obs(), 0, True, None

        reward = 0
        prev_status = reward_calc_helper(self.state)
        
        # Make the bid
        self.state.make_bid(action)
        
        status = reward_calc_helper(self.state)
        # Idk if this works and hopefully we don't have to log for debugging
        # but again it's here for compatibility
        logging.debug("check_game_status state {} : status {}" \
            .format(self.state, status))
        # Kapil:
        # TODO: Not sure if reward should be calculated this way - check it
        # Reward = raise in their bid for the player who played last
        # if the cur_status > 0 (bid is less than the sum of their points)
        # negative of the sum of the team's points (maybe change this to some other negative number)
        # if the last person's bid sends them over the sum,
        # and if the last person passes, their reward is 0
        # NOT SURE if that's a good idea - if you're already above your sum
        # and you pass, should you have a negative reward? This depends on 
        # which team has the max bid rn too...
        
        # Bhairav: Here's how I think the reward should work, roughly based
        # on how actual bridge scoring works. This is a gross approximation b/c
        # actual bridge scoring is convoluted af. The main point is that in
        # bridge scoring, there are no negative points; if the team that won
        # the bidding fails to make their contract, then the opposing team
        # gets positive points.
        # Let   S = sum of the max bidder's points and their partner's points
        #       B = max bid
        # If S >= B, then
        #   the reward for the max bidding team is B + (S - B) / 2
        #       - Note: points for exceeding contract are awarded at discounted rate
        #       - this incentivizes bidding as close to S as possible
        #   the reward for the opposing team is 0
        # If S < B, then
        #   the reward for the max bidding team is 0
        #   the reward for the opposing team is 2 * (B - S)
        if status[0] != 1:
            self.done = True
        if status[0] == 1:
            if self.state.bid_history[-1][1] == PASS:
                reward = 0
            elif self.state.get_last_player().get_name() in ["West", "East"]:
                if status[1] > 0:
                    reward = status[1] - prev_status[1]
                else:
                    # shoulda made a getter but that's pedantic
                    reward = self.state.we_sum_points * -1
            else:
                if status[2] > 0:
                    reward = status[2] - prev_status[2]
                else:
                    reward = self.state.ns_sum_points * -1

        additional_info = {
            'cur_player': self.state.next_player.name
        }
        return self._get_obs(), reward, self.done, additional_info

    def _get_obs(self):
        """
        Returns the bids made by the last three players, max bid of West/East,
        max bid of North/South
        
        The returned vector always has length 3, so if the game just started
        this vector is padded with passes
        """
        history = self.state.bid_history
        pad = [PASS] * (3 - len(history))
        obs = pad + [turn[1] for turn in history[-3:]]


        we_final_bid, ns_final_bid = 0, 0
    
        hist = self.state.bid_history
        for i in range(len(hist)):
            if hist[i][1] == PASS:
                continue
            if hist[i][0].get_name() in ["West", "East"]:
                we_final_bid = hist[i][1]
            else:
                ns_final_bid = hist[i][1]

        # add max bid of West/East
        obs.append(we_final_bid)

        # add max bid of North/South
        obs.append(ns_final_bid)

        # add points for current player
        obs.append(self.state.next_player.get_points())

        assert self.observation_space.contains(obs)
        return obs

    def render(self, mode='human'):
        if mode == 'human':
            showfn = print
        else:
            showfn = logging.info
        showfn(str(self.state) + '\n')

    def show_turn(self, human):
        showfn = print if human else logging.info
        showfn("Player {}'s turn.".format(self.state.next_player.get_name()))

    def get_player_name(self):
        return self.state.next_player.get_name()

    def show_result(self, human):
        showfn = print if human else logging.info
        status = check_game_status(self)
        last_nopass_player = self.state.bid_history[-4][0].get_name()
        team_name = None
        assert status[0] == True
        if status[2]['throw_out']:
            showfn("==== Finished: Starting four bids were all passes ====")
        else:
            if last_nopass_player in ["West", "East"]:
                idx = 1 # index to the WE portion of status
                team_name = "West/East"
                max_bid = status[1][3]
                num_points = self.state.players['West'].get_points() + self.state.players['East'].get_points()
            else:
                idx = 2 # index to the NS portion of status
                team_name = "North/South"
                max_bid = status[1][4]
                num_points = self.state.players['North'].get_points() + self.state.players['South'].get_points()
            if max_bid < num_points:
                msg = "Team {0} wins bidding. They bid under their max winnable bid by {1} hands!" \
                    .format(team_name, num_points - max_bid)
            elif status[idx] == 0:
                msg = "Team {0} wins bidding and bids exactly their max winnable bid!" \
                    .format(team_name)
            else:
                msg = "Team {0} wins bidding. They bid over their max winnable bid by {1} hands!" \
                    .format(team_name, max_bid - num_points)

            showfn(msg + '\n')

        return team_name, self.state.bid_history[-4][1]

    def available_actions(self):
        highest_bid = -1
        for _, bid in self.state.bid_history:
            highest_bid = bid
        ret = list(range(highest_bid + 1, MAX_BID + 1))
        if len(ret) != 0 and ret[0] != PASS:
            ret = [PASS] + ret

        return ret


def set_log_level_by(verbosity):
    """Set log level by verbosity level.

    verbosity vs log level:

        0 -> logging.ERROR
        1 -> logging.WARNING
        2 -> logging.INFO
        3 -> logging.DEBUG

    Args:
        verbosity (int): Verbosity level given by CLI option.

    Returns:
        (int): Matching log level.
    """
    if verbosity == 0:
        level = 40
    elif verbosity == 1:
        level = 30
    elif verbosity == 2:
        level = 20
    elif verbosity >= 3:
        level = 10

    logger = logging.getLogger()
    logger.setLevel(level)
    if len(logger.handlers):
        handler = logger.handlers[0]
    else:
        handler = logging.StreamHandler()
        logger.addHandler(handler)

    handler.setLevel(level)
    handler.setFormatter(LOG_FMT)
    return level