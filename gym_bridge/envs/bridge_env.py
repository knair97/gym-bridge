import random
import gym
from gym import spaces
import logging

MAX_BID = 52
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

def check_game_status(state):
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
        if hist[i][1] == "pass":
            continue
        if hist[i][0].get_name() in ["West", "East"]:
            we_final_bid = hist[i][1]
        else:
            ns_final_bid = hist[i][1]

    WE = state.we_sum_points - we_final_bid
    NS = state.ns_sum_points - ns_final_bid

    if len(hist) < 4:
        return (1, WE, NS)
    
    if len(hist) == 4 and hist[0][1] == "pass" and hist[1][1] == "pass" \
        and hist[2][1] == "pass" and hist[3][1] == "pass":
            return (0, 0, 0)

    if hist[-1][1] == "pass" and hist[-2][1] == "pass" and hist[-3][1] == "pass":
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
        sorted_cards = sorted(self.cards, \
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
        return points

    def __str__(self):
        """
        Print the player in a dope way
        Note: str(card) implicitly calls __str__
        """
        return self.get_name() + ": " + \
            ", ".join([str(card) for card in self.get_cards()])


class GameState:
    def __init__(self, start_player_name = "West"):
        """
        Players: {Player name: Player} e.g. {"West": Player_west, ...}
        Bid history: List of (Player, bid) e.g. [(Player, bid), ...]
        """
        self.reset(start_player_name)

    def reset(self, start_player_name):
        self.players = self.players = self.assign_cards_to_players()
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

    def make_bid(self, bid):
        """
        Given a bid, adds the bid to the bid history.
        Bid must be a possible sum of points in a team and so must be
        in the range (0, ..., 52). A player can also pass.
        Changes the next_player appropriately based on players moving
        clockwise
        """
        assert (bid <= 52 and bid >= 0) | (bid == "pass")
        # TODO: check that the bid is higher than the last non-pass bid
        cur_player = self.next_player
        self.bid_history.append((cur_player, bid))
        # Cards are played clockwise
        next_player_dict = {"West": "North", "North": "East", \
                            "East": "South", "South": "West"}
        self.next_player = self.players[next_player_dict[cur_player.get_name()]]

    def __str__(self):
        string = "Player 1: " + str(self.players[0]) + "\n"
               + "Player 2: " + str(self.players[1]) + "\n"
               + "Player 3: " + str(self.players[2]) + "\n"
               + "Player 4: " + str(self.players[3]) + "\n"
        bid_history_string = ", ".join(["Player " + bid[0].get_name() \
                                        + "- bid: " + str(bid[1])])
        return string + bid_history_string


class BridgeEnv(gym.Env):
    """
    TODO: This is likely very wrong - definitely look over the logic and format of it
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, alpha=0.02, show_number=False, start_player_name = 'West'):
        # Discrete: https://github.com/openai/gym/blob/master/gym/spaces/discrete.py
        # Technically contains 0 in the action space - if this matters, I think you
        # can make a one-dimensional box that has a lower bound
        self.action_space = spaces.Discrete(MAX_BID)
        # Box: https://github.com/openai/gym/blob/master/gym/spaces/box.py
        self.observation_space = spaces.Box(low=np.array([0,0,0]), \
            high=np.array([3, MAX_BID, MAX_BID]))
        self.alpha = alpha
        self.state = GameState()
        self.done = False
        #self._seed()
        self._reset(start_player_name)

    '''
    # I don't know if these functions are useful for us so I commented them out
    # Use them to understand what the examples are doing
    
    def set_start_mark(self, mark):
        self.start_mark = mark

    '''

    def _reset(self, start_player_name):
        self.state.reset(start_player_name)

        return self._get_obs()
    

    def _step(self, action):
        """Step environment by action.

        Args:
            action (int): Location

        Returns:
            list: Observation
            int: Reward
            bool: Done
            dict: Additional information
        """
        assert self.action_space.contains(action)

        loc = action
        if self.done:
            return self._get_obs(), 0, True, None

        reward = 0
        prev_status = check_game_status(self.state)
        
        # Make the bid
        self.state.make_bid(action)
        status = check_game_status(self.state)
        # Idk if this works and hopefully we don't have to log for debugging
        # but again it's here for compatibility
        logging.debug("check_game_status state {} : status {}" \
            .format(self.state, status))
        # TODO: Not sure if reward should be calculated this way - check it
        # Reward = raise in their bid for the player who played last
        # if the cur_status > 0 (bid is less than the sum of their points)
        # negative of the sum of the team's points (maybe change this to some other negative number)
        # if the last person's bid sends them over the sum,
        # and if the last person passes, their reward is 0
        # NOT SURE if that's a good idea - if you're already above your sum
        # and you pass, should you have a negative reward? This depends on 
        # which team has the max bid rn too...
        if status[0] != 1:
            self.done = True
        if status[0] == 1:
            if state.bid_history[-1][1] == "pass":
                reward = 0
            elif state.get_last_player().get_name() in ["West", "East"]:
                if status[1] > 0:
                    reward = status[1] - prev_status[1]
                else:
                    # shoulda made a getter but that's pedantic
                    reward = state.we_sum_points * -1
            else:
                if status[2] > 0:
                    reward = status[2] - prev_status[2]
                else:
                    reward = state.ns_sum_points * -1

        return self._get_obs(), reward, self.done, None

    def _get_obs(self):
        """
        Returns the (num, WE, NS) based on the game status
        Perhaps return the game state in some way?
        Not sure what the observation space should be
        """
        return check_game_status(self.state)

    def _render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            self._show_state(print)  # NOQA
            print('')
        else:
            self._show_state(logging.info)
            logging.info('')

    def show_episode(self, human, episode):
        self._show_episode(print if human else logging.warning, episode)

    def _show_episode(self, showfn, episode):
        showfn("==== Episode {} ====".format(episode))

    def _show_state(self, showfn):
        """
        Display the state
        """
        showfn(str(self.state))

    def show_turn(self, human):
        self._show_turn(print if human else logging.info)

    def _show_turn(self, showfn):
        showfn("Player {}'s turn.".format(self.state.next_player.get_name()))

    def show_result(self, human):
        self._show_result(print if human else logging.info)

    def _show_result(self, showfn):
        status = check_game_status(self.state)
        last_nopass_player = self.state.bid_history[-4].get_name()
        assert status[0] != 1
        if status[0] == 0:
            showfn("==== Finished: Starting four bids were all passes ====")
        else:
            if last_nopass_player in ["West", "East"]:
                idx = 1 # index to the WE portion of status
                team_name = "West/East"
            else:
                idx = 2 # index to the NS portion of status
                team_name = "North/South"
            if status[idx] > 0:
                msg = "Team {0} wins bidding. They bid under their max winnable bid by {1} hands!" \
                    .format(team_name, status[idx])
            elif status[idx] == 0:
                msg = "Team {0} wins bidding and bids exactly thir max winnable bid!" \
                    .format(team_name)
            else:
                msg = "Team {0} wins bidding. They bid over their max winnable bid by {1} hands!" \
                    .format(team_name, -1 * status[idx])

    def available_actions(self):
        highest_bid = -1
        for _, bid in self.state.bid_history:
            if bid != "pass":
                highest_bid = bid
        return ['pass'] + list(range(highest_bid + 1, MAX_BID))


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