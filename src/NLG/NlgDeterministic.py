import itertools

from NLG import NonLocalGame


class Environment(NonLocalGame.abstractEnvironment):
    """ Creates environment for classic deterministic strategies.
    Supports N players and M questions. """

    def __init__(self, game_type, num_players=2, n_questions=2):
        self.num_players = num_players
        self.n_questions = n_questions
        # Question combinations: each player gets a question from 0..n_questions-1
        self.questions = list(itertools.product(range(self.n_questions), repeat=self.num_players))

        self.n_games = 1
        self.n_qubits = 0

        self.game_type = game_type

        # Each player can answer 0 or 1 (binary answers)
        self.possible_answers = {q: (0, 1) for q in range(self.n_questions)}

        # All possible answer combinations (one binary answer per player)
        self.responses = list(itertools.product(range(2), repeat=self.num_players))

    @NonLocalGame.override
    def reset(self):
        return

    @NonLocalGame.override
    def step(self, action):
        return

    def index(self, response):
        """ :returns index of response so that it can be mapped to state"""
        counter = 0
        for r in self.responses:
            if r == response:
                break
            counter += 1
        return counter

    def evaluate(self, question, response):
        """ :returns winning accuracy to input question based on response.
        Works for N players: question and response are tuples of length num_players. """
        n_answer_combinations = len(self.game_type[0]) if len(self.game_type) > 0 else 2 ** self.num_players
        self.state = [0 for _ in range(n_answer_combinations)]
        answer = tuple(self.possible_answers[question[p]][response[p]] for p in range(self.num_players))
        self.state[self.index(answer)] = 1
        return self.measure_probabilities_analytically()

    def play_all_strategies(self):
        """ Plays all possible deterministic strategies for N players, evaluates each
        and :returns: the best and worst accuracy from all strategies.
        Each player's strategy maps each question to a binary answer. """
        accuracies = []

        # Each player's strategy: a tuple of length n_questions mapping question -> answer (0 or 1)
        all_single_player_strategies = list(itertools.product(range(2), repeat=self.n_questions))

        # All combinations of strategies for all N players
        for strategy_combo in itertools.product(all_single_player_strategies, repeat=self.num_players):
            result = []
            for question in self.questions:
                # Each player answers according to their strategy
                response_to_this_question = tuple(
                    strategy_combo[p][question[p]] for p in range(self.num_players)
                )
                result.append(self.evaluate(question, response_to_this_question))
            accuracies.append(self.calc_accuracy(result))

        return max(accuracies), min(accuracies)

    def response_rek(self, n):
        if (n == 0): pass
        else:
            for r in self.responses:
                yield r
                self.response_rek(n - 1)



def rule(a, b, x, y):
    return (a != b) == (x and y)


def create(game_type):
    game = [[0 for _ in range(len(game_type)) for __ in range(len(game_type))] for ___ in range(len(game_type)) for ____ in range(len(game_type))]
    for y1, riadok1 in enumerate(game_type):
        for x1, cell1 in enumerate(riadok1):
            for y2, riadok2 in enumerate(game_type):
            # for x1, cell1 in enumerate(riadok1):
                for x2, cell2 in enumerate(riadok2):
                    if (cell1 == cell2 and cell1 == 1): game[y1 * y2][x1 * x2] = 1
    return game


if __name__ == '__main__':
    game_type = [[1, 0, 0, 1],
                 [1, 0, 0, 1],
                 [1, 0, 0, 1],
                 [0, 1, 1, 0]]
    env = Environment(game_type, 2, 2)
    print(env.play_all_strategies())
