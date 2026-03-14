import itertools
import random
from math import sqrt, pi

import numpy as np
from qiskit.circuit.library import IGate, CXGate
from sklearn.preprocessing import OneHotEncoder

from NLG import NonLocalGame
from NLG.NonLocalGame import Game
from NLG.agents.DQNAgent import DQNAgent


def default_initial_state(n_players):
    """Generate a default entangled initial state for n_players.
    For 2 players: Bell singlet state |Ψ⁻⟩ = (|01⟩ - |10⟩) / √2
    For N players (N>2): GHZ state (|00...0⟩ + |11...1⟩) / √2
    """
    if n_players == 2:
        return np.array([0, 1 / sqrt(2), -1 / sqrt(2), 0], dtype=np.complex64)
    else:
        state = np.zeros(2 ** n_players, dtype=np.complex64)
        state[0] = 1 / sqrt(2)   # |00...0⟩
        state[-1] = 1 / sqrt(2)  # |11...1⟩
        return state


class Environment(NonLocalGame.abstractEnvironment):
    """ Creates nonlocal game environments for quantum strategies.
    Supports N players and M questions.
    Discretizes states and uses discrete actions. """

    def __init__(self, n_questions, game_type, max_gates, n_players=2,
                 initial_state=None, best_or_worst="best", reward_function=None,
                 anneal=False, n_games=1):
        self.n_games = n_games  # how many games are to be played (paralel)
        self.n_questions = n_questions  # how many atomic questions (to one player)
        self.n_players = n_players # players / verifiers
        self.counter = 1   # number of steps
        self.history_actions = []  # history of actions taken
        self.history_actions_anneal = [] # history of action annealed taken

        self.max_gates = max_gates # limit of gates that can be taken
        self.min_gates = 0
        self.game_type = game_type  # game matrix (rules - when do they win)

        # Default initial state: Bell state for 2 players, GHZ state for N players
        if initial_state is None:
            initial_state = default_initial_state(n_players)
        self.initial_state = initial_state  # initial state
        self.state = self.initial_state.copy()

        self.n_qubits = self.n_qubits_from_state()
        self.reduce_by = 2 ** (self.n_qubits - 2)  # reducing for double games

        self.possible_states = list(   # possible states
            itertools.product(list(range(self.n_questions)),
                              repeat=self.n_qubits))

        self.one_game_answers = list(  # possible answers
            itertools.product(list(range(2)),
                              repeat=self.n_players))

        self.repr_state = np.array([x for _ in range(len(self.game_type)) for x in self.state], dtype=np.complex64) # state representation for all comb. of questions

        self.state_size = len(self.repr_state) * 2  # times 2 because of complex array to array of real numbers

        self.accuracy = self.calc_accuracy([self.measure_probabilities_analytically() for _ in range(len(self.game_type))]) # winning probability
        self.max_acc = self.accuracy
        self.min_acc = self.accuracy

        self.max_found_state = self.repr_state.copy()   # best / worst found configurations
        self.max_found_strategy = []
        self.min_found_state = self.repr_state.copy()
        self.min_found_strategy = []
        self.best_or_worst = best_or_worst

        # Combinations of questions: each player gets a question from 0..n_questions-1
        self.questions = list(itertools.product(list(range(self.n_questions)), repeat=self.n_players))
        print(self.questions)
        self.memory_state = dict() # memoization of calculation, repr_state, accuracies
        self.reward_funcion = reward_function
        if self.reward_funcion == None: self.reward_funcion = self.reward_only_difference

        # Immutable actions (not real gates)
        cx_actions = set()
        for p in range(n_players):
            pl = chr(ord('a') + p)
            for qi in range(n_questions):
                cx_actions.add(f"{pl}{qi}cxnot")
                cx_actions.add(f"{pl}{qi}cxnotr")
        self.immutable = {"xxr0", "smallerAngle", "biggerAngle"} | cx_actions

        self.use_annealing = anneal # do you want to use annealing?

    @NonLocalGame.override
    def reset(self):
        self.history_actions_anneal = []
        return self.complex_array_to_real(super().reset())

    def _build_player_operation(self, player_idx, gate_matrix):
        """Build the full-system operation matrix for applying a gate to a specific player's qubit(s).

        Convention: player k's qubit(s) are at position k (counting from right/LSB side).
        For 1 qubit per player: player 0 (a) is rightmost, player 1 (b) is next, etc.
        """
        qubits_per_player = self.n_qubits // self.n_players
        player_dim = 2 ** qubits_per_player

        # Kronecker product: I_left ⊗ gate_matrix ⊗ I_right
        left_dim = player_dim ** (self.n_players - 1 - player_idx)
        right_dim = player_dim ** player_idx

        if left_dim > 1 and right_dim > 1:
            return np.kron(np.kron(np.identity(left_dim), gate_matrix), np.identity(right_dim))
        elif left_dim > 1:
            return np.kron(np.identity(left_dim), gate_matrix)
        elif right_dim > 1:
            return np.kron(gate_matrix, np.identity(right_dim))
        else:
            return gate_matrix

    def calculate_state(self, history_actions, anneal=False):
        """ Calculates the state according to previous actions in parameter history_actions.
        Supports N players and M questions. """
        result = []

        for g, q in enumerate(self.questions):
            self.state = self.initial_state.copy()

            for action in history_actions:
                # decode action
                gate = self.get_gate(action)
                if gate == IGate: continue

                player_letter = action[0]
                question_num = int(action[1])
                player_idx = ord(player_letter) - ord('a')
                rotate_ancilla = action[2] == 'a'
                try: gate_angle = np.array([action[4:]], dtype=np.float32)
                except ValueError: gate_angle = 0

                qubits_per_player = self.n_qubits // self.n_players

                # apply action to state
                operation = []

                # Check if this action applies for this question combination
                if player_idx < len(q) and q[player_idx] == question_num:
                    if gate == CXGate:
                        ctrl = int(action[-1] != "r")
                        gate_matrix = CXGate(ctrl_state=ctrl).to_matrix()
                    else:
                        gate_matrix = gate((gate_angle * pi / 180).item()).to_matrix()

                        # For multi-qubit per player, expand single-qubit gate
                        if qubits_per_player > 1:
                            if rotate_ancilla:
                                gate_matrix = np.kron(gate_matrix, np.identity(2 ** (qubits_per_player - 1)))
                            else:
                                gate_matrix = np.kron(np.identity(2 ** (qubits_per_player - 1)), gate_matrix)

                    operation = self._build_player_operation(player_idx, gate_matrix)

                if len(operation) != 0:
                    self.state = np.matmul(operation, self.state)

            # modify repr_state according to state
            self.repr_state[g * len(self.state):(g + 1) * len(self.state)] = self.state.copy()

            result.append(self.measure_probabilities_analytically())

        return result

    def save_interesting_strategies(self):
        if self.accuracy > self.max_acc:
            self.max_acc = self.accuracy
            self.max_found_state = self.repr_state.copy()
            self.max_found_strategy = self.history_actions_anneal.copy()

        elif self.accuracy == self.max_acc:
            if len(self.history_actions) < len(self.max_found_strategy):
                self.max_found_state = self.repr_state.copy()
                self.max_found_strategy = self.history_actions_anneal.copy()

        if self.accuracy < self.min_acc:
            self.min_acc = self.accuracy
            self.min_found_state = self.repr_state.copy()
            self.min_found_strategy = self.history_actions_anneal.copy()

        elif self.accuracy == self.min_acc:
            if len(self.history_actions) < len(self.min_found_strategy):
                self.min_found_state = self.repr_state.copy()
                self.min_found_strategy = self.history_actions_anneal.copy()

        if self.min_found_strategy == []: self.min_found_strategy.append('xxr0')
        if self.max_found_strategy == []: self.max_found_strategy.append('xxr0')

    @NonLocalGame.override
    def step(self, action):
        # Alice and Bob win when their input (a, b)
        # and their response (s, t) satisfy this relationship.
        done = False

        if type(action) == list: action = action[0]
        # play game
        self.history_actions.append(action)
        self.history_actions_anneal.append(action)

        # accuracy of winning CHSH game
        before = self.accuracy

        try:
            result, self.repr_state, _, self.accuracy, to_complex = self.memory_state[tuple(self.history_actions)]
        except KeyError:
            try: result, self.repr_state, self.history_actions_anneal[:-1] = self.memory_state[tuple(self.history_actions[:-1])][:3]
            except KeyError: pass
            if action not in self.immutable and self.use_annealing:
                self.history_actions_anneal[-1] = self.history_actions_anneal[-1][:4] + str(
                    self.anneal())  # simulated annealing on the last chosen action

            if self.use_annealing: result = self.calculate_state(self.history_actions_anneal)
            else: result = self.calculate_state(self.history_actions)

            self.accuracy = self.calc_accuracy(result)
            to_complex = self.complex_array_to_real(self.repr_state)
            self.memory_state[tuple(self.history_actions)] = (
                result, self.repr_state.copy(), self.history_actions_anneal.copy(), self.accuracy, to_complex)

        difference_in_accuracy = self.accuracy - before

        if self.best_or_worst == "worst": difference_in_accuracy *= (-1)

        try: reward = self.reward_funcion(self, difference_in_accuracy)  # because I needed to call like this when using Optimalizing hyperparam.
        except: reward = self.reward_funcion(difference_in_accuracy)

        self.save_interesting_strategies()

        if self.counter == self.max_gates or action == 'xxr0': done = True
        if not done: self.counter += 1
        return to_complex, reward, done

    def anneal(self, steps=80, t_start=2, t_end=0.001):
        """ Finds the maximal value of the fitness function by
        executing the simulated annealing algorithm.
        Returns a state (e.g. x) for which fitness(x) is maximal. """
        x = self.random_state()
        t = t_start
        for i in range(steps):
            neighbor = np.random.choice(self.neighbors(x))
            ΔE = self.fitness(neighbor) - self.fitness(x)
            if ΔE > 0:  # //neighbor is better then x
                x = neighbor
            elif np.random.random() < np.math.e ** (ΔE / t):  # //neighbor is worse then x
                x = neighbor
            t = t_start * (t_end / t_start) ** (i / steps)
        return x

    def fitness(self, x):
        """ Calculates fitness of the state given by calculation of accuracy over history of actions."""
        last = [self.history_actions_anneal[-1][:4] + str(x)]
        return self.calc_accuracy(self.calculate_state(self.history_actions_anneal[:-1] + last, anneal=True))

    def neighbors(self, x, span=30, delta=0.5):
        """ Creates neighboring gate angle to angle x"""
        res = []
        if x > -span + 3 * delta: res += [x - i * delta for i in range(1, 4)]
        if x < span - 3 * delta: res += [x + i * delta for i in range(1, 4)]
        return res

    def random_state(self):
        return random.uniform(-180, 180)


import warnings

warnings.filterwarnings('ignore')
from NLG.NonLocalGame import show_plot_of

if __name__ == '__main__':
    # Hyperparameters setting
    # ACTIONS = [q + axis + "0" for axis in 'xyz' for q in 'ra']
    ACTIONS = [q + axis + "0" for axis in 'y' for q in 'r']
    PERSON = ['a', 'b']
    QUESTION = ['0', '1']

    ALL_POSSIBLE_ACTIONS = [[p + q + a] for p in PERSON for q in QUESTION for a in ACTIONS]  # place one gate at some place
    ALL_POSSIBLE_ACTIONS.append(["xxr0"])

    # # for 1 game with 2 EPR
    # ALL_POSSIBLE_ACTIONS.append(["a0cxnot"])
    # ALL_POSSIBLE_ACTIONS.append(["b0cxnot"])
    #
    # # for xor paralel with 2EPR
    # ALL_POSSIBLE_ACTIONS.append(["a0cxnotr"])
    # ALL_POSSIBLE_ACTIONS.append(["b0cxnotr"])

    N = 4000
    n_questions = 2
    game_type = [[1, 0, 0, 1],
                 [1, 0, 0, 1],
                 [1, 0, 0, 1],
                 [0, 1, 1, 0]]

    max_gates = 15
    round_to = 6

    # game_type = create(game_type)

    state = np.array([0, 1 / sqrt(2), -1 / sqrt(2), 0], dtype=np.complex64)
    # state = np.array([ 0+0j, 0+0j, 0+0j, 0.5+0j, 0+0j, 0+0j, -0.5+0j, 0+0j, 0+0j, -0.5+0j, 0+0j, 0+0j, 0.5+0j, 0+0j, 0+0j, 0+0j ], dtype=np.complex64)
    #

    env = Environment(n_questions, game_type, max_gates, initial_state=state,
                      reward_function=Environment.reward_only_difference,
                      anneal=True, n_games=1)



    # transform actions to noncorellated encoding
    encoder = OneHotEncoder(drop='first', sparse=False)
    # transform data
    onehot = encoder.fit_transform(ALL_POSSIBLE_ACTIONS)
    onehot_to_action = dict()
    action_to_onehot = dict()
    for x, a_encoded in enumerate(onehot):
        onehot_to_action[str(a_encoded)] = x
        action_to_onehot[x] = str(a_encoded)

    hidden_dim = [len(env.repr_state) , len(env.repr_state), len(env.repr_state)]
    agent = DQNAgent(state_size=env.state_size, action_size=len(ALL_POSSIBLE_ACTIONS), gamma=0.9, eps=1, eps_min=0.01,
                     eps_decay=0.9998, ALL_POSSIBLE_ACTIONS=ALL_POSSIBLE_ACTIONS, learning_rate=0.001, hidden_layers=len(hidden_dim),
                     hidden_dim=hidden_dim, onehot_to_action=onehot_to_action, action_to_onehot=action_to_onehot)
    # divide data by
    batch_size = 128

    game = Game(round_to=round_to, batch_size=batch_size)
    portfolio_value, rewards = game.evaluate_train(N, agent, env)

    # agent = DQNAgent(state_size=env.state_size, action_size=len(ALL_POSSIBLE_ACTIONS), gamma=1, eps=1, eps_min=0.01,
    #                  eps_decay=0.9998, ALL_POSSIBLE_ACTIONS=ALL_POSSIBLE_ACTIONS, learning_rate=0.001, hidden_layers=len(hidden_dim),
    #                  hidden_dim=hidden_dim, onehot_to_action=onehot_to_action, action_to_onehot=action_to_onehot)

    # The size of a batch must be more than or equal to one and less than or equal to the number of samples in the training dataset.


    # plot relevant information
    show_plot_of(rewards, "reward")

    if agent.model.losses is not None:
        show_plot_of(agent.model.losses, "loss")

    show_plot_of(portfolio_value, "accuracy", [0.85, 0.75])

    # save portfolio value for each episode
    np.save(f'.training/train.npy', portfolio_value)

    portfolio_value = game.evaluate_test(agent, env)
    print(portfolio_value)
    a = np.load(f'.training/train.npy')
    print(f"average accuracy: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")
