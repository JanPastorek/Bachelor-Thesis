import itertools
import random
from math import sqrt, pi

import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import IGate

from NLG.NonLocalGame import abstractEnvironment, override
from NLG.optimalizers.GeneticAlg import GeneticAlg


class CHSHgeneticOptimizer(GeneticAlg, abstractEnvironment):
    """ Creates genetic optimizer for nonlocal games. Supports N players and M questions. """

    @override
    def __init__(self, population_size=15, n_crossover=3, mutation_prob=0.05,
                 state=[0, float(1 / sqrt(2)), -float(1 / sqrt(2)), 0],
                 history_actions=['a0r0', 'b0r0', 'a1r0', 'b1r0'], game_type=[], num_players=2, n_questions=2, best_or_worst="best"):
        # Initialize the population - create population of 'size' individuals,
        # each individual is a bit string of length 'word_len'.
        super().__init__()
        self.n_questions = n_questions
        self.best_or_worst = best_or_worst
        self.population_size = population_size
        self.n_crossover = n_crossover
        self.mutation_prob = mutation_prob
        self.num_players = num_players
        self.n_players = num_players
        self.initial = state
        self.state = self.initial.copy()
        self.game_type = game_type
        self.n_games = 1

        self.n_qubits = self.n_qubits_from_state()

        self.reset(history_actions, n_crossover)

        # generate "questions" in equal number
        self.questions = list(itertools.product(list(range(self.n_questions)), repeat=self.num_players))

    @override
    def reset(self, history_actions, n_crossover):
        """ Initializes number of crossovers and environment with :param history_actions - new previous actions"""
        self.state = self.initial.copy()
        self.n_crossover = n_crossover
        n_question_combos = len(self.game_type) if len(self.game_type) > 0 else self.n_questions ** self.num_players
        self.repr_state = np.array([x for _ in range(n_question_combos) for x in self.state], dtype=np.complex128)
        self.history_actions = history_actions
        self.for_plot = []
        self.population = [self.generate_individual() for _ in range(self.population_size)]

    @override
    def step(self, action):
        pass

    @override
    def generate_individual(self):
        """Generate random individual."""
        # tieto hyperparametre treba optimalizovat - brany
        return [str(action[0:4]) + str(random.uniform(-180, 180)) if action != 'xxr0' else 'xxr0' for action in
                self.history_actions]

    @override
    def fitness(self, x):
        """ Returns fitness of a given individual. Supports N players. """
        result = []

        state_len = len(self.initial)
        n_question_combos = len(self.game_type) if len(self.game_type) > 0 else self.n_questions ** self.num_players

        for g, q in enumerate(self.questions):
            self.state = self.initial.copy()
            self.repr_state = np.array([x for _ in range(n_question_combos) for x in self.state], dtype=np.complex128)

            for action in x:
                gate = self.get_gate(action)
                if gate == IGate: continue
                to_whom = action[0:2]
                try: gate_angle = np.array([action[4:]], dtype=np.float64)
                except ValueError: gate_angle = 0

                # Decode player index and question number
                player_letter = action[0]
                question_num = int(action[1])
                player_idx = ord(player_letter) - ord('a')

                operation = []

                # Check if this action applies for this question combination
                if player_idx < len(q) and q[player_idx] == question_num:
                    gate_matrix = gate((gate_angle * pi / 180).item()).to_matrix()
                    # Build full operation: I_left ⊗ gate ⊗ I_right
                    # Convention: player k's qubit is at position k (from right/LSB)
                    left_dim = 2 ** (self.num_players - 1 - player_idx)
                    right_dim = 2 ** player_idx

                    if left_dim > 1 and right_dim > 1:
                        operation = np.kron(np.kron(np.identity(left_dim), gate_matrix), np.identity(right_dim))
                    elif left_dim > 1:
                        operation = np.kron(np.identity(left_dim), gate_matrix)
                    elif right_dim > 1:
                        operation = np.kron(gate_matrix, np.identity(right_dim))
                    else:
                        operation = gate_matrix

                if len(operation) != 0:
                    self.state = np.matmul(operation, self.state)

            self.repr_state[g * state_len:(g + 1) * state_len] = self.state.copy()

            result.append(self.measure_probabilities_analytically())
        fitness_individual = self.calc_accuracy(result)
        return fitness_individual

    @override
    def number_mutation(self, x, prob):
        """ Elements of x are real numbers [0.0 .. 1.0]. Mutate (i.e. add/substract random number)
         each number in x with given probabipity."""
        potomok = x
        for poc in range(len(potomok)):
            if random.random() <= prob:
                spocitaj = [float(gate[4:]) for gate in potomok]
                priemer = sum(spocitaj) / len(spocitaj)
                sigma_na_druhu = 0

                for i in spocitaj:
                    sigma_na_druhu += (i - priemer) ** 2

                sigma_na_druhu = sigma_na_druhu / (len(spocitaj)) / 360  # Normal distribution

                if random.random() > 0.5:
                    if potomok[poc] != 'xxr0':
                        nahodne = random.uniform(0, sigma_na_druhu)
                        potomok[poc] = potomok[poc][:4] + str(float(potomok[poc][4:]) - nahodne)

                else:
                    if potomok[poc] != 'xxr0':
                        nahodne = random.uniform(0, sigma_na_druhu)
                        potomok[poc] = potomok[poc][:4] + str(float(potomok[poc][4:]) + nahodne)

        return potomok

    @override
    def mutation(self, x, prob):
        return self.number_mutation(x, prob)

    @override
    def solve(self, max_generations, goal_fitness=1):
        """Implementation of genetic algorithm. Produce generations until some
        # individual`s fitness reaches goal_fitness, or you exceed total number
        # of max_generations generations. Return best found individual. """
        best = super().solve(max_generations, goal_fitness)
        accuracy = self.fitness(best)
        return best, accuracy, self.repr_state  # all is for best


if __name__ == "__main__":
    # Solve to find optimal individual
    ACTIONS = ['r' + axis + "0" for axis in 'y']
    PERSON = ['a', 'b']
    QUESTION = ['0', '1']

    ALL_POSSIBLE_ACTIONS = [p + q + a for p in PERSON for q in QUESTION for a in ACTIONS]  # place one gate at some place
    game = [[1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]]
    ga = CHSHgeneticOptimizer(population_size=30, n_crossover=len(ALL_POSSIBLE_ACTIONS) - 1, mutation_prob=0.1, history_actions=ALL_POSSIBLE_ACTIONS,
                              game_type=game, best_or_worst="best", state=np.array([0, 1 / sqrt(2), -1 / sqrt(2), 0], dtype=np.complex128))
    best = ga.solve(22)  # you can also play with max. generations
    ga.show_individual(best[0])
    print(best[1])

    fig_dims = (10, 6)

    fig, ax = plt.subplots(figsize=fig_dims)
    plt.axhline(y=0.853, color='r', linestyle='-')
    plt.axhline(y=0.75, color='r', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Win rate')

    plt.plot(ga.for_plot)
    plt.show()
