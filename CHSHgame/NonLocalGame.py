from math import sqrt

import matplotlib.pyplot as plt
from qiskit.extensions import RYGate, RZGate, RXGate, IGate, CXGate
from sklearn.preprocessing import StandardScaler

from agents.BasicAgent import BasicAgent
from agents.DQNAgent import DQNAgent
from models.LinearModel import LinearModel


def get_scaler(env, N, ALL_POSSIBLE_ACTIONS, round_to=2):
    """:returns scikit-learn scaler object to scale the states"""
    # Note: you could also populate the replay buffer here
    states = []
    for _ in range(N):
        action = np.random.choice(ALL_POSSIBLE_ACTIONS)
        state, reward, done = env.step(action)
        states.append(np.round(state, round_to))

        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


def show_plot_of(plot_this, label, place_line_at=()):
    # plot relevant information
    fig_dims = (10, 6)

    fig, ax = plt.subplots(figsize=fig_dims)

    for pl in place_line_at:
        plt.axhline(y=pl, color='r', linestyle='-')

    plt.xlabel('Epochs')
    plt.ylabel(label)

    plt.plot(plot_this)
    plt.show()


def override(f): return f


from abc import ABC, abstractmethod


class abstractEnvironment(ABC):
    """ abstract environment to create CHSH framework """

    @abstractmethod
    def reset(self):
        """Return initial_time_step."""
        self.counter = 1
        self.history_actions = []
        self.state = self.initial_state.copy()
        self.accuracy = self.calc_accuracy([self.measure_analytic() for _ in range(self.n_questions)])
        # self.repr_state = np.array([x for _ in range(self.num_players ** 2) for x in self.state] + [len(self.history_actions)], dtype=np.float64)
        self.repr_state = np.array([x for _ in range(self.num_players ** 2) for x in self.state], dtype=np.float64)
        return self.repr_state

    @abstractmethod
    def step(self, action):
        """Apply action and return new time_step."""
        pass

    def measure_analytic(self):
        """ :returns probabilities of questions (e.g. 00,01,10,11) happening in matrix """
        weights = [abs(a) ** 2 for a in self.state]
        return weights

    def calc_accuracy(self, result):
        """ :returns winning accuracy / win rate based on winning game_type """
        win_rate = 0
        for x, riadok in enumerate(self.game_type):
            for y, stlpec in enumerate(riadok):
                win_rate += (stlpec * result[x][y])
        win_rate = win_rate * 1 / len(self.game_type)
        return win_rate

    def count_gates(self):
        """ :returns count of relevant gates """
        count = 0
        for action in self.history_actions:
            if action in {"xxr0"}:  # ending action
                pass
            elif action in {"smallerAngle", "biggerAngle"}:
                count += 0.5
            else:
                count += 1

        return count

    def get_gate(self, action):
        """ :returns gate got from string code of action """
        gate = action[2:4]
        if gate == "rx":
            return RXGate
        elif gate == "ry":
            return RYGate
        elif gate == "rz":
            return RZGate
        elif gate == "cx":
            return CXGate
        else:
            return IGate

    def reward_only_negative(self, difference):
        return -1

    def reward_only_difference(self, difference):
        # reward is the increase in accuracy
        return difference

    def reward_qubic(self, difference):
        return (difference ** 3) * 1000

    def reward_complex1(self, difference):
        reward = difference
        if np.round(reward, 5) <= np.round(0, 5):
            reward -= self.reward_only_negative(difference)
        else:
            reward += difference
        return reward

    def reward_complex2(self, difference):
        reward = self.reward_qubic(difference)
        if np.round(self.accuracy, 2) >= np.round(self.max_acc, 2):
            # done = True
            self.max_acc = self.accuracy
            reward += 5 * (1 / (self.count_gates() + 1)) * self.accuracy
        return reward

    def rewardOnlyBest(self, difference):
        reward = difference * 100

        # always award only the best (who is best changes through evolution)
        if np.round(self.accuracy, 2) > np.round(self.max_acc, 2):
            reward += 50 * (self.max_acc - self.accuracy)
            self.min_gates = len(self.history_actions)
            self.max_acc = self.accuracy
        elif np.round(self.accuracy, 2) == np.round(self.max_acc, 2):
            if self.min_gates > len(self.history_actions):
                self.min_gates = len(self.history_actions)

        # end when it has applied max number of gates / xxr0
        if self.counter == self.max_gates or self.history_actions[-1] == "xxr0":
            if np.round(self.max_acc, 2) == np.round(self.accuracy, 2) and self.min_gates == self.count_gates():
                reward = 5000 * (1 / (self.count_gates() + 1)) * self.accuracy
            elif np.round(self.max_acc, 2) == np.round(self.accuracy, 2):
                reward -= 1000 * (self.count_gates() + 1) / self.accuracy
            else:
                reward -= 10000 * (self.count_gates() + 1) / self.accuracy  # alebo tu dam tiez nejaky vzorcek
        return reward

    def reward_combined(self, difference):
        reward = difference
        # skonci, ak uz ma maximalny pocet bran
        if np.round(self.accuracy, 2) >= np.round(self.max_acc, 2):
            self.max_acc = self.accuracy
            if self.history_actions[-1] == "xxr0":
                reward += 40 * (1 / (self.count_gates() + 1)) * self.accuracy # alebo za count_gates len(history_actuons)
        # if self.counter == self.max_gates:
        #     reward += 50 * (1 / (self.count_gates() + 1))
        return reward


import random

import warnings

warnings.filterwarnings('ignore')
import pickle
import numpy as np


class Game:
    """ creates CHSH game framework for easier manipulation """

    def __init__(self, scaler=None, round_to=2, batch_size=32):
        self.scaler = scaler
        self.round_to = round_to
        self.batch_size = batch_size

    def play_one_episode(self, agent, env, DO):
        """ Plays one episode of CHSH training
        :returns last accuracy acquired and rewards from whole episode """
        # in this version we will NOT use "exploring starts" method
        # instead we will explore using an epsilon-soft policy
        state = env.reset()
        if self.scaler is not None: state = self.scaler.transform([state])
        else: state = np.array([np.around(state, self.round_to)], dtype=np.float64)
        done = False

        # be aware of the timing
        # each triple is s(t), a(t), r(t)
        # but r(t) results from taking action a(t-1) from s(t-1) and landing in s(t)

        rew_accum = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action[0])
            if self.scaler is not None: next_state = self.scaler.transform([np.around(next_state, self.round_to)])
            else: next_state = np.array([np.around(next_state, self.round_to)], dtype=np.float64)
            if DO == 'train':
                if type(agent) == BasicAgent:
                    agent.train(state.copy(), action[1], reward, next_state.copy(), done)
                elif type(agent) == DQNAgent:
                    agent.update_replay_memory(state.copy(), action[1], reward, next_state.copy(), done)
                    agent.replay(self.batch_size)
            state = next_state.copy()
            rew_accum += reward
        print(env.history_actions)
        return env.accuracy, rew_accum

    def evaluate_train(self, N, agent, env):
        """ Performes the whole training of agent in env in N steps
        :returns portfolio value and rewards for all episodes - serves to plot how it has trained"""
        DO = "train"

        portfolio_value = []
        rewards = []

        for e in range(N):
            val, rew = self.play_one_episode(agent, env, DO)
            print('episode:', end=' ')
            print(e, end=' ')
            print('acc:', end=' ')
            print(val)
            print('rew:', end=' ')
            print(rew)

            portfolio_value.append(val)  # append episode end portfolio value
            rewards.append(rew)

        # save the weights when we are done
        if DO == 'train':
            # # save the DQN

            agent.save(f'.training/linear.npz')

            # save the scaler
            with open(f'.training/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

        return portfolio_value, rewards

    def evaluate_test(self, agent, env):
        """ Tests what has the agent learnt in N=1 steps :returns accuracy and reward """
        DO = "test"

        portfolio_value = []
        if DO == 'test':
            N = 1

            # then load the previous scaler
            if self.scaler != None:
                with open(f'.training/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)

            # make sure epsilon is not 1!
            # no need to run multiple episodes if epsilon = 0, it's deterministic
            agent.epsilon = 0

            # load trained weights
            agent.load(f'.training/linear.npz')

        # play the game num_episodes times

        for e in range(N):
            val = self.play_one_episode(agent, env, DO)
            print('Test value:', end=' ')
            print(val)

            portfolio_value.append(val)  # append episode end portfolio value

        return portfolio_value


import itertools


def game_with_rows_all_zeroes(game):
    """ Controls whether there is not full zero row in game """
    for row in game:
        if 1 not in row or 0 not in row:
            return True
    return False


def generate_only_interesting_games(size=4, n_questions=2):
    """ Generates only interesting evaluation tactics
    because some are almost duplicates and some will have no difference between classic and quantum strategies. """
    product = list(itertools.product(list(range(n_questions)), repeat=size))
    games = list(itertools.product(product, repeat=size))
    print(len(games))
    if size != 4: return games  # this function works best only for size 4, in bigger scenarios its harder to tell which game is interesting
    interesting_games = dict()
    for game in games:
        if game_with_rows_all_zeroes(game): continue  # hry, ktore maju nulove riadky su nezaujimave tiez
        try:
            if interesting_games[(game[1], game[0], game[3], game[2])]: pass  # x za y, symetricke hry
        except KeyError:
            try:
                if interesting_games[(game[3], game[2], game[1], game[0])]: pass  # 0 za 1, symetricke hry
            except KeyError:
                interesting_games[game] = True

    print(len(interesting_games.keys()))
    return list(interesting_games.keys())


import NlgDeterministic


def play_deterministic(game, which="best"):
    """ Learns to play the best classic strategy according to game """
    env = NlgDeterministic.Environment(game)
    best, worst = env.play_all_strategies()
    return best, worst


import NlgDiscreteStatesActions


def play_quantum(game, which="best", agent_type=BasicAgent, n_qubits=2):
    """ Learns to play the best quantum strategy according to game """
    ACTIONS2 = ['r' + axis + str(180 / 16 * i) for i in range(1, 9) for axis in 'xyz']
    ACTIONS = ['r' + axis + str(- 180 / 16 * i) for i in range(1, 9) for axis in 'xyz']
    ACTIONS2.extend(ACTIONS)
    PERSON = ['a', 'b']
    QUESTION = ['0', '1']

    ALL_POSSIBLE_ACTIONS = [p + q + a for p in PERSON for q in QUESTION for a in ACTIONS2]
    ALL_POSSIBLE_ACTIONS.append("xxr0")
    ALL_POSSIBLE_ACTIONS.append("smallerAngle")
    ALL_POSSIBLE_ACTIONS.append("biggerAngle")

    N = 3000
    n_questions = 4
    max_gates = 9
    round_to = 2

    # learning_rates = [0.1, 1, 0.01]
    # gammas = [1, 0.9, 0.1]

    learning_rates = [0.1]
    gammas = [1]
    if n_qubits == 2:
        states = [np.array([0, 1 / sqrt(2), -1 / sqrt(2), 0], dtype=np.float64), np.array([1, 0, 0, 0], dtype=np.float64)]
    else:
        ALL_POSSIBLE_ACTIONS.append("a0cxnot")
        ALL_POSSIBLE_ACTIONS.append("b0cxnot")
        states = [np.array(
            [0 + 0j, 0 + 0j, 0.707 + 0j, 0 + 0j, -0.707 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j,
             0 + 0j])]

    best = 0
    worst = 1
    min_state = None
    max_state = None
    min_strategy = None
    max_strategy = None

    for state in states:
        for alpha in learning_rates:
            for gamma in gammas:
                env = NlgDiscreteStatesActions.Environment(n_questions=n_questions, game_type=game, max_gates=max_gates,
                                                           initial_state=state,
                                                           best_or_worst=which)  # mozno optimalnejsie by to bolo keby sa to resetovalo iba

                # (state_size, action_size, gamma, eps, eps_min, eps_decay, alpha, momentum)
                if agent_type == BasicAgent:
                    agent = BasicAgent(state_size=len(env.repr_state), action_size=len(ALL_POSSIBLE_ACTIONS), gamma=gamma, eps=1,
                                       eps_min=0.01,
                                       eps_decay=0.9995, alpha=alpha, momentum=0.9, ALL_POSSIBLE_ACTIONS=ALL_POSSIBLE_ACTIONS,
                                       model_type=LinearModel)

                else:
                    hidden_dim = [len(env.repr_state)]
                    agent = DQNAgent(state_size=len(env.repr_state), action_size=len(ALL_POSSIBLE_ACTIONS), gamma=gamma, eps=1, eps_min=0.01,
                                     eps_decay=0.9995, ALL_POSSIBLE_ACTIONS=ALL_POSSIBLE_ACTIONS, learning_rate=alpha, hidden_layers=len(hidden_dim),
                                     hidden_dim=hidden_dim)

                # scaler = get_scaler(env, N, ALL_POSSIBLE_ACTIONS, round_to=round_to)
                batch_size = 128

                # store the final value of the portfolio (end of episode)
                game_api = Game(round_to=round_to)
                portfolio_val = game_api.evaluate_train(N, agent, env)

                # save portfolio value for each episode
                np.save(f'.training/train.npy', portfolio_val)
                # portfolio_val = game.evaluate_test(agent, env)
                # return portfolio_val[0][0]  # acc

                load_acc = np.load(f'.training/train.npy')[0]
                load_acc_max = load_acc.max()
                load_acc_min = load_acc.min()

                # take the best found quantum, not just learned value
                if load_acc_max > best:
                    best = load_acc_max
                    max_strategy = env.max_found_strategy.copy()
                    max_state = env.max_found_state.copy()

                # take the best found quantum, not just learned value
                if load_acc_min < worst:
                    worst = load_acc_min
                    min_strategy = env.min_found_strategy.copy()
                    min_state = env.min_found_state.copy()

    return best, worst, min_state, max_state, min_strategy, max_strategy


def calc_difficulty_of_game(game):
    diff = 0
    for row in game:
        for x in row:
            if x == 1:
                diff += 1
    return diff


def to_list(tuple):
    return [list(x) for x in tuple]


def categorize(cutGames):
    categories = dict()
    for game in cutGames:
        classical_max_min = play_deterministic(game)
        if classical_max_min not in (0, 1):  # these are not interesting
            try:
                categories[classical_max_min][calc_difficulty_of_game(game)].append(to_list(game))
            except KeyError:
                try: categories[classical_max_min][calc_difficulty_of_game(game)] = [to_list(game)]
                except KeyError: categories[classical_max_min] = {calc_difficulty_of_game(game): [to_list(game)]}
    return categories


def Convert(list):
    categories = dict()
    for dict_row in list:
        try:
            categories[tuple(dict_row[0][0])][dict_row[1]].append(dict_row[2][0])
        except KeyError:
            try: categories[tuple(dict_row[0][0])][dict_row[1]] = [dict_row[2][0]]
            except: categories[tuple(dict_row[0][0])] = {dict_row[1]: [dict_row[2][0]]}
    return categories


from database import DB


def max_entangled_difference(n_players=2, n_questions=2, choose_n_games_from_each_category=5, best_or_worst="best", agent_type=BasicAgent,
                             n_qubits=2):
    """ Prints evaluation tactics that had the biggest difference between classical and quantum strategy """
    assert n_qubits == 2 or n_qubits == 4
    db = DB.CHSHdb()

    size_of_game = n_players * n_questions

    categories = db.query_categories_games(n_questions=n_questions, num_players=n_players)

    if categories == []:
        categories = categorize(generate_only_interesting_games(size_of_game))
        db.insert_categories_games(num_players=n_players, n_questions=n_questions, generated_games=categories)
    else:
        categories = Convert(categories)

    differences = []
    for category, difficulties in categories.items():
        for difficulty in difficulties.keys():
            for _ in range(choose_n_games_from_each_category):  # choose 10 tactics from each category randomly
                game_type = random.choice(categories[category][difficulty])
                classical_max, classical_min = play_deterministic(game_type, best_or_worst)
                quantum_max, quantum_min, min_state, max_state, min_strategy, max_strategy = play_quantum(game_type, best_or_worst,
                                                                                                          agent_type=agent_type, n_qubits=n_qubits)
                # quantum_max = 0

                difference_max = 0 if classical_max > quantum_max else quantum_max - classical_max
                difference_min = 0 if classical_min < quantum_min else quantum_min - classical_min
                differences.append(
                    (category, difficulty, classical_min, quantum_min, classical_max, quantum_max, game_type, difference_min, difference_max,
                     min_state.tolist(), max_state.tolist(), min_strategy, max_strategy))
        #     break
        # break

    # differences.sort(key=lambda x: x[1])  # sorts according to difference in winning rate
    for category, difficulty, classical_min, quantum_min, classical_max, quantum_max, game_type, difference_min, difference_max, min_state, max_state, min_strategy, max_strategy in differences:
        print()
        print("category: ", category)
        print("difficulty: ", difficulty)
        print("game = ")
        game_type = list(game_type)
        for i, row in enumerate(game_type):
            game_type[i] = list(game_type[i])
            print(row)
        print("difference_max = ", difference_max)
        print("difference_min = ", difference_min)
        print("max state = ", max_state)
        print("min state = ", min_state)
        print("max strategy = ", max_strategy)
        print("min strategy = ", min_strategy)
        print()

        db.insert(category=list(category), difficulty=difficulty, classic_min=classical_min, quantum_min=quantum_min, classic_max=classical_max,
                  quantum_max=quantum_max, difference_min=difference_min, difference_max=difference_max, min_state=min_state, max_state=max_state,
                  min_strategy=min_strategy, max_strategy=max_strategy, game=game_type)


if __name__ == '__main__':
    # max_entangled_difference(size=4)
    # game_type = [[1, 0, 0, 1],
    #                      [1, 0, 0, 1],
    #                      [1, 0, 0, 1],
    #                      [0, 1, 1, 0]]
    # print(play_deterministic(game_type))

    # print(len(generate_only_interesting_games(4)))

    # print([name for name, val in CHSHv02qDiscreteStatesActions.Environment.__dict__.items() if callable(val)])  # dostanem mena funkcii

    max_entangled_difference(choose_n_games_from_each_category=1, best_or_worst="best", agent_type=DQNAgent, n_qubits=2)
