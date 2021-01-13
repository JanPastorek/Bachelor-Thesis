import random
from math import sqrt, pi

import numpy as np
from qiskit.extensions import RYGate
from sklearn.preprocessing import StandardScaler


def get_scaler(env,N):
    # return scikit-learn scaler object to scale the states
    # Note: you could also populate the replay buffer here

    states = []
    for _ in range(N):
        choice = np.random.randint(0, len(ALL_POSSIBLE_ACTIONS) - 1)
        action = ALL_POSSIBLE_ACTIONS[choice],choice
        state, reward, done = env.step(action)
        states.append(state)

        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler

class LinearModel:
    """ A linear regression model """

    def __init__(self, input_dim, n_action):
        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        self.b = np.zeros(n_action)

        # momentum terms
        self.vW = 0
        self.vb = 0

        self.losses = []

    def predict(self, X):
        # make sure X is N x D
        assert (len(X.shape) == 2)
        return X.dot(self.W) + self.b

    def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
        # make sure X is N x D
        assert (len(X.shape) == 2)

        # the loss values are 2-D
        # normally we would divide by N only
        # but now we divide by N x K
        num_values = np.prod(Y.shape)

        # do one step of gradient descent
        # we multiply by 2 to get the exact gradient
        # (not adjusting the learning rate)
        # i.e. d/dx (x^2) --> 2x
        Yhat = self.predict(X)

        # print([X.shape, Y.shape])
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values

        # update momentum terms
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        # update params
        self.W += self.vW
        self.b += self.vb

        mse = np.mean((Yhat - Y) ** 2)
        self.losses.append(mse)

    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)

class Environment:

    def __init__(self, n_questions, tactic, max_gates):
        self.pointer = 0  # time
        self.n_questions = n_questions
        self.counter = 1
        self.history_actions = ['a0r0' for i in range(max_gates)]
        self.max_gates = max_gates
        self.tactic = tactic
        self.initial_state = np.array([0, 1 / sqrt(2), 1/ sqrt(2), 0], dtype=np.longdouble)
        self.state = self.initial_state.copy()
        self.accuracy = 0.25
        self.num_players = 2
        self.repr_state = [0 for i in range(max_gates)]

        # input, generate "questions" in equal number
        self.a = []
        self.b = []
        for x in range(2):
            for y in range(2):
                self.a.append(x)
                self.b.append(y)

    def reset(self):
        self.counter = 1
        self.history_actions = ['a0r0' for i in range(self.max_gates)]
        self.accuracy = 0.25
        self.state = self.initial_state.copy() ########## INITIAL STATE
        self.repr_state = [0 for i in range(self.max_gates)]
        return self.repr_state

    # Returns probabilities of 00,01,10,10 happening in matrix
    def measure_analytic(self):
        weights = [abs(a) ** 2 for a in self.state]
        return weights

    # Calculates winning accuracy / win rate based on winning tactic
    def calc_accuracy(self, tactic, result):
        win_rate = 0
        for x, riadok in enumerate(tactic):
            for y, stlpec in enumerate(riadok):
                win_rate += (stlpec * result[x][y])
        win_rate = win_rate * 1 / 4
        return win_rate

    def step(self, action):

        # Alice and Bob win when their input (a, b)
        # and their response (s, t) satisfy this relationship.
        done = False

        # play game
        result = []
        self.repr_state[self.counter - 1] = action[1]
        self.history_actions[self.counter - 1] = action[0]

        for g in range(self.n_questions):
            # Alice - a and Bob - b share an entangled state
            # The input to alice and bob is random
            # Alice chooses her operation based on her input, Bob too - eg. a0 if alice gets 0 as input

            self.state = self.initial_state.copy() ########## INITIAL STATE

            for action in self.history_actions:
                gate = np.array([action[3:]],dtype=np.longdouble)

                if self.a[g] == 0 and action[0:2] == 'a0':
                    self.state[:4] = np.matmul(np.kron(RYGate((gate * pi / 180).item()).to_matrix(), np.identity(2)),
                                               self.state[:4])

                if self.a[g] == 1 and action[0:2] == 'a1':
                    self.state[:4] = np.matmul(np.kron(RYGate((gate * pi / 180).item()).to_matrix(), np.identity(2)),
                                               self.state[:4])

                if self.b[g] == 0 and action[0:2] == 'b0':
                    self.state[:4] = np.matmul(np.kron(np.identity(2), RYGate((gate * pi / 180).item()).to_matrix()),
                                               self.state[:4])

                if self.b[g] == 1 and action[0:2] == 'b1':
                    self.state[:4] = np.matmul(np.kron(np.identity(2), RYGate((gate * pi / 180).item()).to_matrix()),
                                               self.state[:4])

            result.append(self.measure_analytic())

        # for i in result:
        #     print(i)

        # accuracy of winning CHSH game
        before = self.accuracy
        self.accuracy = self.calc_accuracy(self.tactic, result)

        # reward is the increase in accuracy
        rozdiel_acc = self.accuracy - before
        reward = rozdiel_acc * 100

        # skonci, ak uz ma maximalny pocet bran alebo presiahol pozadovanu uroven self.accuracy
        if self.accuracy >= 0.83:
            done = True
            self.counter = 1
            reward += 500 * (1 / (len(self.history_actions) + 1))

        elif self.counter == self.max_gates:
            done = True
            self.counter = 1
            # reward = -100

        print("acc: ", end="")
        print(self.accuracy)

        print("rew: ", end="")
        print(reward)

        if done == False:
            self.counter += 1
        return sorted(self.repr_state), reward, done


class Agent:
    def __init__(self, state_size, action_size, gamma, eps, eps_min, eps_decay, alpha, momentum):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount rate
        self.epsilon = eps  # exploration rate
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay
        self.alpha = alpha
        self.momentum = momentum
        self.model = LinearModel(state_size, action_size)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            choice = random.randint(0, self.action_size - 1)
            return ALL_POSSIBLE_ACTIONS[choice], choice
        act_values = self.model.predict(state)
        choice = np.argmax(act_values[0])
        return ALL_POSSIBLE_ACTIONS[choice], choice

    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1)

        target_full = self.model.predict(state)
        target_full[0, action] = target

        # Run one training step
        self.model.sgd(state, target_full, self.alpha, self.momentum)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pickle


class Game:

    def __init__(self, scaler):
        self.scaler = scaler

    def play_one_episode(self, agent, env, N, is_train):
        # returns a list of states and corresponding returns
        # in this version we will NOT use "exploring starts" method
        # instead we will explore using an epsilon-soft policy
        state = env.reset()
        state = self.scaler.transform([state])
        done = False

        # be aware of the timing
        # each triple is s(t), a(t), r(t)
        # but r(t) results from taking action a(t-1) from s(t-1) and landing in s(t)

        rew_accum = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            # next_state = [s.real for s in next_state]
            # print(state)
            next_state = self.scaler.transform([next_state])
            if is_train == 'train':
                agent.train(state, action[1], reward, next_state, done)
            state = next_state.copy()
            rew_accum += reward
        print(env.history_actions)
        return env.accuracy, rew_accum

    def evaluate_train(self, N, agent, env):
        co = "train"

        portfolio_value = []
        rewards = []

        for e in range(N):
            val, rew = self.play_one_episode(agent, env, N, co)
            print('episode:', end=' ')
            print(e, end=' ')
            print('acc:', end=' ')
            print(val)

            portfolio_value.append(val)  # append episode end portfolio value
            rewards.append(rew)

        # save the weights when we are done
        if co == 'train':
            # # save the DQN
            agent.save(f'linear.npz')

            # save the scaler
            with open(f'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

        return portfolio_value, rewards

    def evaluate_test(self, agent, n_questions, tactic, max_gates):
        co = "test"
        N = 1
        portfolio_value = []
        if co == 'test':


            # then load the previous scaler
            with open(f'scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)

            # remake the env with test data
            env = Environment(n_questions, tactic, max_gates)

            # make sure epsilon is not 1!
            # no need to run multiple episodes if epsilon = 0, it's deterministic
            agent.epsilon = 0

            # load trained weights
            agent.load(f'linear.npz')

        # play the game num_episodes times

        for e in range(N):
            val = self.play_one_episode(agent, env, N, co)
            print('episode:', end=' ')
            print(e, end=' ')
            print('value:', end=' ')
            print(val)

            portfolio_value.append(val)  # append episode end portfolio value

        return portfolio_value


ACTIONS2 = ['r' + str(180/16 * i) for i in range(0,9)]
ACTIONS = ['r' + str(- 180/16 * i) for i in range(1,9)]
ACTIONS2.extend(ACTIONS) # complexne gaty zatial neural network cez sklearn nedokaze , cize S, T, Y
PERSON = ['a', 'b']
QUESTION = ['0', '1']

ALL_POSSIBLE_ACTIONS = [p + q + a for p in PERSON for q in QUESTION for a in ACTIONS2] # place one gate at some place

N = 6000
n_questions = 4
tactic = [[1, 0, 0, 1],
          [1, 0, 0, 1],
          [1, 0, 0, 1],
          [0, 1, 1, 0]]
max_gates = 10

env = Environment(n_questions,tactic, max_gates)

# (state_size, action_size, gamma, eps, eps_min, eps_decay, alpha, momentum)
agent = Agent(len(env.repr_state),len(ALL_POSSIBLE_ACTIONS), 0.9 , 1 ,0,  0.995, 0.001 , 0.9)
scaler = get_scaler(env, N)
batch_size = 128

# store the final value of the portfolio (end of episode)
game = Game(scaler)
portfolio_value, rewards = game.evaluate_train(N, agent, env)

# plot relevant information
fig_dims = (10, 6)

fig, ax = plt.subplots(figsize=fig_dims)
plt.xlabel('Epochs')
plt.ylabel('Reward')

plt.plot(rewards)
plt.show()

plt.axhline(y=0.853, color='r', linestyle='-')
plt.axhline(y=0.75, color='r', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Win rate')
plt.plot(portfolio_value)
plt.show()
# save portfolio value for each episode
np.save(f'train.npy', portfolio_value)

portfolio_value = game.evaluate_test(agent, n_questions, tactic, max_gates)
print(portfolio_value)

a = np.load(f'train.npy')

print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")


plt.plot(a)
plt.show()