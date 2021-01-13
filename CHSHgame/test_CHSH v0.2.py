import unittest
from CHSHv02 import Environment
import numpy as np
from qiskit.extensions import RYGate
from math import pi


class TestCHSH(unittest.TestCase):

    def testRYGate(self):
        assert (np.around(RYGate((0 * pi / 180)).to_matrix(), 5).all() == np.eye(2).all())

    def testIfCorrectStrategyAndAccuracy(self):
        n_questions = 4
        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]]
        max_gates = 10
        env = Environment(n_questions, tactic, max_gates)
        save_state = env.initial_state.copy()
        nauceneVyhodil = ['b0r-78.75', 'b0r-78.75', 'a0r90.0', 'b0r-78.75', 'b1r56.25', 'b1r-22.5', 'b0r11.25',
                          'b1r0.0', 'b1r0.0', 'b1r0.0']  # toto sa naucil
        dokopy = ['a0r90', 'b0r-225', 'b1r33.75']
        for a in dokopy:
            env.step(a)

        A_0 = np.kron(RYGate((90 * pi / 180)).to_matrix(), np.identity(2))
        A_1 = np.kron(np.identity(2), np.identity(2))
        B_0 = np.kron(np.identity(2), RYGate((-225 * pi / 180)).to_matrix())
        B_1 = np.kron(np.identity(2), RYGate((33.75 * pi / 180)).to_matrix())

        ax = np.array([
            *[x for x in np.matmul(B_0, np.matmul(A_0, save_state))],
            *[x for x in np.matmul(B_1, np.matmul(A_0, save_state))],
            *[x for x in np.matmul(B_0, np.matmul(A_1, save_state))],
            *[x for x in np.matmul(B_1, np.matmul(A_1, save_state))]
        ])
        print(ax)
        assert (env.accuracy > 0.85)
        for poc, state in enumerate(env.repr_state):
            if poc % 4 == 0:
                assert (np.round(
                    env.repr_state[poc] ** 2 + env.repr_state[poc + 1] ** 2 + env.repr_state[poc + 2] ** 2 +
                    env.repr_state[poc + 3] ** 2, 2) == 1)

        for poc, state in enumerate(ax):
            if poc % 4 == 0:
                assert (np.round(ax[poc] ** 2 + ax[poc + 1] ** 2 + ax[poc + 2] ** 2 + ax[poc + 3] ** 2, 2) == 1)

        assert (env.repr_state.all() == ax.all())

    def testInitialAccuracy(self):
        n_questions = 4
        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]]
        max_gates = 10
        env = Environment(n_questions, tactic, max_gates)
        assert (np.round(env.accuracy,2) == 0.25)

    # check if the other way of calculating accuracy is correct through comparing with already known good way, but inflexible
    def testCalcWinRate(self):
        n_questions = 4
        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]]
        max_gates = 10
        env = Environment(n_questions, tactic, max_gates)
        result = [env.measure_analytic() for i in range(4)]

        # this is for sure good way to calculate
        win_rate = 0
        for mat in result[:-1]:
            print(mat)
            win_rate += 1 / 4 * (mat[0] + mat[3])

        win_rate += 1 / 4 * (result[-1][1] + result[-1][2])
        assert (win_rate == env.calc_accuracy(env.tactic, result))

    def testCalcWinRate1(self):
        n_questions = 4
        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1]]
        max_gates = 10
        env = Environment(n_questions, tactic, max_gates)
        result = [env.measure_analytic() for i in range(4)]

        # this is for sure good way to calculate
        win_rate = 0
        for mat in result:
            print(mat)
            win_rate += 1 / 4 * (mat[0] + mat[3])

        assert (win_rate == env.calc_accuracy(env.tactic, result))

    def testCalcWinRate2(self):
        n_questions = 4
        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 0, 1]]
        max_gates = 10
        env = Environment(n_questions, tactic, max_gates)
        result = [env.measure_analytic() for i in range(4)]

        # this is for sure good way to calculate
        win_rate = 0
        for mat in result[:-1]:
            print(mat)
            win_rate += 1 / 4 * (mat[0] + mat[3])

        win_rate += 1 / 4 * (result[-1][1] + result[-1][3])

        assert (win_rate == env.calc_accuracy(env.tactic, result))


if __name__ == "__main__":
    unittest.main()
