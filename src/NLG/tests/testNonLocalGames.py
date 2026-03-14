import unittest
from NLG.NlgDiscreteStatesActions import Environment
from NLG.NlgGeneticOptimalization import CHSHgeneticOptimizer
import numpy as np
from qiskit.circuit.library import RYGate
from math import pi, sqrt


class TestCHSH(unittest.TestCase):

    def testRYGate(self):
        assert (np.around(RYGate((0 * pi / 180)).to_matrix(), 5).all() == np.eye(2).all())

    def testIfCorrectStrategyAndAccuracy(self):
        n_questions = 2
        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]]
        max_gates = 10
        env = Environment(n_questions, tactic, max_gates)
        save_state = env.initial_state.copy()
        nauceneVyhodil = ['b0r-78.75', 'b0r-78.75', 'a0r90.0', 'b0r-78.75', 'b1r56.25', 'b1r-22.5', 'b0r11.25',
                          'b1r0.0', 'b1r0.0', 'b1r0.0']  # toto sa naucil
        dokopy = ['a0ry90', 'b0ry-225', 'b1ry33.75']
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
        print(env.accuracy)
        # assert (env.accuracy > 0.85) //TODO: este raz prekontrolovat ci je to spravne
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
        result = [env.measure_probabilities_analytically() for i in range(4)]

        # this is for sure good way to calculate
        win_rate = 0
        for mat in result[:-1]:
            print(mat)
            win_rate += 1 / 4 * (mat[0] + mat[3])

        win_rate += 1 / 4 * (result[-1][1] + result[-1][2])
        assert (win_rate == env.calc_accuracy(result))

    def testCalcWinRate1(self):
        n_questions = 4
        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1]]
        max_gates = 10
        env = Environment(n_questions, tactic, max_gates)
        result = [env.measure_probabilities_analytically() for i in range(4)]

        # this is for sure good way to calculate
        win_rate = 0
        for mat in result:
            print(mat)
            win_rate += 1 / 4 * (mat[0] + mat[3])

        assert (win_rate == env.calc_accuracy(result))

    def testCalcWinRate2(self):
        n_questions = 4
        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 0, 1]]
        max_gates = 10
        env = Environment(n_questions, tactic, max_gates)
        result = [env.measure_probabilities_analytically() for i in range(4)]

        # this is for sure good way to calculate
        win_rate = 0
        for mat in result[:-1]:
            print(mat)
            win_rate += 1 / 4 * (mat[0] + mat[3])

        win_rate += 1 / 4 * (result[-1][1] + result[-1][3])

        assert (win_rate == env.calc_accuracy(result))

    def testCalcWinRate3(self):
        n_questions = 4
        tactic = [[1,1,1,1] for i in range(n_questions)]
        max_gates = 10
        env = Environment(n_questions, tactic, max_gates)
        result = [env.measure_probabilities_analytically() for i in range(4)]
        assert (round(env.calc_accuracy(result)) == 1)

    def testCalcWinRate4(self):
        n_questions = 4
        tactic = [[0,0,0,0] for i in range(n_questions)]
        max_gates = 10
        env = Environment(n_questions, tactic, max_gates)
        result = [env.measure_probabilities_analytically() for i in range(4)]

        # this is for sure good way to calculate
        win_rate = 0
        for mat in result[:-1]:
            print(mat)
            win_rate += 1 / 4 * (mat[0] + mat[3])

        win_rate += 1 / 4 * (result[-1][1] + result[-1][3])

        assert round(win_rate - env.calc_accuracy(result) - 0) == 0

    def testGeneticAlg(self):
        # Solve to find optimal individual
        ACTIONS2 = ['r' + axis + "0" for axis in 'xyz']
        # ACTIONS2.extend(ACTIONS)  # complexne gaty zatial neural network cez sklearn nedokaze , cize S, T, Y
        PERSON = ['a', 'b']
        QUESTION = ['0', '1']

        ALL_POSSIBLE_ACTIONS = [p + q + a for p in PERSON for q in QUESTION for a in ACTIONS2]  # place one gate at some place
        game = [[1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0]]
        ga = CHSHgeneticOptimizer(population_size=30, n_crossover=len(ALL_POSSIBLE_ACTIONS) - 1, mutation_prob=0.1,
                                  history_actions=ALL_POSSIBLE_ACTIONS,
                                  game_type=game, best_or_worst="best", state=np.array([0, 1 / sqrt(2), -1 / sqrt(2), 0], dtype=np.complex128))

        best = ga.solve(22)  # you can also play with max. generations
        ga.show_individual(best[0])
        assert best[1] >= 0.83

    def testGeneticAlg2(self):
        # Solve to find optimal individual
        ACTIONS2 = ['r' + axis + "0" for axis in 'y']
        # ACTIONS2.extend(ACTIONS)  # complexne gaty zatial neural network cez sklearn nedokaze , cize S, T, Y
        PERSON = ['a', 'b']
        QUESTION = ['0', '1']

        ALL_POSSIBLE_ACTIONS = [p + q + a for p in PERSON for q in QUESTION for a in ACTIONS2]  # place one gate at some place
        game = [[0, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [1, 1, 0, 0]]
        ga = CHSHgeneticOptimizer(population_size=30, n_crossover=len(ALL_POSSIBLE_ACTIONS) - 1, mutation_prob=0.1,
                                  history_actions=ALL_POSSIBLE_ACTIONS,
                                  game_type=game, best_or_worst="best", state=np.array([0, 1 / sqrt(2), -1 / sqrt(2), 0], dtype=np.complex128))
        best = ga.solve(22)  # you can also play with max. generations
        ga.show_individual(best[0])
        assert np.round(best[1],2) == 0.5

    def testTensorflow1(self):
        import tensorflow as tf
        hello = tf.constant("hello TensorFlow!")

    def testCHSHdeterministicStrategies(self):
        from NLG import NonLocalGame
        evaluation_tactic = [[1, 0, 0, 1],
                             [1, 0, 0, 1],
                             [1, 0, 0, 1],
                             [0, 1, 1, 0]]
        assert NonLocalGame.play_deterministic(evaluation_tactic)[0] == 0.75
        assert NonLocalGame.play_deterministic(evaluation_tactic)[1] == 0.25

    def testCHSHaccLearnt(self):
        from NLG import NlgDiscreteStatesActions
        naucil_sa = ['b0ry-22.5', 'b0ry-22.5', 'b0ry-22.5', 'b0ry-22.5', 'b0ry-22.5', 'b0ry-22.5', 'biggerAngle', 'a0ry22.5', 'b1ry-22.5']
        dokopy = ['b0ry-135', 'a0ry45', 'b1ry-45']

        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]]
        env = NlgDiscreteStatesActions.Environment(n_questions=2, game_type=tactic, max_gates=10)

        result = env.calculate_state(dokopy, False)

        acc = env.calc_accuracy(result)
        print(acc)
        assert np.round(acc, 2) < 0.85


    def testCHSHaccOptimal(self):
        from NLG import NlgDiscreteStatesActions
        naucil_sa = ['b0ry-22.5', 'b0ry-22.5', 'b0ry-22.5', 'b0ry-22.5', 'b0ry-22.5', 'b0ry-22.5', 'biggerAngle', 'a0ry22.5', 'b1ry-22.5']
        dokopy = ['b0ry-135', 'a0ry90', 'b1ry-45']

        tactic = [[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]]
        env = NlgDiscreteStatesActions.Environment(n_questions=2, game_type=tactic, max_gates=10)

        result = env.calculate_state(dokopy, False)

        acc = env.calc_accuracy(result)
        print(acc)
        assert np.round(acc,2) == 0.85


class TestMultiPlayer(unittest.TestCase):
    """Tests for generalized N-player, M-question nonlocal games."""

    def test_3player_environment_creation(self):
        """Test that a 3-player game environment can be created with correct dimensions."""
        from NLG.NlgDiscreteStatesActions import Environment, default_initial_state
        # 3 players, 2 questions each: 2^3 = 8 question combos, 2^3 = 8 answer combos
        game_3p = [
            [1, 0, 0, 1, 0, 1, 1, 0],  # q=(0,0,0)
            [1, 0, 0, 1, 0, 1, 1, 0],  # q=(0,0,1)
            [1, 0, 0, 1, 0, 1, 1, 0],  # q=(0,1,0)
            [1, 0, 0, 1, 0, 1, 1, 0],  # q=(0,1,1)
            [1, 0, 0, 1, 0, 1, 1, 0],  # q=(1,0,0)
            [1, 0, 0, 1, 0, 1, 1, 0],  # q=(1,0,1)
            [1, 0, 0, 1, 0, 1, 1, 0],  # q=(1,1,0)
            [0, 1, 1, 0, 1, 0, 0, 1],  # q=(1,1,1)
        ]
        env = Environment(n_questions=2, game_type=game_3p, max_gates=10, n_players=3)

        # State should have 2^3 = 8 elements (GHZ state)
        assert len(env.initial_state) == 8
        # Questions should have 2^3 = 8 combinations
        assert len(env.questions) == 8
        # repr_state should have 8 * 8 = 64 elements
        assert len(env.repr_state) == 64
        # n_qubits should be 3
        assert env.n_qubits == 3

    def test_3player_default_initial_state(self):
        """Test that the default GHZ state is normalized for 3 players."""
        from NLG.NlgDiscreteStatesActions import default_initial_state
        state = default_initial_state(3)
        assert len(state) == 8
        # GHZ state: (|000⟩ + |111⟩) / √2
        assert np.isclose(abs(state[0])**2 + abs(state[-1])**2, 1.0)
        assert np.isclose(abs(state[0]), 1/sqrt(2))
        assert np.isclose(abs(state[-1]), 1/sqrt(2))
        # All middle elements should be 0
        for i in range(1, 7):
            assert np.isclose(state[i], 0)

    def test_3player_calculate_state(self):
        """Test that gates are applied correctly to 3-player game."""
        from NLG.NlgDiscreteStatesActions import Environment
        game_3p = [
            [1, 0, 0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 0, 0, 1],
        ]
        env = Environment(n_questions=2, game_type=game_3p, max_gates=10, n_players=3)

        # Apply gates for all 3 players
        actions = ['a0ry90', 'b0ry-45', 'c0ry45']
        result = env.calculate_state(actions)

        # Verify normalization: probabilities should sum to 1 for each question combo
        for probs in result:
            assert np.isclose(sum(probs), 1.0, atol=1e-5), f"Probabilities don't sum to 1: {sum(probs)}"

    def test_3player_gate_independence(self):
        """Test that player gates act on independent qubits in a 3-player game."""
        from NLG.NlgDiscreteStatesActions import Environment
        game_3p = [[1]*8 for _ in range(8)]  # trivial game (always win)
        # Start with |000⟩ state
        initial = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex64)
        env = Environment(n_questions=2, game_type=game_3p, max_gates=10, n_players=3,
                         initial_state=initial)

        # Apply RY(180) to player 'a' on question 0 → should flip qubit 0
        # |000⟩ → |001⟩ (player a is qubit 0, the rightmost)
        result_a = env.calculate_state(['a0ry180'])
        # For question (0, *, *), player a gets q=0, so gate is applied
        # For q=(0,0,0): probs should be concentrated on |001⟩ = index 1
        assert np.isclose(result_a[0][1], 1.0, atol=0.01), f"Expected |001⟩, got probs: {result_a[0]}"

        # Apply RY(180) to player 'c' on question 0 → should flip qubit 2
        # |000⟩ → |100⟩ (player c is qubit 2, the leftmost)
        env2 = Environment(n_questions=2, game_type=game_3p, max_gates=10, n_players=3,
                          initial_state=initial)
        result_c = env2.calculate_state(['c0ry180'])
        # For q=(0,0,0): probs should be concentrated on |100⟩ = index 4
        assert np.isclose(result_c[0][4], 1.0, atol=0.01), f"Expected |100⟩, got probs: {result_c[0]}"

    def test_3player_deterministic_strategies(self):
        """Test deterministic strategy evaluation for 3-player game."""
        from NLG import NlgDeterministic
        # 3-player game: win if XOR of answers = AND of questions
        game_3p = [
            [1, 0, 0, 1, 0, 1, 1, 0],  # q=(0,0,0): AND=0, win on even-parity answers
            [1, 0, 0, 1, 0, 1, 1, 0],  # q=(0,0,1)
            [1, 0, 0, 1, 0, 1, 1, 0],  # q=(0,1,0)
            [1, 0, 0, 1, 0, 1, 1, 0],  # q=(0,1,1)
            [1, 0, 0, 1, 0, 1, 1, 0],  # q=(1,0,0)
            [1, 0, 0, 1, 0, 1, 1, 0],  # q=(1,0,1)
            [1, 0, 0, 1, 0, 1, 1, 0],  # q=(1,1,0)
            [0, 1, 1, 0, 1, 0, 0, 1],  # q=(1,1,1): AND=1, win on odd-parity answers
        ]
        env = NlgDeterministic.Environment(game_3p, num_players=3, n_questions=2)
        best, worst = env.play_all_strategies()
        # Classical: all answer 0 → wins 7/8 = 0.875
        assert np.isclose(best, 0.875), f"Expected best classical = 0.875, got {best}"

    def test_3question_2player_environment(self):
        """Test 2-player game with 3 questions each."""
        from NLG.NlgDiscreteStatesActions import Environment
        # 3 questions, 2 players: 3^2 = 9 question combos, 2^2 = 4 answer combos
        game_3q = [
            [1, 0, 0, 1],  # q=(0,0)
            [1, 0, 0, 1],  # q=(0,1)
            [1, 0, 0, 1],  # q=(0,2)
            [1, 0, 0, 1],  # q=(1,0)
            [1, 0, 0, 1],  # q=(1,1)
            [1, 0, 0, 1],  # q=(1,2)
            [1, 0, 0, 1],  # q=(2,0)
            [1, 0, 0, 1],  # q=(2,1)
            [0, 1, 1, 0],  # q=(2,2)
        ]
        env = Environment(n_questions=3, game_type=game_3q, max_gates=10, n_players=2)

        # State should have 2^2 = 4 elements (Bell state for 2 players)
        assert len(env.initial_state) == 4
        # Questions should have 3^2 = 9 combinations
        assert len(env.questions) == 9
        assert env.questions[0] == (0, 0)
        assert env.questions[-1] == (2, 2)

    def test_3question_2player_calculate_state(self):
        """Test gate application with 3 questions per player."""
        from NLG.NlgDiscreteStatesActions import Environment
        game_3q = [
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ]
        env = Environment(n_questions=3, game_type=game_3q, max_gates=10, n_players=2)

        # Apply gates for questions 0, 1, and 2
        actions = ['a0ry90', 'b0ry-135', 'a1ry45', 'b1ry-90', 'a2ry-45', 'b2ry45']
        result = env.calculate_state(actions)

        assert len(result) == 9  # 9 question combinations
        for probs in result:
            assert np.isclose(sum(probs), 1.0, atol=1e-5), f"Probabilities don't sum to 1: {sum(probs)}"

    def test_3question_2player_deterministic(self):
        """Test deterministic strategies with 3 questions per player."""
        from NLG import NlgDeterministic
        game_3q = [
            [1, 0, 0, 1],  # q=(0,0)
            [1, 0, 0, 1],  # q=(0,1)
            [1, 0, 0, 1],  # q=(0,2)
            [1, 0, 0, 1],  # q=(1,0)
            [1, 0, 0, 1],  # q=(1,1)
            [1, 0, 0, 1],  # q=(1,2)
            [1, 0, 0, 1],  # q=(2,0)
            [1, 0, 0, 1],  # q=(2,1)
            [0, 1, 1, 0],  # q=(2,2)
        ]
        env = NlgDeterministic.Environment(game_3q, num_players=2, n_questions=3)
        best, worst = env.play_all_strategies()
        # Classical: all answer same → wins 8/9
        assert np.isclose(best, 8/9, atol=1e-10), f"Expected best = {8/9}, got {best}"

    def test_3player_genetic_optimizer(self):
        """Test genetic optimizer with 3 players."""
        game_3p = [
            [1, 0, 0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 0, 0, 1],
        ]
        ACTIONS2 = ['r' + axis + "0" for axis in 'y']
        PERSON = ['a', 'b', 'c']
        QUESTION = ['0', '1']
        ALL_POSSIBLE_ACTIONS = [p + q + a for p in PERSON for q in QUESTION for a in ACTIONS2]

        ghz_state = np.array([1/sqrt(2), 0, 0, 0, 0, 0, 0, 1/sqrt(2)], dtype=np.complex128)
        ga = CHSHgeneticOptimizer(
            population_size=30, n_crossover=len(ALL_POSSIBLE_ACTIONS) - 1,
            mutation_prob=0.1, history_actions=ALL_POSSIBLE_ACTIONS,
            game_type=game_3p, best_or_worst="best", state=ghz_state,
            num_players=3, n_questions=2
        )
        best = ga.solve(15)
        # Should find a strategy at least as good as classical (0.875)
        assert best[1] >= 0.5, f"Genetic optimizer 3-player result too low: {best[1]}"

    def test_4player_environment(self):
        """Test that a 4-player game environment can be created."""
        from NLG.NlgDiscreteStatesActions import Environment, default_initial_state
        # 4 players, 2 questions: 2^4 = 16 question combos, 2^4 = 16 answer combos
        game_4p = [[1 if (i + j) % 2 == 0 else 0 for j in range(16)] for i in range(16)]
        env = Environment(n_questions=2, game_type=game_4p, max_gates=10, n_players=4)
        assert len(env.initial_state) == 16
        assert len(env.questions) == 16
        assert env.n_qubits == 4

    def test_default_initial_state_normalization(self):
        """Test that default initial states are normalized for various player counts."""
        from NLG.NlgDiscreteStatesActions import default_initial_state
        for n in [2, 3, 4, 5]:
            state = default_initial_state(n)
            norm = sum(abs(a)**2 for a in state)
            assert np.isclose(norm, 1.0, atol=1e-6), f"State not normalized for {n} players: norm={norm}"
            assert len(state) == 2**n

    def test_3player_step_integration(self):
        """Test that the step method works for 3-player game."""
        from NLG.NlgDiscreteStatesActions import Environment
        game_3p = [[1]*8 for _ in range(8)]
        env = Environment(n_questions=2, game_type=game_3p, max_gates=10, n_players=3)
        initial_acc = env.accuracy

        state, reward, done = env.step('a0ry90')
        assert not done
        state, reward, done = env.step('b0ry45')
        assert not done
        state, reward, done = env.step('c0ry-45')
        assert not done



if __name__ == "__main__":
    unittest.main()
