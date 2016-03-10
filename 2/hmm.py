import math
import eln
import sys

class Model:

    def __init__(self, hidden_states, emission_states):
        self.hidden_states = hidden_states
        self.emission_states = emission_states

        self.initial = [0.0] * self.hidden_states
        self.transitions = [[0.0] * self.hidden_states for _ in range(self.hidden_states)]
        self.emission = [[0.0] * self.emission_states for _ in range(self.hidden_states)]

    @staticmethod
    def load(file):
        next(file)
        hidden_states = int(next(file))
        next(file)
        emission_states = int(next(file))
        model = Model(hidden_states, emission_states)

        next(file)
        model.initial = [eln.ln(float(x)) for x in next(file).split()]
        next(file)
        for i in range(hidden_states):
            model.transitions[i] = [eln.ln(float(x)) for x in next(file).split()]
        next(file)
        for i in range(hidden_states):
            model.emission[i] = [eln.ln(float(x)) for x in next(file).split()]
        return model

    def print(self, file):
        file.write('Initial:\n')
        for initial in self.initial:
            file.write('\t%f' % eln.exp(initial))
        file.write('\n')

        file.write('Transitions:\n')
        for outer in self.transitions:
            for transition in outer:
                file.write('\t%f' % eln.exp(transition))
            file.write('\n')
        file.write('Emission:\n')
        for outer in self.emission:
            for emission in outer:
                file.write('\t%f' % eln.exp(emission))
            file.write('\n')

    def viterbi(self, observations):
        n = len(observations)
        v = [[math.nan] * self.hidden_states for _ in range(n)]
        prev = [[-1] * self.hidden_states for _ in range(n)]

        for i in range(self.hidden_states):
            v[0][i] = eln.product(self.initial[i], self.emission[i][observations[0]])

        for i in range(1, n):
            for j in range(self.hidden_states):
                for k in range(self.hidden_states):
                    tmp = eln.product(v[i - 1][k], self.transitions[k][j])
                    if eln.greater(tmp, v[i][j]):
                        v[i][j] = tmp
                        prev[i][j] = k
                v[i][j] = eln.product(v[i][j], self.emission[j][observations[i]])

        best = v[n - 1][0]
        best_i = 0
        for i in range(1, self.hidden_states):
            if eln.greater(v[n - 1][i], best):
                best = v[n - 1][i]
                best_i = i

        states = [0] * n
        states[n - 1] = best_i
        for i in range(n - 1, 0, -1):
            states[i - 1] = prev[i][states[i]]
        return states, best

    def viterbi_train(self, states):
        transitions = [[0] * self.hidden_states for _ in range(self.hidden_states)]
        for i in range(len(states) - 1):
            transitions[states[i]][states[i + 1]] += 1
        transitions_sum = [sum(row) for row in transitions]
        for i in range(self.hidden_states):
            for j in range(self.hidden_states):
                self.transitions[i][j] = float(transitions[i][j]) / transitions_sum[i]

    def backward_forward(self, observations):
        n = len(observations)
        alpha = [[math.nan] * self.hidden_states for _ in range(n)]
        for i in range(self.hidden_states):
            alpha[0][i] = eln.product(self.initial[i], self.emission[i][observations[0]])
        for t in range(1, n):
            for j in range(self.hidden_states):
                for k in range(self.hidden_states):
                    alpha[t][j] = eln.sum(alpha[t][j], eln.product(alpha[t - 1][k], self.transitions[k][j]))
                alpha[t][j] = eln.product(alpha[t][j], self.emission[j][observations[t]])

        beta = [[math.nan] * self.hidden_states for _ in range(n)]
        for i in range(self.hidden_states):
            beta[n - 1][i] = eln.ln(1)
        for t in range(n - 2, -1, -1):
            for j in range(self.hidden_states):
                for k in range(self.hidden_states):
                    beta[t][j] = eln.sum(beta[t][j],
                                         eln.product(beta[t + 1][k],
                                                     eln.product(self.transitions[j][k],
                                                                 self.emission[k][observations[t + 1]])))

        gamma = [[math.nan] * self.hidden_states for _ in range(n)]
        sequence_probability = math.nan
        for i in range(self.hidden_states):
            sequence_probability = eln.sum(sequence_probability, alpha[-1][i])

        for t in range(n):
            for j in range(self.hidden_states):
                gamma[t][j] = eln.product(eln.product(alpha[t][j], beta[t][j]), -sequence_probability)

        return alpha, beta, gamma

    def baum_welch(self, observations, alpha, beta, gamma):
        print('-------------')
        print('-------------')
        self.print(sys.stdout)

        eln.print_matrix(alpha, 'alpha')
        eln.print_matrix(beta, 'beta')
        eln.print_matrix(gamma, 'gamma')

        sequence_probability = math.nan
        for i in range(self.hidden_states):
            sequence_probability = eln.sum(sequence_probability, alpha[-1][i])

        n = len(observations)
        xi = [[[math.nan] * self.hidden_states for _ in range(self.hidden_states)] for _ in range(n - 1)]
        for t in range(n - 1):
            for i in range(self.hidden_states):
                for j in range(self.hidden_states):
                    xi[t][i][j] = eln.product(eln.product(alpha[t][i], self.transitions[i][j]),
                                              eln.product(beta[t + 1][j], self.emission[j][observations[t + 1]]))
                    xi[t][i][j] = eln.product(xi[t][i][j], -sequence_probability)

        for t in range(n - 1):
            eln.print_matrix(xi[t], 'xi[t = %d]' % t)

        for i in range(self.hidden_states):
            self.initial[i] = gamma[0][i]

        gamma_sum = [math.nan] * self.hidden_states
        for i in range(self.hidden_states):
            for t in range(1, n):
                gamma_sum[i] = eln.sum(gamma_sum[i], gamma[t][i])

        for i in range(self.hidden_states):
            for j in range(self.hidden_states):
                sum_xi = math.nan
                for t in range(n - 1):
                    sum_xi = eln.sum(sum_xi, xi[t][i][j])
                self.transitions[i][j] = eln.product(sum_xi, -gamma_sum[i])

        for i in range(self.hidden_states):
            gamma_sum[i] = eln.sum(gamma_sum[i], gamma[0][i])
            gamma_actual = [math.nan] * self.emission_states
            for t in range(n):
                gamma_actual[observations[t]] = eln.sum(gamma_actual[observations[t]], gamma[t][i])
            for j in range(self.emission_states):
                self.emission[i][j] = eln.product(gamma_actual[j], -gamma_sum[i])
