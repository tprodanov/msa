import hmm
import fa
import eln
import argparse
import math


def get_intervals(seq):
    value = seq[0]
    begin = 0
    intervals = []
    for i in range(1, len(seq)):
        if seq[i] != value:
            intervals.append((begin, i - 1, value))
            value = seq[i]
            begin = i
    intervals.append((begin, len(seq) - 1, value))
    return intervals


def main():
    parser = argparse.ArgumentParser(prog='MSA 2')
    parser.add_argument('-f', '--fasta', type=argparse.FileType('r'))
    parser.add_argument('-m', '--hmm', type=argparse.FileType('r'))
    parser.add_argument('-o', '--output', type=argparse.FileType('w'))

    args = parser.parse_args(['-f', 'vibrio.chrII.fa', '-m', 'hmm.txt', '-o', 'output.txt'])
    input_seq = fa.read_fasta(args.fasta)[0][1]
    state = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    observations = [state[nt] for nt in input_seq]

    hmm_file = args.hmm
    model = hmm.Model.load(hmm_file)
    output = args.output
    viterbi(observations, model, output)
    output.flush()
    print('viterbi train')
    viterbi_train(observations, model, output)
    output.flush()

    print('baum')
    hmm_file.seek(0)
    model = hmm.Model.load(hmm_file)
    baum_welch(observations, model, output)
    print('Success')


def viterbi(observations, model, output):
    output.write('=============================\n')
    output.write('========== Viterbi ==========\n')
    output.write('=============================\n')

    states, probability = model.viterbi(observations)
    output.write('Log probability: %f\n' % probability)

    segments = get_intervals(states)
    output.write('Segments: %d\n' % len(segments))
    output.write('\n\n')
    return states


def viterbi_train(observations, model, output):
    output.write('=============================\n')
    output.write('======= Viterbi Train =======\n')
    output.write('=============================\n')

    states, probability = model.viterbi(observations)
    previous_probability = math.nan
    i = 0
    while math.isnan(previous_probability) or previous_probability + 0.1 < probability:
        i += 1
        output.write('After iteration %d\n' % i)

        model.viterbi_train(states)
        previous_probability = probability
        states, probability = model.viterbi(observations)
        output.write('\tLog probability: %f\n' % probability)
        output.write('\tSegments: %d\n' % len(get_intervals(states)))
        output.write('\tTransitions:\n')
        for outer in model.transitions:
            for transition in outer:
                output.write('\t\t%f' % eln.exp(transition))
            output.write('\n')
        output.flush()

    output.write('Segments:\n')
    i = 1
    for begin, end, value in get_intervals(states):
        output.write('\t%s\t%d\t%d\t%d\n' % ('CG' if value else 'AT', i, begin, end))
        i += 1
    output.flush()
    output.write('\n\n')


def baum_welch(observations, model, output):
    output.write('=============================\n')
    output.write('======== Baum-Welch =========\n')
    output.write('=============================\n')
    alpha, beta, gamma, probability = model.backward_forward(observations)
    previous_probability = math.nan

    i = 0
    while math.isnan(previous_probability) or previous_probability + 0.1 < probability:
        output.write('After iteration %d\n' % i)
        i += 1
        output.write('\tLog probability: %f\n' % probability)
        model.print(output, 1)
        output.flush()
        model.baum_welch(observations, alpha, beta, gamma, probability)

        previous_probability = probability
        alpha, beta, gamma, probability = model.backward_forward(observations)


if __name__ == '__main__':
    main()
