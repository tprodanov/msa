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
    parser.add_argument('-u', '--human_output', type=argparse.FileType('w'))
    parser.add_argument('-o', '--output', type=argparse.FileType('w'))
    parser.add_argument('-m', '--hmm', type=argparse.FileType('r'))
    parser.add_argument('-s', '--segments', type=argparse.FileType('w'))

    args = parser.parse_args(['-f', '1.fa', '-u', 'output.txt',
                              '-o', 'output.csv', '-m', 'hmm1.txt', '-s', 'cg_segments.csv'])
    input_seq = fa.read_fasta(args.fasta)[0][1]

    observations = [0 if nucl == 'A' or nucl == 'T' else 1 for nucl in input_seq]
    # model = hmm.Model.load(args.hmm)
    human_output = args.human_output
    # viterbi(observations, model, human_output)
    # viterbi_train(observations, model, human_output, args.segments)

    model = hmm.Model.load(args.hmm)
    baum_welch(observations, model, human_output)



def viterbi(observations, model, human_output):
    states, probability = model.viterbi(observations)
    human_output.write('Viterbi\n')
    human_output.write('Log probability: %f\n' % probability)

    segments = get_intervals(states)
    human_output.write('Segments: %d\n' % len(segments))
    human_output.write('\n\n')
    return states


def viterbi_train(observations, model, human_output, segments):
    human_output.write('Viterbi train\n')

    states, probability = model.viterbi(observations)
    for i in range(10):
        human_output.write('After iteration %d\n' % (i + 1))
        model.viterbi_train(states)
        states, probability = model.viterbi(observations)
        human_output.write('\tLog probability: %f\n' % probability)
        human_output.write('\tSegments: %d\n' % len(get_intervals(states)))
        human_output.write('\tTransitions:\n')
        for outer in model.transitions:
            for transition in outer:
                human_output.write('\t%f' % eln.exp(transition))
            human_output.write('\n')

    segments.write('no\tbegin\tend\n')
    i = 0
    for begin, end, value in get_intervals(states):
        if value == 1:
            segments.write('%d\t%d\t%d\n' % (i, begin, end))
            i += 1


def baum_welch(observations, model, human_output):
    human_output.write('Baum-Welch\n')
    alpha, beta, gamma = model.backward_forward(observations)
    for i in range(3):
        human_output.write('After iteration %d\n' % i)
        probability = math.nan
        for gamma_i in gamma[-1]:
            probability = eln.sum(probability, gamma_i)
        human_output.write('\tLog probability: %f\n' % probability)
        model.baum_welch(observations, alpha, beta, gamma)

    model.print(human_output)


if __name__ == '__main__':
    main()
