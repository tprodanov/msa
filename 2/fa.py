def read_fasta(fasta_in):
    current_name = next(fasta_in).strip('>\n ')
    current_seq = []
    seqs = []
    for line in fasta_in:
        if line[0] == '>':
            seqs.append((current_name, ''.join(current_seq)))
            current_name = line.strip('>\n ')
            current_seq = []
        else:
            current_seq.append(line.strip())
    seqs.append((current_name, ''.join(current_seq)))
    return seqs

