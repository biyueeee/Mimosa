
import os, sys, re


root_path = os.getcwd()


def reverse_seq(seq):
    rseq = ''
    for i in range(len(seq)):
        rseq += seq[len(seq) - 1 - i]
    return rseq


def read_file(inputfile):
    if os.path.exists(inputfile) == False:
        print('Error: file " %s " does not exist.' % inputfile)
        sys.exit(1)
    with open(inputfile) as f:
        record = f.readlines()
    if len(record) > 50:
        print('Error: the numer of miRNA-mRNA pairs in the file " %s " must be less than 50!' % inputfile)
        sys.exit(1)

    data = []
    count = 0
    for line in record:
        tmp = line.strip('\n').split('\t')
        mirna_name = tmp[0]
        mirna_seq = tmp[1].upper().replace('T', 'U')
        mrna_name = tmp[2]
        mrna_seq = reverse_seq(tmp[3]).upper().replace('T', 'U')
        mrna_seq_cut = mrna_seq
        if len(mrna_seq) > 10000:
            count += 1
            mrna_seq_cut = mrna_seq[:10000]
        pair = [mirna_name, mirna_seq, mrna_name, mrna_seq_cut]
        data.append(pair)

    if count > 1:
        print('Warning: some mRNA sequences will be truncated less than 10,000 nt.')
    return data
