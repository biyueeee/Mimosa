import numpy as np


def segmentation_for_mRNA(mrna, step_size):
    kmers = []
    #step_size = 5
    if len(mrna) >= 40:
        for i in range(0, len(mrna), step_size):
            if i + 40 <= len(mrna):
                cut = mrna[i:i + 40]
                kmers.append(cut)
        return kmers
    else:
        pad_mrna = mrna + 'X' * (40 - len(mrna))
        kmers.append(pad_mrna)
        return kmers



def get_start_stop_codon(seq,count_start):
    '''seq 就是全长序列；start是计算密码子的起始位置，可输入0，1，2'''
    Number_codon = len(seq)//3
    stop = []
    for i in range(Number_codon):
        codon = seq[count_start+i*3:(count_start+i*3+3)]
        if codon == 'TAG' or codon == 'TAA' or codon == 'TGA':
            stop.append(count_start+i*3)
        else:
            pass

    ORF = {} #{起始位置1：终止位置1，起始位置2：终止位置2，...}
    # print(stop)

    for i in range(len(stop)):
        if i == 0:
            tmp = seq[0:stop[0]]
            num = len(tmp)//3
            for j in range(num):
                codon = seq[(count_start + j * 3):(count_start + j * 3 + 3)]
                if codon == 'ATG':
                    ORF[(count_start + j * 3)] = stop[i]
        else:
            tmp = seq[stop[i-1]:stop[i]]
            num = len(tmp)//3
            for j in range(num):
                codon = seq[(stop[i-1]+j*3):(stop[i-1]+j*3+3)]
                if codon == 'ATG':
                    ORF[(stop[i-1]+j*3)] = stop[i]
    return ORF




def get_ORF(seq):
    orf1 = get_start_stop_codon(seq,0) #从0位置开始数
    orf2 = get_start_stop_codon(seq,1) #从1位置开始数
    orf3 = get_start_stop_codon(seq,2) #从2位置开始数
    orf1.update(orf2)
    orf1.update(orf3)
    max_len = 0
    max_len_start = 0 #ORF最长的起始点
    max_len_stop = 0
    if orf1:
        for i in orf1:
            tmp = orf1[i]-i
            if tmp > max_len:
                max_len = tmp
                max_len_start = i
                max_len_stop = orf1[i]
            else:
                pass
        UTR5 = seq[0:max_len_start]
        ORF = seq[max_len_start:max_len_stop]
        UTR3 = seq[max_len_stop:]
        return [UTR5, ORF, UTR3]
    else:
        return ['','','']






