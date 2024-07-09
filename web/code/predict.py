import numpy as np
import argparse
import os, sys, re
import joblib
import pandas as pd
from encoding import get_embedding, get_interaction_map, Smith_Waterman
from preprocess import get_ORF, segmentation_for_mRNA
from readfile import read_file
import torch
from transfomer import Transformer
root_path = os.getcwd()
import torch.nn as nn



def determination(data_all):
    probabilities = [i[1] for i in data_all]
    max_value = np.max(probabilities)

    count = len([i for i in probabilities if i > 0.5])
    if count >= 1:
        return 1
    else:
        return 0





def get_interaction_map_for_test(mirna,mrna):
    len_mi = len(mirna)

    score, m_pair, mi_pair = Smith_Waterman(mirna[:10], mrna[5:35])
    map_m = [0] * 5 + m_pair + [0] * 5
    map_mi = mi_pair + (len_mi - 10) * [0]
    return map_m, map_mi


def get_interaction_map_for_test_short(mirna,mrna):
    len_mi = len(mirna)

    score, m_pair, mi_pair = Smith_Waterman(mirna[:10], mrna[:])
    map_m = m_pair
    map_mi = mi_pair + (len_mi - 10) * [0]
    return map_m, map_mi


def kmers_predict(kmers,mirna,model):

    mirna = mirna + 'X'*(30-len(mirna))
    fea1 = []
    fea2 = []
    fea3 = []
    fea4 = []
    if len(kmers) == 0:
        return 0
    else:
        for i in kmers:
            fea_1 = get_embedding(i)
            fea_2 = get_embedding(mirna)
            if 'X' in i:
                pairing_m, pairing_mi = get_interaction_map_for_test_short(mirna, i)
            else:
                pairing_m, pairing_mi = get_interaction_map_for_test(mirna, i)

            fea1.append(fea_1)
            fea2.append(fea_2)
            fea3.append(pairing_m)
            fea4.append(pairing_mi)


        fea1 = torch.tensor(fea1, dtype=torch.long)
        fea2 = torch.tensor(fea2)
        fea3 = torch.tensor(fea3)
        fea4 = torch.tensor(fea4)
        pppp = model(fea1, fea2, fea3, fea4).detach().numpy().tolist()
        pro = determination(pppp)
        return pro






def Mimosa_predict(inputfile,outputfile, step_size):
    all = read_file(inputfile)
    clf = torch.load('/var/www/html/Mimosa/code/model.pth')

    f = open( outputfile, 'w')
    f.write('miRNA_id'+','+'mRNA_id'+',' + 'Prediction label'+'\n')
    for pair in all:  # range(len(test))
        mi_id,mirna,m_id,mrna = pair[0], pair[1], pair[2], pair[3]

        kmers = segmentation_for_mRNA(mrna, step_size)
        pre = kmers_predict(kmers, mirna, clf)
        
        f.write(mi_id + ',' + m_id + ',' + str(round(pre)) + '\n')

    f.close()




def Mimosa_predict_cut_utr(inputfile,outputfile, step_size):
    all = read_file(inputfile)
    clf = torch.load('/var/www/html/Mimosa/code/model.pth')
    f = open(outputfile, 'w')
    f.write('miRNA_id' + ',' + 'mRNA_id' + ',' + 'Prediction label' + '\n')
    for pair in all:
        mi_id, mirna, m_id, mrna_complete = pair[0], pair[1], pair[2], pair[3]
        _,_,utr3 = get_ORF(mrna_complete)
        if utr3 is None:
            pre = 0
            f.write(mi_id + ',' + m_id + ',' + str(round(pre)) + '\n')
        else:
            kmers = segmentation_for_mRNA(utr3, step_size)
            pre = kmers_predict(kmers, mirna, clf)
            
            f.write(mi_id + ',' + m_id + ',' + str(round(pre)) + '\n')

    f.close()

