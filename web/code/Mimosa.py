
import argparse
import os
from predict import Mimosa_predict,Mimosa_predict_cut_utr
root_path = os.getcwd()
#os.chmod(root_path, stat.S_IWOTH)
from transfomer import Transformer

# 将mi_file, m_file中的fasta序列合并到inputfile文件中
def incorperate(mi_file, m_file, inputfile):
    mi_name, mi_seq = [], [] 
    m_name, m_seq = [], []
    with open(mi_file, mode='r') as file_1:
        for line in file_1.readlines():
            if line[0] == '>':
                mi_name.append(line[1:-1].strip())
            else:
                mi_seq.append(line.strip())

    with open(m_file, mode='r') as file_2:
        for line in file_2.readlines():
            if line[0] == '>':
                m_name.append(line[1:-1].strip())
            else:
                m_seq.append(line.strip())
    
    with open(inputfile, mode='w') as file_3:
        for index, _ in enumerate(mi_name):
            file_3.write(mi_name[index] + '\t' + mi_seq[index] +'\t' + m_name[index] + '\t' + m_seq[index] + '\n')








def main():
    parser = argparse.ArgumentParser(description='Mimosa is a deep learning model to predict miRNA-mRNA interactions')
    parser.add_argument('-input',dest='inputfile',type=str, required=True,
                        help='Query miRNA-mRNA pairs')
    parser.add_argument('-input_mi',dest='input_mi_file',type=str, required=True)
    parser.add_argument('-input_m',dest='input_m_file',type=str, required=True)

    parser.add_argument('-type', dest='type', type=str, required=True,
                        help="Type of input mRNA: complete mRNA or mRNA 3'UTR")
    parser.add_argument('-region', dest='region', type=str, required=True,
                        help="The region that performs prediction: complete mRNA or mRNA 3'UTR")
                        
    parser.add_argument('-stepsize',dest='stepsize', type=int, required=False,
                        help='The step size of the sliding window: choose one size from 1 to 10')
                        
    parser.add_argument('-output',dest='outputfile',type=str,required=False,
                        help='The path where you want to save the prediction results')
    args = parser.parse_args()


    mi_file = args.input_mi_file
    m_file = args.input_m_file
    inputfile = args.inputfile
    # 合并miRNA和mRNA序列为一个文件
    incorperate(mi_file, m_file, inputfile)
    
    # inputfile = args.inputfile
    outputfile = args.outputfile
    type = args.type
    region = args.region
    size = int(args.stepsize)
    default_output = 'results'
    default_size = "5"
    
    if outputfile != None:
        default_output = outputfile
    if size != None:
        default_size = size
    
    if type == "3UTR":
        Mimosa_predict(inputfile, default_output, default_size)
        print('Prediction results have been saved in ' + default_output)


    if type == "complete" and  region =='complete':
        Mimosa_predict(inputfile, default_output, default_size)
        print('Prediction results have been saved in ' + default_output)


    if type == "complete" and  region == "3UTR":
        Mimosa_predict_cut_utr(inputfile, default_output, default_size)
        print('Prediction results have been saved in ' + default_output)










if __name__ == "__main__":
    main()











