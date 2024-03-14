import pandas as pd
import numpy as np
import sys,os, re
import torch.nn as nn
import torch.optim as optim
import torch


from sklearn.metrics import recall_score,f1_score
from sklearn.metrics import accuracy_score,average_precision_score
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


torch.manual_seed(1234)
torch.set_num_threads(20)

from utils import read_data, read_test
from utils import specificity_score, NPV
from utils import reverse_seq
from utils import get_embedding
from utils import Smith_Waterman
from utils import get_interaction_map
from utils import get_interaction_map_for_test, get_interaction_map_for_test_short
from utils import decision_for_whole




class myDataset(Dataset):
    def __init__(self, data):
        self.data = data


    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        self.sample = self.data[index]
        self.mirna, self.mrna, self.label = self.sample
        self.reverse_mrna = reverse_seq(self.mrna)
        self.mirna = self.mirna + 'X' * (30 - len(self.mirna))
        emb_m = get_embedding(self.reverse_mrna)
        emb_mi = get_embedding(self.mirna)
        pairing_m, pairing_mi = get_interaction_map(self.mirna,self.reverse_mrna)

        emb_m = torch.tensor(emb_m)
        emb_mi = torch.tensor(emb_mi)
        pairing_m = torch.tensor(pairing_m)
        pairing_mi = torch.tensor(pairing_mi)



        label = torch.tensor(self.label,dtype=torch.float)
        return {'fea1': emb_m,
                'fea2': emb_mi,
                'fea3': pairing_m,
                'fea4': pairing_mi,
                'label': label,
               }











# define the model architecture of Mimosa
class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout, output_size):
        super(Transformer, self).__init__()

        self.embedding_m = nn.Embedding(input_size, hidden_size)
        self.position_encoding_m = nn.Parameter(torch.zeros(1, 100, hidden_size))
        self.interaction_embedding_m = nn.Embedding(3, hidden_size)
        nn.init.normal_(self.position_encoding_m, mean=0, std=0.1)


        self.embedding_mi = nn.Embedding(input_size, hidden_size)
        self.position_encoding_mi = nn.Parameter(torch.zeros(1, 100, hidden_size))
        self.interaction_embedding_mi = nn.Embedding(3, hidden_size)
        nn.init.normal_(self.position_encoding_mi, mean=0, std=0.1)


        encoder_layers_m = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.encoder_m = nn.TransformerEncoder(encoder_layers_m, num_layers)

        encoder_layers_mi = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.encoder_mi = nn.TransformerEncoder(encoder_layers_mi, num_layers)

        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads)

        self.fc1 = nn.Linear(40, 12)
        self.fc2 = nn.Linear(12, output_size)


    def forward(self, emb_m, emb_mi, pairing_m, pairing_mi):#


        m_emb = self.embedding_m(emb_m) + self.position_encoding_m[:, :emb_m.size(1), :] + self.interaction_embedding_m(pairing_m)
        mi_emb = self.embedding_mi(emb_mi)  + self.position_encoding_mi[:,:emb_mi.size(1),:] + self.interaction_embedding_mi(pairing_mi)

        m_emb = m_emb.permute(1, 0, 2)
        mi_emb = mi_emb.permute(1, 0, 2)

        encoder_output_m = self.encoder_m(m_emb)
        encoder_output_mi = self.encoder_mi(mi_emb)

        cross_attend, _ = self.cross_attention(encoder_output_m, encoder_output_mi, encoder_output_mi)
        output = cross_attend.permute(1,0,2).mean(dim=2)


        output = self.fc1(output)
        output = torch.relu(output)
        output = self.fc2(output)
        out = torch.softmax(output, dim=1)
        return out














def Deep_train(model, dataloader, optimizer, criterion):
    print('Training')
    model.train()
    counter = 0
    train_loss = 0.0
    for i, data in enumerate(dataloader):
        counter += 1
        features1, features2, features3, features4, target = data['fea1'], data['fea2'], data['fea3'], data['fea4'], data['label']
        outputs = model(features1,features2,features3,features4)
        loss = criterion(outputs,target)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_total_loss = train_loss/counter
    return train_total_loss




def Deep_validate(model, dataloader, criterion):
    print('Validating')
    model.eval()
    counter = 0
    val_loss = 0.0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            counter += 1
            features1, features2, features3, features4, target = data['fea1'], data['fea2'], data['fea3'], data['fea4'],data['label']
            outputs = model(features1, features2,features3,features4)
            loss = criterion(outputs,target)

            val_loss += loss.item()
            predictions = []
            outputs = outputs.cpu().numpy()
            target = target.cpu().numpy()
            # print(type(outputs))
            for i in outputs:
                i = i.tolist()
                if i[1] > 0.5:
                    predictions.append(1)
                else:
                    predictions.append(0)
            all_predictions.extend(predictions)
            all_targets.extend([i[1] for i in target])

        val_total_loss = val_loss/counter


        acc = accuracy_score(all_targets, all_predictions)
        pre = average_precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets,all_predictions)
        spec = specificity_score(all_targets,all_predictions)
        f1 = f1_score(all_targets,all_predictions)
        npv = NPV(all_targets, all_predictions)




        print('acc',acc)
        print('pre',pre)
        print('recall',recall)
        print('specificity',spec)
        print('f1',f1)
        print('npv',npv)



        return val_total_loss










def perform_train():
    # train positive: 26995, train negative: 27469, val positive: 2193, val negative: 2136
    batchsize = 256
    learningrate = 1e-4
    epochs = 40
    train, val = read_data('miRAW_Train_Validation.txt')

    train_dataset = myDataset(train)
    val_dataset = myDataset(val)
    train_loader = DataLoader(train_dataset,batch_size=batchsize,shuffle=True) #collate_fn=my_collate_fn,
    val_loader = DataLoader(val_dataset, batch_size=batchsize,shuffle=True) #,collate_fn=my_collate_fn

    model = Transformer(input_size=5, hidden_size=64, num_layers=16, num_heads=8, dropout=0.1, output_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningrate,weight_decay=1e-5) #,weight_decay=1e-5

    best_val_loss = 1
    train_loss = []
    valid_loss = []
    for epoch in range(epochs):
        print(f'Epoch {epoch+1} of {epochs}')
        train_epoch_loss = Deep_train(model, train_loader,optimizer, criterion)
        valid_epoch_loss = Deep_validate(model,val_loader,criterion)

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        print('Train Loss:', train_epoch_loss)
        print('Val Loss:',valid_epoch_loss)
        if valid_epoch_loss < best_val_loss:
            best_val_loss = valid_epoch_loss
            # torch.save(model,'model_concate_{}.pth'.format(epoch))










def get_cts(rmrna, stepsize):
    '''segment full-length mRNAS into 40-nt segments using a sliding window with predefined stepsize'''
    kmers = []

    if len(rmrna) >= 40:
        for i in range(0, len(rmrna),stepsize):
            if i + 40 <= len(rmrna):
                cut = rmrna[i:i + 40]
                kmers.append(cut)

        return kmers
    else:
        pad_rmrna = rmrna + 'X' * (40 - len(rmrna))
        kmers.append(pad_rmrna)
        return kmers




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
        pros = model(fea1, fea2, fea3, fea4).detach().numpy().tolist()
        pppp = decision_for_whole(pros)

        return pppp






def perform_test(pathfile,stepsize):

    test = read_test(pathfile)
    y_true = []
    y_pred = []
    model = torch.load('model.pth')

    print('个数',len(test))

    for index in range(len(test)): #range(len(test))
        fasta = test[index]


        mirna = fasta[0].upper().replace('T', 'U')

        mrna = fasta[1].upper().replace('T', 'U')
        reverse_mrna = reverse_seq(mrna)
        y_true.append(fasta[2])


        kmers = get_cts(reverse_mrna,stepsize)

        if kmers is None:
            pre = 0
            y_pred.append(pre)

        else:

            pre = kmers_predict(kmers, mirna, model)
            y_pred.append(pre)

    print(y_true)
    print(y_pred)
    acc = accuracy_score(y_true, y_pred)
    pre = average_precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    spec = specificity_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = NPV(y_true, y_pred)

    print('acc', acc)
    print('PPV', pre)
    print('recall', recall)
    print('specificity', spec)
    print('f1', f1)
    print('NPV', auc)




perform_train()
# perform_test('miRAW_Test0.txt', stepsize=5)








































































































