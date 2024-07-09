
import torch
import torch.nn as nn







class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
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
        self.fc2 = nn.Linear(12, 2)

    def forward(self, emb_m, emb_mi, pairing_m, pairing_mi):  #

        m_emb = self.embedding_m(emb_m) + self.position_encoding_m[:, :emb_m.size(1), :] + self.interaction_embedding_m(
            pairing_m)
        mi_emb = self.embedding_mi(emb_mi) + self.position_encoding_mi[:, :emb_mi.size(1),
                                             :] + self.interaction_embedding_mi(pairing_mi)
        # print('aaa',m_emb)

        m_emb = m_emb.permute(1, 0, 2)
        mi_emb = mi_emb.permute(1, 0, 2)

        encoder_output_m = self.encoder_m(m_emb)
        encoder_output_mi = self.encoder_mi(mi_emb)


        cross_attend, _ = self.cross_attention(encoder_output_m, encoder_output_mi, encoder_output_mi)
        output = cross_attend.permute(1, 0, 2).mean(dim=2)


        output = self.fc1(output)
        output = torch.relu(output)
        output = self.fc2(output)
        out = torch.softmax(output, dim=1)

        return out
