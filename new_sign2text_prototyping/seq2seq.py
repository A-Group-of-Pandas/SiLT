import torch
from torch import nn


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.ehidden = 64
        self.elayer = 2
        self.encoder = nn.LSTM(29, self.ehidden, self.elayer)
        self.decoder1 = nn.LSTMCell(29, 64)
        self.decoder2 = nn.LSTMCell(64, 64)
        self.classifier = nn.Linear(64, 29)
        # includes stop char

    def forward(self, input_seq):
        # (T,N,29)
        h = torch.zeros((self.elayer, input_seq.shape[1], self.ehidden))
        c = torch.zeros((self.elayer, input_seq.shape[1], self.ehidden))
        encode_seq, (h1, c1) = self.encoder(input_seq, (h, c))
        h1, c1 = h1[1], c1[1]
        h2 = torch.zeros((input_seq.shape[1], self.ehidden))
        c2 = torch.zeros((input_seq.shape[1], self.ehidden))
        outputs = []
        let = torch.zeros((input_seq.shape[1], 29))
        let[:, 27] = 1  # start char
        for i in range(len(input_seq)):
            h1, c1 = self.decoder1(let, (h1, c1))
            h2, c2 = self.decoder2(h1, (h2, c2))
            let_pred = self.classifier(h2)
            outputs.append(let_pred[None, ...])
            max_lets = torch.max(let_pred, dim=-1).indices
            let = torch.zeros((input_seq.shape[1], 29))
            let[:, max_lets] = 1.0
        # (T,N,29)
        return torch.cat(outputs)
