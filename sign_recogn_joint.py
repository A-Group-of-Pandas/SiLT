import torch
from torch import nn


class SignRecogJoint(nn.Module):
    def __init__(self):
        super(SignRecogJoint, self).__init__()
        self.dense1 = nn.Linear(63, 128)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(128, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(128, 26)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout1(self.batch_norm1(self.relu(self.dense1(x))))
        x = self.dropout2(self.batch_norm2(self.relu(self.dense2(x))))
        return self.dense3(x)


class RecogJoint(nn.Module):
    def __init__(self):
        super(RecogJoint, self).__init__()
        self.dense1 = nn.Linear(105, 196)
        self.batch_norm1 = nn.BatchNorm1d(196)
        self.dropout1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(196, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        self.dense3 = nn.Linear(128, 26)
        self.relu = nn.ReLU()
        self.shift = 3.1415926 / 2

    def forward(self, x):
        # (N,21,3)
        # print(x.shape)
        xs = [x for i in range(4)]
        for i in range(4):
            if i > 4 // 2:
                xs[0][:, :, 0] -= self.shift * (4 - i)
            else:
                xs[0][:, :, 0] += self.shift * i
        xs = [
            torch.cat([x, torch.sin(x[:, :, 0:1]), torch.cos(x[:, :, 0:1])], dim=-1)
            for x in xs
        ]
        # print(xs[0].shape)
        xs = [torch.flatten(x, start_dim=1) for x in xs]
        xs = [self.dropout1(self.batch_norm1(self.relu(self.dense1(x)))) for x in xs]
        xs = [self.dropout2(self.batch_norm2(self.relu(self.dense2(x)))) for x in xs]
        # 6,N,128
        x = torch.mean(torch.stack(xs, dim=0), dim=0)
        # print(x.shape)
        return self.dense3(x)
