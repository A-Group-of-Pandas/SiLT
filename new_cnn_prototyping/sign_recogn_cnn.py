import torch
from torch import nn
from torchvision.models import mobilenet_v3_large


class RecogJoint(nn.Module):
    def __init__(self):
        super(RecogJoint, self).__init__()
        self.dense1 = nn.Linear(105, 196)
        self.batch_norm1 = nn.BatchNorm1d(196)
        self.dropout1 = nn.Dropout(0.0)
        self.dense2 = nn.Linear(196, 32)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.0)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (N,21,3)
        x = torch.cat([x, torch.sin(x[:, :, 0:1]), torch.cos(x[:, :, 0:1])], dim=-1)
        # print(xs[0].shape)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout1(self.batch_norm1(self.relu(self.dense1(x))))
        x = self.dropout2(self.batch_norm2(self.relu(self.dense2(x))))
        # print(x.shape)
        return x


class SignRecogCNN(nn.Module):
    def __init__(self):
        super(SignRecogCNN, self).__init__()
        mobilenet = mobilenet_v3_large(pretrained=True)
        extract_layers = list(mobilenet.children())[0][:11]
        self.feature_extract = nn.Sequential(*extract_layers)
        for param in self.feature_extract.parameters():
            param.requires_grad = False
        # 14*14*80
        self.dropout0 = nn.Dropout(0.0)
        self.conv1 = nn.Conv2d(80, 128, (3, 3), padding="same")
        self.batchnorm1 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(128, 128, (3, 3), padding="same")
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.0)
        self.pool1 = nn.MaxPool2d((3, 3), stride=2)
        # 6*6*64
        self.conv3 = nn.Conv2d(128, 128, (3, 3))
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.0)
        # 4*4*64
        self.conv4 = nn.Conv2d(128, 196, (2, 2))
        self.batchnorm4 = nn.BatchNorm2d(196)
        self.dropout4 = nn.Dropout(0.0)
        # 3*3*64
        self.dense1 = nn.Linear(196, 218)
        self.batchnorm6 = nn.BatchNorm1d(218)
        self.dropout5 = nn.Dropout(0.0)
        self.joint_recog = RecogJoint()
        self.dense2 = nn.Linear(250, 26)
        self.relu = nn.ReLU()

    def forward(self, x, joints):
        # x : (N,3,224,224)
        # joints : (N,21,3)
        x = self.dropout0(self.feature_extract(x))
        x = self.dropout1(self.batchnorm1(self.relu(self.conv1(x))))
        x = self.dropout2(self.batchnorm2(self.relu(self.conv2(x))))
        x = self.pool1(x)
        x = self.dropout3(self.batchnorm3(self.relu(self.conv3(x))))
        x = self.dropout4(self.batchnorm4(self.relu(self.conv4(x))))
        x = torch.max(torch.flatten(x, start_dim=2, end_dim=3), dim=2).values
        x = self.dropout5(self.batchnorm6(self.relu(self.dense1(x))))
        y = self.joint_recog(joints)
        x = self.dense2(torch.cat([x, y], dim=-1))
        return x
