import torch
from torch import nn
from torchvision.models import mobilenet_v3_large


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
        self.dense2 = nn.Linear(218, 26)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout0(self.feature_extract(x))
        x = self.dropout1(self.batchnorm1(self.relu(self.conv1(x))))
        x = self.dropout2(self.batchnorm2(self.relu(self.conv2(x))))
        x = self.pool1(x)
        x = self.dropout3(self.batchnorm3(self.relu(self.conv3(x))))
        x = self.dropout4(self.batchnorm4(self.relu(self.conv4(x))))
        x = torch.max(torch.flatten(x, start_dim=2, end_dim=3), dim=2).values
        x = self.dropout5(self.batchnorm6(self.relu(self.dense1(x))))
        return self.dense2(x)


# class SignRecogCNN(nn.Module):
# 	def __init__(self):
# 		super(SignRecogCNN, self).__init__()
# 		mobilenet = mobilenet_v3_large(pretrained=True)
# 		extract_layers = list(mobilenet.children())[0][:7]
# 		self.feature_extract = nn.Sequential(*extract_layers)
# 		for param in self.feature_extract.parameters():
# 			param.requires_grad = False
# 		self.conv1 = nn.Conv2d(40,120,(1,1),padding='same')
# 		self.batchnorm1 = nn.BatchNorm2d(120)
# 		self.conv2 = nn.Conv2d(120,40,(3,3),padding='same')
# 		self.batchnorm2 = nn.BatchNorm2d(40)
# 		self.conv3 = nn.Conv2d(40,120,(3,3),padding='same')
# 		self.batchnorm3 = nn.BatchNorm2d(120)
# 		self.pool1 = nn.MaxPool2d((3,3),stride=2)
# 		# 13*13*40
# 		self.conv4 = nn.Conv2d(120,40,(3,3),padding='same')
# 		self.batchnorm4 = nn.BatchNorm2d(40)
# 		self.conv5 = nn.Conv2d(40,120,(3,3),padding='same')
# 		self.batchnorm5 = nn.BatchNorm2d(120)
# 		self.pool2 = nn.MaxPool2d((4,4),stride=2)
# 		# 5*5*40
# 		self.conv6 = nn.Conv2d(120,40,(3,3),padding='same')
# 		self.batchnorm6 = nn.BatchNorm2d(40)
# 		self.conv7 = nn.Conv2d(40,120,(3,3),padding='same')
# 		self.batchnorm7 = nn.BatchNorm2d(120)
# 		self.dense1 = nn.Linear(120,512)
# 		self.batchnorm8 = nn.BatchNorm1d(512)
# 		self.dense2 = nn.Linear(512,26)
# 		self.relu = nn.ReLU()

# 	def forward(self, x):
# 		x = self.feature_extract(x)
# 		x1 = self.batchnorm1(self.relu(self.conv1(x)))
# 		x = self.batchnorm2(self.relu(self.conv2(x1)))
# 		x = self.relu(self.conv3(x))
# 		x3 = self.batchnorm3(self.pool1(x+x1))
# 		x = self.batchnorm4(self.relu(self.conv4(x3)))
# 		x = self.relu(self.conv5(x))
# 		x5 = self.batchnorm5(self.pool2(x+x3))
# 		x = self.batchnorm6(self.relu(self.conv6(x5)))
# 		x = self.batchnorm7(self.relu(self.conv7(x))+x5)
# 		x = torch.flatten(x, start_dim=2, end_dim=3)
# 		x = torch.mean(x, dim=2)
# 		x = self.batchnorm8(self.relu(self.dense1(x)))
# 		return self.dense2(x)

# class module_3_5(nn.Module):
# 	def __init__(self, input_dim, output_dim):
# 		super(module_3_5, self).__init__()
# 		# always keep input_dim and ouput_dim even!
# 		# 512 --> 256
# 		self.condense1 = nn.Conv2d(input_dim, input_dim//2, (1,1))
# 		self.conv3 = nn.Conv2d(input_dim//2, output_dim//2, (3,3), padding='same')
# 		# 3 by 3 convnet
# 		self.batch_norm1 = nn.BatchNorm2d(output_dim//2)
# 		# batch normalization
# 		self.conv5 = nn.Conv2d(input_dim//2, output_dim//2, (5,5), padding='same')
# 		# 5 by 5 convnet
# 		self.batch_norm2 = nn.BatchNorm2d(output_dim//2)
# 		self.relu = nn.ReLU()

# 	def forward(self, x):
# 		# forward pass
# 		x = self.relu(self.condense1(x))
# 		x3 = self.conv3(x)
# 		x3 = self.relu(self.batch_norm1(x3))
# 		x5 = self.conv5(x)
# 		x5 = self.relu(self.batch_norm2(x5))
# 		return torch.cat([x3,x5],dim=1) # pytorch is channel first!

# class SignRecogCNN(nn.Module):
# 	def __init__(self):
# 		super(SignRecogCNN, self).__init__()
# 		# size: 100
# 		self.conv1 = nn.Conv2d(3,64,(3,3),padding='same')
# 		self.module1 = module_3_5(64,64)
# 		self.module2 = module_3_5(64,64)
# 		self.pool1 = nn.MaxPool2d((3,3),2)
# 		# MaxPool2d "slides the windows"
# 		# size: 49
# 		self.module3 = module_3_5(64,96)
# 		self.module4 = module_3_5(96,96)
# 		self.pool2 = nn.MaxPool2d((2,2),2)
# 		# size: 24
# 		self.module5 = module_3_5(96,128)
# 		self.module6 = module_3_5(128,128)
# 		self.pool3 = nn.MaxPool2d((2,2),2)
# 		# size: 11
# 		self.module7 = module_3_5(128,144)
# 		self.module8 = module_3_5(144,144)
# 		self.pool4 = nn.MaxPool2d((2,2),2)
# 		# size: 5
# 		self.module9 = module_3_5(144,144)
# 		self.module10 = module_3_5(144,144)
# 		# global average pooling
# 		self.dense1 = nn.Linear(144, 512)
# 		self.batch_norm1 = nn.BatchNorm1d(512)
# 		self.dropout1 = nn.Dropout(0.4)
# 		self.dense2 = nn.Linear(512, 26)
# 		self.relu = nn.ReLU()

# 	def forward(self, x):
# 		# forward pass
# 		x = self.relu(self.conv1(x))
# 		# just a bunch of arithmetic
# 		x1 = self.module1(x)
# 		x2 = self.module2(x1)
# 		x = self.pool1(x1+x2)
# 		x3 = self.module3(x)
# 		x4 = self.module4(x3)
# 		x = self.pool2(x3+x4)
# 		x5 = self.module5(x)
# 		x6 = self.module6(x5)
# 		x = self.pool3(x5+x6)
# 		x7 = self.module7(x)
# 		x8 = self.module8(x7)
# 		x = self.pool4(x7+x8)
# 		x9 = self.module9(x)
# 		x10 = self.module10(x9)
# 		x = x9+x10
# 		# shape (N,128,4,4)
# 		x = torch.flatten(x, start_dim=2, end_dim=3)
# 		x = torch.mean(x, dim=-1)
# 		x = self.dropout1(self.batch_norm1(self.relu(self.dense1(x))))
# 		x = self.dense2(x)
# 		return x
