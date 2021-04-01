import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()

        self.branch1 = nn.Sequential(
            BasicConv2d(3, 32, (3, 3), stride=2),
            BasicConv2d(32, 32, (3, 3)),
            BasicConv2d(32, 64, (3, 3), padding=1)
        )

        self.branch2_a = nn.MaxPool2d((3, 3), stride=2)
        self.branch2_b = BasicConv2d(64, 96, (3, 3), stride=2)

        self.branch3_a = nn.Sequential(
            BasicConv2d(160, 64, (1, 1)),
            BasicConv2d(64, 96, (3, 3))
        )

        self.branch3_b = nn.Sequential(
            BasicConv2d(160, 64, (1, 1)),
            BasicConv2d(64, 64, (7, 1), padding=(3, 0)),
            BasicConv2d(64, 64, (1, 7), padding=(0, 3)),
            BasicConv2d(64, 96, (3, 3))
        )

        self.branch4_a = BasicConv2d(192, 192, (3, 3), stride=2)
        self.branch4_b = nn.MaxPool2d((3, 3), stride=2)

    def forward(self, x):
        x = self.branch1(x)

        x_a = self.branch2_a(x)
        x_b = self.branch2_b(x)
        x = torch.cat((x_a, x_b), axis=1)

        x_a = self.branch3_a(x)
        x_b = self.branch3_b(x)
        x = torch.cat((x_a, x_b), axis=1)

        x_a = self.branch4_a(x)
        x_b = self.branch4_b(x)
        x = torch.cat((x_a, x_b), axis=1)

        return x


class InceptionBlockA(nn.Module):
    def __init__(self):
        super(InceptionBlockA, self).__init__()

        self.relu1 = nn.ReLU()

        self.branch1_a = BasicConv2d(384, 32, (1, 1))
        self.branch1_b = nn.Sequential(
            BasicConv2d(384, 32, (1, 1)),
            BasicConv2d(32, 32, (3, 3), padding=1),
        )
        self.branch1_c = nn.Sequential(
            BasicConv2d(384, 32, (1, 1)),
            BasicConv2d(32, 48, (3, 3), padding=1),
            BasicConv2d(48, 64, (3, 3), padding=1)
        )

        self.branch2 = BasicConv2d(128, 384, (1, 1))

        self.relu2 = nn.ReLU()

    def forward(self, input):
        x = self.relu1(input)

        x_a = self.branch1_a(x)
        x_b = self.branch1_b(x)
        x_c = self.branch1_c(x)
        x = torch.cat((x_a, x_b, x_c), axis=1)

        x = self.branch2(x)
        x = x + input

        x = self.relu2(x)

        return x


class ReductionBlockA(nn.Module):
    def __init__(self):
        super(ReductionBlockA, self).__init__()

        self.branch1_a = nn.MaxPool2d((3, 3), stride=2)
        self.branch1_b = BasicConv2d(384, 384, (3, 3), stride=2)
        self.branch1_c = nn.Sequential(
            BasicConv2d(384, 256, (1, 1)),
            BasicConv2d(256, 256, (3, 3), padding=1),
            BasicConv2d(256, 384, (3, 3), stride=2)
        )

    def forward(self, x):
        x_a = self.branch1_a(x)
        x_b = self.branch1_b(x)
        x_c = self.branch1_c(x)
        x = torch.cat((x_a, x_b, x_c), axis=1)

        return x


class InceptionBlockB(nn.Module):
    def __init__(self):
        super(InceptionBlockB, self).__init__()

        self.relu1 = nn.ReLU()

        self.branch1_a = BasicConv2d(1152, 192, (1, 1))
        self.branch1_b = nn.Sequential(
            BasicConv2d(1152, 128, (1, 1)),
            BasicConv2d(128, 160, (1, 7), padding=(0, 3)),
            BasicConv2d(160, 192, (7, 1), padding=(3, 0))
        )

        self.branch2 = BasicConv2d(384, 1152, (1, 1))

        self.relu2 = nn.ReLU()

    def forward(self, input):
        x = self.relu1(input)

        x_a = self.branch1_a(x)
        x_b = self.branch1_b(x)
        x = torch.cat((x_a, x_b), axis=1)

        x = self.branch2(x)
        x = x + input

        x = self.relu2(x)

        return x


class ReductionBlockB(nn.Module):
    def __init__(self):
        super(ReductionBlockB, self).__init__()

        self.branch1_a = nn.MaxPool2d((3, 3), stride=2)
        self.branch1_b = nn.Sequential(
            BasicConv2d(1152, 256, (1, 1)),
            BasicConv2d(256, 384, (3, 3), stride=2)
        )
        self.branch1_c = nn.Sequential(
            BasicConv2d(1152, 256, (1, 1)),
            BasicConv2d(256, 288, (3, 3), stride=2)
        )
        self.branch1_d = nn.Sequential(
            BasicConv2d(1152, 256, (1, 1)),
            BasicConv2d(256, 288, (3, 3), padding=1),
            BasicConv2d(288, 320, (3, 3), stride=2)
        )

    def forward(self, x):
        x_a = self.branch1_a(x)
        x_b = self.branch1_b(x)
        x_c = self.branch1_c(x)
        x_d = self.branch1_d(x)
        x = torch.cat((x_a, x_b, x_c, x_d), axis=1)

        return x


class InceptionBlockC(nn.Module):
    def __init__(self):
        super(InceptionBlockC, self).__init__()

        self.relu1 = nn.ReLU()

        self.branch1_a = BasicConv2d(2144, 192, (1, 1))
        self.branch1_b = nn.Sequential(
            BasicConv2d(2144, 192, (1, 1)),
            BasicConv2d(192, 224, (1, 3), padding=(0, 1)),
            BasicConv2d(224, 256, (3, 1), padding=(1, 0))
        )

        self.branch2 = BasicConv2d(448, 2144, (1, 1))

        self.relu2 = nn.ReLU()

    def forward(self, input):
        x = self.relu1(input)

        x_a = self.branch1_a(x)
        x_b = self.branch1_b(x)
        x = torch.cat((x_a, x_b), axis=1)

        x = self.branch2(x)
        x = x + input

        x = self.relu2(x)

        return x


class InceptionResnetV2(nn.Module):
    def __init__(self):
        super(InceptionResnetV2, self).__init__()

        self.model = nn.Sequential(
            Stem(),
            *[InceptionBlockA() for _ in range(5)],
            ReductionBlockA(),
            *[InceptionBlockB() for _ in range(10)],
            ReductionBlockB(),
            *[InceptionBlockC() for _ in range(5)],
            nn.AvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Dropout(0.8),
            nn.Linear(2144, 8)
        )

    def forward(self, x):
        return self.model(x)
