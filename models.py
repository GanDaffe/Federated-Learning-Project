import torch
import torch.nn as nn
import torchvision.models as models

def calculate(size, kernel, stride, padding):
    return int(((size+(2*padding)-kernel)/stride) + 1)

class fMLP(nn.Module):
    def __init__(self) -> None:
        super(fMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(28*28, 200)
        self.layer2 = nn.Linear(200, 200)
        self.layer3 = nn.Linear(200, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

class eMLP(nn.Module):
    def __init__(self) -> None:
        super(eMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(28*28, 200)
        self.layer2 = nn.Linear(200, 200)
        self.layer3 = nn.Linear(200, 47)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x

class fCNN(nn.Module):
    def __init__(self) -> None:
        super(fCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(4*4*64, 512),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class eCNN(nn.Module):
    def __init__(self) -> None:
        super(eCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(4*4*64, 512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.fc2 = nn.Linear(512, 47)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class CNN2(nn.Module):
    def __init__(self) -> None:
        super(CNN2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(5*5*64, 512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.fc2 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class CNN4(nn.Module):
    def __init__(self) -> None:
        super(CNN4, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(5*5*64, 512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.fc3 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

def calculate(size, kernel, stride, padding):
    return (size - kernel + 2 * padding) // stride + 1

class CustomCNN(nn.Module):
    def __init__(self, num_layer, im_size, in_shape, hidden, out_shape):
        super(CustomCNN, self).__init__()
        out = im_size

        self.conv_layers = nn.ModuleList()
        factor = 1  # Hệ số nhân số kênh

        for i in range(num_layer):
            self.conv_layers.append(nn.Conv2d(in_shape, factor * hidden, kernel_size=3, stride=1, padding=1))
            out = calculate(out, kernel=3, stride=1, padding=1)

            self.conv_layers.append(nn.ReLU())

            self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            out = calculate(out, kernel=2, stride=2, padding=0)

            in_shape = factor * hidden

            if (i + 1) % 2 == 0:
                factor *= 2

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=out * out * (factor // 2) * hidden, out_features=out_shape)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(2000, 50)
        self.lstm = nn.LSTM(input_size=50, hidden_size=64)

        self.fc1 = nn.Linear(64, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = self.fc1(lstm_out)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

class ResNet101(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(ResNet101, self).__init__()
        self.resnet = models.resnet101(weights=None)

        self.resnet.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x

class VGG16(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG16, self).__init__()

        self.vgg = models.vgg16(weights=None)

        self.vgg.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        self.vgg.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg(x)

class ResNet50(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet50, self).__init__()
        # Load pre-trained ResNet50 model from torchvision
        self.resnet = models.resnet50(weights=None)

        # Điều chỉnh lớp đầu vào conv1 của ResNet để phù hợp với cấu trúc mà bạn yêu cầu (3 kênh đầu vào)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Thay thế lớp fully connected cuối cùng để phù hợp với 100 lớp của CIFAR-100
        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        return x

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.classes = torch.unique(torch.tensor(targets)).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]