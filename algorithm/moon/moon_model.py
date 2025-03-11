from torch import nn  
from utils.train_helper import get_model
from models import * 
import torch.nn.functional as F

class Moon_CNN(nn.Module):

    def __init__(self, in_feat, im_size, hidden=[120, 84]): 
        
        super(Moon_CNN, self).__init__()
        out = im_size 

        self.conv1 = nn.Conv2d(in_feat, 32, kernel_size=3, stride=1, padding=1)
        out = calculate(out, 3, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
        out = calculate(out, kernel=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        out = calculate(out, 3, 1, 1)
        out = calculate(out, kernel=2, stride=2, padding=0)
        
        self.after_conv = out * out * 64
        self.fc1 = nn.Linear(in_features=self.after_conv, out_features=hidden[0]) 
        self.fc2 = nn.Linear(in_features=hidden[0], out_features=hidden[1]) 
    
    def forward(self, X): 
        X = self.pool(F.relu(self.conv1(X)))
        X = self.pool(F.relu(self.conv2(X)))

        X = X.view(-1, self.after_conv)

        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))

        return X
    
class ModelMOON(nn.Module):
    """Model for MOON."""

    def __init__(self, 
                 model_name,
                 basemodel, 
                 out_dim, 
                 n_classes):

        super().__init__()

        if model_name == 'resnet50' or model_name == 'vgg16' or model_name == 'resnet101' or model_name == 'lstm':
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif model_name == 'cnn': 
            self.features = basemodel
            num_ftrs = 84
        
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        self.l3 = nn.Linear(out_dim, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except KeyError as err:
            raise ValueError("Invalid model name.") from err

    def forward(self, x):
        """Forward."""
        h = self.features(x)
        h = h.squeeze()
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        return h, x, y

def init_model(model_name, 
               model_config,
               out_dim=256): 

    if model_name == 'cnn': 
        model = Moon_CNN(in_feat=model_config['in_shape'],
                         im_size=model_config['im_size']) 
    elif model_name == 'resnet50': 
        model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=model_config['out_shape'])
    elif model_name == 'resnet101': 
        model = ResNet(ResidualBlock, [3, 4, 23, 3], num_classes=model_config['out_shape'])
    elif model_name == 'vgg16':
        model = VGG16(in_channels=model_config['in_shape'], num_classes=model_config['out_shape'])
    elif model_name == 'lstm': 
        model = LSTM() 

    return ModelMOON(model_name=model_name, 
                     basemodel=model, 
                     out_dim=out_dim,
                     n_classes=model_config['out_shape'])
