import torch
import torch.nn as nn
from torchvision import models

class TransferLearningModel(nn.Module):
    def __init__(self, num_classes, base_model='resnet18', pretrained=True):
        super(TransferLearningModel, self).__init__()
        
        if base_model == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        elif base_model == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        elif base_model == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        elif base_model == 'vgg16':
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            self.backbone = models.vgg16(weights=weights)
            num_ftrs = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Linear(num_ftrs, num_classes)
        elif base_model == 'mobilenet_v2':
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)
            num_ftrs = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"Model {base_model} not supported")
            
    def forward(self, x):
        return self.backbone(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128) # Assuming 224x224 input
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CRNN(nn.Module):
    """
    CNN + RNN architecture for capturing both frequency and temporal patterns.
    Input: (batch, 3, height, width) - spectrogram images
    """
    def __init__(self, num_classes, input_height=224, input_width=224):
        super(CRNN, self).__init__()
        
        # CNN Feature Extractor
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112x112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14
        )
        
        # Calculate flattened feature size after CNN
        self.feature_size = 256 * 14  # 256 channels * 14 height
        
        # RNN for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, 3, 224, 224)
        batch_size = x.size(0)
        
        # CNN features
        x = self.conv_block(x)  # (batch, 256, 14, 14)
        
        # Reshape for RNN: treat width as time sequence
        # (batch, channels, height, width) -> (batch, width, channels*height)
        x = x.permute(0, 3, 1, 2)  # (batch, 14, 256, 14)
        x = x.reshape(batch_size, 14, -1)  # (batch, 14, 256*14)
        
        # RNN
        x, _ = self.lstm(x)  # (batch, 14, 256)
        
        # Take last timestep
        x = x[:, -1, :]  # (batch, 256)
        
        # Classifier
        x = self.fc(x)
        
        return x


class AttentionCNN(nn.Module):
    """
    CNN with self-attention mechanism for focusing on important frequency-time regions.
    """
    def __init__(self, num_classes):
        super(AttentionCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Self-attention
        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        
        # Attention weights
        attn = self.attention(x)
        
        # Apply attention
        x = x * attn
        
        # Classify
        x = self.classifier(x)
        
        return x
