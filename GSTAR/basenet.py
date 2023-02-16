import torch.nn as nn
import torchvision

class ResNet50(nn.Module):   
    def __init__(self, n_classes, pretrained = True, hidden_size=2048, dropout=0.5):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)                
        self.resnet.fc = nn.Linear(2048, hidden_size)
        self.dense = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)        

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(self.relu(x))
        x = self.dense(x)

        return x