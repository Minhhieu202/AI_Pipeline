import torch
import torch.nn as nn
import torch.nn.functional as F

class MYCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = self.make_block(in_channels=3, out_channel=8)
        self.conv2 = self.make_block(in_channels=8, out_channel=16)
        self.conv3 = self.make_block(in_channels=16, out_channel=32)
        self.conv4 = self.make_block(in_channels=32, out_channel=64)
        self.conv5 = self.make_block(in_channels=64, out_channel=64)


        self.flatten_size = 64 * (224 // 2**5) * (224 // 2**5)  

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=self.flatten_size, out_features=1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

    def make_block(self, in_channels, out_channel, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  
        x = self.conv3(x)  
        x = self.conv4(x)  
        x = self.conv5(x)  

        x = x.view(x.shape[0], -1)  

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

if __name__ == '__main__':
    toy_data = torch.rand(16, 3, 224, 224) 
    print("Input shape:", toy_data.shape)

    model = MYCNN(num_classes=12)
    output = model(toy_data)
    
    print("Output shape:", output.shape)  