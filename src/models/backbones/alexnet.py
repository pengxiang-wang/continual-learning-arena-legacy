from torch import nn
import numpy as np

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class AlexNet(nn.Module):
    def __init__(self, input_channels, size):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=size//8)
        s = compute_conv_output_size(size, size//8)
        s = s // 2
        self.activation1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=size//10)
        s = compute_conv_output_size(s, size//10)
        s = s // 2
        self.activation2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.activation3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(256*s*s, 2048)
        self.activation4 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(2048, 2048)
        self.activation5 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        
        # self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=96, kernel_size=11, stride=4)
        # self.activation1 = nn.ReLU()
        # self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        # self.activation2 = nn.ReLU()
        # self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        # self.activation3 = nn.ReLU()
        
        # self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        # self.activation4 = nn.ReLU()
        
        # self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        # self.activation5 = nn.ReLU()
        
        # self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # self.fc1 = nn.Linear(256*5*5, 4096)
        # self.activation6 = nn.ReLU()
        # self.dropout1 = nn.Dropout(0.5)

        # self.fc2 = nn.Linear(4096, 4096)
        # self.activation7 = nn.ReLU()
        # self.dropout2 = nn.Dropout(0.5)


        
        

    def forward(self, x):
        h = self.conv1(x)
        a = self.activation1(h)
        a = self.maxpool1(a)
        
        h = self.conv2(a)
        a = self.activation2(h)
        a = self.maxpool2(a)
        
        h = self.conv3(a)
        a = self.activation3(h)
        a = self.maxpool3(a)
        
        a = a.reshape(a.shape[0], -1)

        
        h = self.fc1(a)
        a = self.activation4(h)
        a = self.dropout1(a)

                
        h = self.fc2(a)
        a = self.activation5(h)
        a = self.dropout2(a)

        
        return a


if __name__ == "__main__":
    _ = AlexNet()
