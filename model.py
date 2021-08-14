import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD




class Net(nn.Module) :
    def __init__(self) :
        super(Net,self).__init__()
        self.pool = nn.MaxPool2d(2,2)
        ## output size = (W-F)/S +1 
        ##input shape : 1,224,224
        self.conv1 = nn.Conv2d(1,68,3)
        ## output size = (W-F)/S +1 = 68,222,222
        ## after pooling = 68,111,111
        
        self.conv2 = nn.Conv2d(68,136,3)
        ## output size = (W-F)/S +1 = 136,109,109
        ## after pooling = 136,54,54
        
        self.conv3 = nn.Conv2d(136,272,3)
        ## output size = (W-F)/S +1 = 272,52,52
        ## after pooling = 272,26,26
        
        self.conv4 = nn.Conv2d(272,544,3)
        ## output size = (W-F)/S +1 = 544,24,24
        ## after pooling = 544,12,12
        
        self.conv5 = nn.Conv2d(544,1088,3)
        ## output size = (W-F)/S +1 = 1088,10,10
        ## after pooling = 1088,5,5
        
        self.conv6 = nn.Conv2d(1088,2176,3)
        ## output size = (W-F)/S +1 = 2176,3,3
        ## after pooling = 2176,1,1
        
        self.fc1 = nn.Linear(2176,1000)
        self.fc2 = nn.Linear(1000,1000)
        self.fc3 = nn.Linear(1000,136)
    
    
    def forward(self , x) :
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        
        x = x.view(x.shape[0] , -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x