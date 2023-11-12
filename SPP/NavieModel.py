import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from attention import SelfAttention

class NavieModel(nn.Module):
    def __init__(self):
        super(NavieModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=120, out_channels=240, kernel_size=3, padding=1)
        self.bact_norm = nn.BatchNorm1d(240,momentum=1e-3)
        self.lstm = nn.LSTM(input_size=5, hidden_size=240, batch_first=True,dropout=0.2)
        self.attention = SelfAttention(6,240)
        self.layer_norm_1 = nn.LayerNorm(480)
        
        
        self.ff_1 = nn.Linear(480,480*2)
        self.ff_2 = nn.Linear(480*2,480)
        self.ff_3 = nn.Linear(480,480//2)
        self.layer_norm_2 = nn.LayerNorm(960)
        
        self.lin_1 = nn.Linear(960,960//2)
        self.lin_2 = nn.Linear(960//2,120)
        
        self.conv1d_1 = nn.Conv1d(240,120,kernel_size=3,stride=2)
        self.conv1d_2 = nn.Conv1d(120,1,kernel_size=3,stride=2)
        
        #self.group_pool = nn.GlobalAveragePooling()
        
        self.op = nn.Linear(120,1)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.bact_norm(x)
        x, _ = self.lstm(x)
        residual = x
        x = self.attention(x)
        x = torch.cat((x,residual),dim=-1)
        x = self.layer_norm_1(x)
        residual = x
        x = F.relu(self.ff_1(x))
        x = F.relu(self.ff_2(x))
        x = torch.cat((x,residual),dim=-1)
        x = self.layer_norm_2(x)
        
        x = F.relu(self.lin_1(x))
        x = F.relu(self.lin_2(x))
        
        x = F.relu6(self.conv1d_1(x))
        x = F.relu6(self.conv1d_2(x))
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.shape[0], -1)
        #x = self.group_pool(x)
        #x = F.relu(self.op(x))
        #print(x.shape)
        #x = self.lin(x)
        return x