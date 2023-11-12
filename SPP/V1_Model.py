import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from attention import SelfAttention

class Conv_Up(nn.Module):
    def __init__(self,in_channel,kernel_size,padding):
        super(Conv_Up, self).__init__()
        self.conv_up = nn.Sequential(
            nn.Conv1d(in_channel,in_channel*2,kernel_size=kernel_size,padding=padding),
            nn.LeakyReLU(),
            nn.Conv1d(in_channel*2,in_channel,kernel_size=kernel_size,padding=padding),
            nn.BatchNorm1d(in_channel),
            nn.Conv1d(in_channel,in_channel*2,kernel_size=kernel_size,padding=padding),
            nn.SiLU()
        )
    def forward(self,input):
        input = self.conv_up(input)
        return input

class LSTM_UP(nn.Module):
    def __init__(self,input_size,hidden_size,dropout_rate,batch_first=True):
        super(LSTM_UP, self).__init__()
        self.lstm_up_1 = nn.LSTM(input_size, hidden_size=hidden_size//2, batch_first=batch_first,dropout=dropout_rate)
        self.lstm_up_2 = nn.LSTM(hidden_size//2, hidden_size=hidden_size, batch_first=batch_first,dropout=dropout_rate)
    def forward(self, x):
        x,_ = self.lstm_up_1(x)
        x = F.silu(x)
        x,_ = self.lstm_up_2(x)
        x = F.silu(x)
        return x
    
class Up_Modeling(nn.Module):
    def __init__(self,seq_len):
        super(Up_Modeling, self).__init__()
        #seq_len = 120
        # (bs,seq_len,dim) => (bs,2*seq_len,dim)
        self.conv_up_1 = Conv_Up(in_channel=seq_len,kernel_size=3,padding=1)
        # (bs,2*seq_len,dim) => (bs,2*seq_len,2*seq_len)
        self.lstm_up_1 = LSTM_UP(seq_len,seq_len*2,dropout_rate=0.2)
        # (bs,2*seq_len,2*seq_len) => (bs,2*seq_len,2*seq_len)
        self.attention_1 = SelfAttention(6,seq_len*2)
        self.layer_norm_1 = nn.LayerNorm(seq_len*2*2)
        self.conv_1 = nn.Conv1d(seq_len*2,seq_len,kernel_size=3,padding=1,stride=2)
        self.pool_1 = nn.MaxPool1d(kernel_size=2)
        #self.conv_2 = nn.Conv1d(seq_len,seq_len,kernel_size=3,padding=1,stride=2)
    def forward(self,x):
        x = self.conv_up_1(x)
        x = self.lstm_up_1(x)
        
        residual = x
        x = self.attention_1(x)
        x = torch.cat((x,residual),dim=-1)
        x = self.layer_norm_1(x)
        
        x = self.conv_1(x)
        x = F.silu(x)
        x = self.pool_1(x)
        return x

class V1_Model(nn.Module):
    def __init__(self):
        super(V1_Model, self).__init__()
        seq_len = 120
        self.lstm_up_init_1 = LSTM_UP(5,seq_len,dropout_rate=0.2)
        
        self.up_m_1 = Up_Modeling(seq_len)
        self.lstm_up_1 = LSTM_UP(seq_len,2*seq_len,dropout_rate=0.2)
        
        self.up_m_2 = Up_Modeling(2*seq_len)
        self.lstm_up_2 = LSTM_UP(2*seq_len,2*2*seq_len,dropout_rate=0.2)
        
        self.conv_down_1 = nn.Conv1d(2*2*seq_len,2*seq_len,kernel_size=3,padding=1,stride=2)
        self.lstm_down_1 = nn.LSTM(2*seq_len, hidden_size=seq_len,dropout=0.2)
        
        self.conv_down_2 = nn.Conv1d(2*seq_len,seq_len//2,kernel_size=3,padding=1,stride=2)
        
        self.attention_down_1 = SelfAttention(3,60)
        
        self.conv_down_3 = nn.Conv1d(seq_len,seq_len//2,kernel_size=3,padding=1)
        
        self.lstm_down_3 = nn.LSTM(seq_len//2,hidden_size=seq_len//4,dropout=0.2)
        
        self.conv_final = nn.Conv1d(seq_len//2,1,kernel_size=3,padding=1)
        #self.linear_final = nn.Linear(seq_len//2,1)
    def forward(self,x):
        #(bs,seq_len,dim) => (bs,seq_len,seq_len)
        x = self.lstm_up_init_1(x) 
        print(f'X:{x.shape}')
        
        residual = x
        #(bs,seq_len,seq_len)=> (bs,seq_len,seq_len)
        x = self.up_m_1(x)
        #(bs,seq_len,seq_len)=> (bs,2*seq_len,seq_len)
        x = torch.cat((x,residual),dim=1)
        #(bs,2*seq_len,seq_len)=> (bs,2*seq_len,2*seq_len)
        x = self.lstm_up_1(x)
        
        residual = x
        x = self.up_m_2(x)
        x = torch.cat((x,residual),dim=1)
        x = self.lstm_up_2(x)
        
        x = F.silu(self.conv_down_1(x))
        x,_ = self.lstm_down_1(x)
        x = F.tanh(x)
        
        x = F.silu(self.conv_down_2(x))
        
        residual = x
        x = self.attention_down_1(x)
        x = torch.cat((x,residual),dim=1)
        
        x = F.silu(self.conv_down_3(x))
        x = F.silu(self.lstm_down_3(x)[0])
        
        x = F.silu(self.conv_final(x))
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.shape[0], -1)
        print(f'X:{x.shape}')
        return x
    
if __name__ == '__main__':
    x = torch.randn((566, 5))
    y = torch.randn((566,))
    from TimeSeriesDatasetGenerater  import TimeSeriesDatasetGenerater
    
    sampling_rate = 2
    sequence_length = 120
    delay = sampling_rate * (sequence_length + 24 - 1)
    batch_size = 256
    dataset = TimeSeriesDatasetGenerater(x, y, sequence_length, sequence_stride=2, sampling_rate = sampling_rate, batch_size = batch_size, shuffle = False, seed=None, start_index=0, end_index=566-sequence_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset.batch_size, shuffle=dataset.shuffle)
    
    vm = V1_Model()#.to('cuda')
    
    for batch in dataloader:
        x,y = batch
        #x,y = x.to('cuda'),y.to('cuda')
        print(x.shape,y.shape)
        x = vm(x)
        break
    