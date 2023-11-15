import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from attention import SelfAttention
from tqdm import tqdm
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
        x = torch.squeeze(x)
        
        return x
def MyDataset_Formater(batch_size=None):
    # Replace 'your_file.xlsx' with the actual path to your Excel file
    excel_file_path = '/home/pradhyumna/Stock_prize_Prediction/Data/Stocks.xlsx'

    # Read all sheets into a dictionary of DataFrames
    all_sheets = pd.read_excel(excel_file_path, sheet_name=None)

    # Now, all_sheets is a dictionary where keys are sheet names and values are DataFrames
    # You can access a specific sheet using its name, for example:
    first_sheet_name = list(all_sheets.keys())
    
    df_amazon= all_sheets[first_sheet_name[0]]
    
    df_amazon = df_amazon[df_amazon['Date'].dt.year>2018]
    df_amazon = df_amazon.drop('Date', axis=1)
    
    means = df_amazon.mean(axis=0)
    df_amazon -= means
    std = df_amazon.std(axis=0)
    df_amazon /= std
    
    train_size = int(0.8 * len(df_amazon))
    train_df = df_amazon.iloc[:train_size]
    #test_df = df_amazon.iloc[train_size:]
    
    
    
    sampling_rate = 2
    sequence_length = 120
    delay = sampling_rate * (sequence_length + 24 - 1)
    
    data = train_df.iloc[:-delay].to_numpy()
    targets = train_df['Open'].iloc[delay:].to_numpy()
    
    #batch_size = 256
    dataset = TimeSeriesDatasetGenerater(data, targets, sequence_length, sequence_stride=2, sampling_rate = sampling_rate, batch_size = batch_size, shuffle = False, seed=None, start_index=0, end_index=566-sequence_length)
    return dataset


if __name__ == '__main__':
    x = torch.randn((566, 5))
    y = torch.randn((566,))
    from TimeSeriesDatasetGenerater  import TimeSeriesDatasetGenerater
    
    dataset = MyDataset_Formater(256)
    #print(dataset.__getitem__(20))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset.batch_size, shuffle=dataset.shuffle)
    
    NM = NavieModel().to('cuda')
    optima = torch.optim.SGD(NM.parameters(), lr=0.01)
    loss_function = nn.MSELoss()
    
    epochs = 10*10
    for epoch in range(epochs):
        
        for batch in tqdm(dataloader):
            x,y = batch
            x=x.to('cuda').float()
            y=y.to('cuda').float()
            x_ = NM(x)
            loss = loss_function(x_, y)
            optima.zero_grad()
            loss.backward()
            optima.step()
        print(f'Epoch:{epoch+1},Loss:{loss :.4f}')
        
        if (epoch+1)%10 == 0:
            def save_checkpoint(checkpoint_path):
                torch.save({
                    'model_state_dict': NM.state_dict(),
                    'optimizer_state_dict': optima.state_dict(),
                }, checkpoint_path)
                
            save_checkpoint(f'/home/pradhyumna/Stock_prize_Prediction/Models/Navie_Model_V1/checkpoints/checkpoint_{epoch+1}.pt')
            print(f"Checkpoint saved @ {epoch+1}")