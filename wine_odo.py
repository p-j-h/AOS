import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR

class MultiSensorNet(nn.Module):
    def __init__(self, lstm_config, fc_configs):
        super(MultiSensorNet, self).__init__()

        # Depthwise CNN
        # self.depthwise_cnn_layers = nn.ModuleList()
        # prev_channels = 48
        # for config in depthwise_configs:
        #     self.depthwise_cnn_layers.append(nn.Conv1d(in_channels=48, out_channels=48,
        #                                               kernel_size=config.get('kernel_size', 100),
        #                                               padding=config.get('padding', 50),
        #                                               groups=48))
        #     nn.init.kaiming_normal_(self.depthwise_cnn_layers[-1].weight, nonlinearity='relu')
        #     self.depthwise_cnn_layers.append(nn.Dropout(p=0.5))
        #     self.depthwise_cnn_layers.append(nn.ReLU())

        # LSTM
        self.lstm = nn.LSTM(input_size=48,
                            hidden_size=lstm_config.get('hidden_size', 64),
                            num_layers=lstm_config.get('num_layers', 1),
                            batch_first=True)

        # 2D CNN
        # self.cnn_layers = nn.ModuleList()
        # prev_channels = lstm_config.get('hidden_size', 64)
        # for config in cnn_configs:
        #    self.cnn_layers.append(nn.Conv2d(prev_channels, config['out_channels'], 
        #                                     kernel_size=config.get('kernel_size', 3), 
        #                                     padding=config.get('padding', 1)))
        #    self.cnn_layers.append(nn.ReLU())
        #    self.cnn_layers.append(nn.Dropout2d(p=config.get('dropout', 0.5)))
        #    prev_channels = config['out_channels']

        # Fully Connected Layers (FCN)
        self.fc_layers = nn.ModuleList()
        prev_fc_size = lstm_config.get('hidden_size', 64) * 841
        #prev_fc_size = 48 * 843
        for config in fc_configs:
            self.fc_layers.append(nn.Linear(prev_fc_size, config['output_dim']))
            nn.init.kaiming_normal_(self.fc_layers[-1].weight, nonlinearity='relu')
            #self.fc_layers.append(nn.BatchNorm1d(config['output_dim']))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(p=0.3))
            prev_fc_size = config['output_dim']

    def forward(self, x):
        # Depthwise CNN
        # for layer in self.depthwise_cnn_layers:
        #     x = layer(x)

        #LSTM
        #x = x.transpose(1,2)  # Change to [batch_size, time_steps, lstm_input_size] for LSTM
        x, _ = self.lstm(x.transpose(1,2))
        #x = x.permute(0, 2, 1).unsqueeze(3)  # Change to [batch_size, lstm_output_size, time_steps] for 2D CNN

        # for layer in self.cnn_layers:
        #     x = layer(x)

        #x = x.view(x.size(0), -1)  # Flatten
        x = x.reshape(x.size(0), -1)
        #fc_input_size = x.size(1)
        #self.fc_layers[0] = nn.Linear(fc_input_size, self.fc_layers[0].out_features).to(x.device)
        # Fully Connected Layers (FCN)
        for layer in self.fc_layers:
            x = layer(x)

        return x
    
# misc
depthwise_configs = [{'kernel_size': 50, 'padding': 25}, {'kernel_size': 30, 'padding': 15}]
lstm_config = {'hidden_size': 64, 'num_layers': 1}
cnn_configs = [{'out_channels': 64, 'kernel_size': 20, 'padding': 10}, {'out_channels': 128, 'kernel_size': 10, 'padding': 5}]
#fc_configs = [{'output_dim': 500},{'output_dim': 300},{'output_dim': 100}, {'output_dim': 10}]
fc_configs = [{'output_dim': 300},{'output_dim': 150}, {'output_dim': 10}]

#model = MultiSensorNet(depthwise_configs, lstm_config, cnn_configs, fc_configs)
model = MultiSensorNet(lstm_config, fc_configs)
data_path = 'X:\\Dropbox\\003 Tactile\\chem\\data\\wine_concat\\concat_data.csv'
df = pd.read_csv(data_path)

# dataset divide
train_df = df[df['trial'].isin([1, 2, 3])]
val_df = df[df['trial'] == 4]
test_df = df[df['trial'] == 5]

class WineDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.sequence_length = 841 # 0.5sec * (841-1) = 420sec
        # 48-dimensional input (12 readings x 4 sensor types)
        num_sensors = 12
        num_sensor_types = 4
        self.sensor_data = np.zeros((len(self.data), num_sensors * num_sensor_types))

        for i in range(num_sensor_types):
            self.sensor_data[:, i*num_sensors:(i+1)*num_sensors] = self.data.iloc[:, 1:1+num_sensors].values

            #self.sensor_data[:, i*num_sensors:(i+1)*num_sensors] = self.data[:, 1:1+num_sensors]

        self.labels = self.data.iloc[:, -3].values - 1 # labels start from 1
        self.scaler = StandardScaler()
        self.sensor_data = self.scaler.fit_transform(self.sensor_data)

    def __len__(self):
        return len(self.sensor_data) // self.sequence_length

    def __getitem__(self, idx):
        start_idx = idx * self.sequence_length
        end_idx = (idx + 1) * self.sequence_length
    
        # Check if we're going out of bounds and if so, just return the last sequence_length worth of data
        if end_idx > len(self.data):
            start_idx = len(self.data) - self.sequence_length
            end_idx = len(self.data)
        
        sample_data = self.sensor_data[start_idx:end_idx, :]
        sample_data = sample_data.transpose(1, 0).astype(np.float32)
        label = self.labels[start_idx]
        return sample_data, label
    
train_dataset = WineDataset(dataframe = train_df, transform=None)
val_dataset = WineDataset(dataframe=val_df, transform=None)
test_dataset = WineDataset(dataframe=test_df, transform=None)

train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=48, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False)



epochs = 200
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
device = next(model.parameters()).device
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=lr)
#scheduler = MultiStepLR(optimizer, milestones=[20,60,150], gamma=0.1)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# train and validate function
def train(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        #scheduler.step()
        total_loss += loss.item()
    print(f"Epoch: {epoch}, Loss: {total_loss / len(train_loader)}")

def validate(epoch):
    model.eval()
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = 100. * correct / len(val_loader.dataset)
    print(f"Epoch: {epoch}, Validation Accuracy: {accuracy:.2f}%")

    # cm = confusion_matrix(all_targets, all_preds)
    # display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    # display.plot()
    # plt.show()

for epoch in range(1, epochs + 1):
    train(epoch)
    validate(epoch)


# test function
def test():
    model.eval()
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test Accuracy: {accuracy:.2f}%")

    cm = confusion_matrix(all_targets, all_preds)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    display.plot()
    plt.show()

test()

torch.save(model.state_dict(),'X:\\Dropbox\\003 Tactile\\chem\\data\\wine_concat\\wine_concat_1.pth')
torch.save(model,'X:\\Dropbox\\003 Tactile\\chem\\data\\wine_concat\\wine_concat_whole_1.pth')