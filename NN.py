import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split

EPOCH = 10000

# Load the data
data = pd.read_csv('./Dataset/heart.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data to tensors
X_train = torch.FloatTensor(X_train.values)
X_test = torch.FloatTensor(X_test.values)
y_train = torch.LongTensor(y_train.values)
y_test = torch.LongTensor(y_test.values)

# Define the model
class NeuroNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = NeuroNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_func(y_pred, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch} Loss: {loss.item()}')
        
# Test the model
with torch.no_grad():
    y_pred = model(X_test)
    loss = loss_func(y_pred, y_test)
    print(f'Loss: {loss.item()}')

    correct = (y_pred.argmax(1) == y_test).type(torch.float32).sum().item()
    print(f'Accuracy: {correct / len(y_test)}')
    
