import torch  # torch provides basic functions, from setting a random seed (for reproducability) to creating tensors.
import torch.nn as nn  # torch.nn allows us to create a neural network.
import torch.nn.functional as F  # nn.functional give us access to the activation and loss functions.
from torch.optim import SGD  ## optim contains many optimizers. Here, we're using SGD, stochastic gradient descent.
import matplotlib.pyplot as plt  ## matplotlib allows us to draw graphs.
import seaborn as sns  ## seaborn makes it easier to draw nice-looking graphs.
import os
from tqdm import tqdm
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing

import pandas
import numpy as np


Bankruptcy_Data = pandas.read_excel('/Users/carstenjuliansavage/Desktop/R Working Directory/MasterDataforML.xlsx')
pandas.set_option('display.max_columns', None)

# Filtering dataset for input and output variables only

Bankruptcy_Data_Slim = (Bankruptcy_Data
    .filter(['cash_debt', 'curr_debt', 'int_totdebt', 'quick_ratio', 'de_ratio', 'debt_assets', 'intcov','isBankrupt'])
    #.dropna()
)

X = Bankruptcy_Data_Slim[['cash_debt',
                          'curr_debt',
                          'int_totdebt',
                          'quick_ratio',
                          'de_ratio',
                          'debt_assets',
                          'intcov']]
y = Bankruptcy_Data_Slim[['isBankrupt']]

# Scaling the data to be between 0 and 1
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
y = min_max_scaler.fit_transform(y)

# Split dataframe into training and testing data. Remember to set a seed.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

# Let's confirm that the scaling worked as intended.
# All values should be between 0 and 1 for all variables.
X_Stats = pandas.DataFrame(X)
X_Stats.describe()

# Convert to float Tensor
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.squeeze(torch.from_numpy(y_train).float())
y_test = torch.squeeze(torch.from_numpy(y_test).float())


# Initializing the neural network class
class Net(nn.Module):

  def __init__(self, n_features):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_features, 12)
    self.fc2 = nn.Linear(12, 8)
    self.fc3 = nn.Linear(8, 1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return torch.sigmoid(self.fc3(x))
net = Net(X_train.shape[1])

# Loss Function
criterion = nn.CrossEntropyLoss()
optimizer = SGD(net.parameters(), lr=0.1)  ## here we're creating an optimizer to train the neural network.
#This learning rate seems to be working well so far

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X_train = X_train.to(device)
y_train = y_train.to(device)

X_test = X_test.to(device)
y_test = y_test.to(device)
net = net.to(device)
criterion = criterion.to(device)

def calculate_accuracy(y_true, y_pred):
  predicted = y_pred.ge(.5).view(-1)
  return (y_true == predicted).sum().float() / len(y_true)

def round_tensor(t, decimal_places=3):
  return round(t.item(), decimal_places)

for epoch in range(1000):

    y_pred = net(X_train)
    y_pred = torch.squeeze(y_pred)
    train_loss = criterion(y_pred, y_train)
    train_acc = calculate_accuracy(y_train, y_pred)
    y_test_pred = net(X_test)
    y_test_pred = torch.squeeze(y_test_pred)
    test_loss = criterion(y_test_pred, y_test)
    test_acc = calculate_accuracy(y_test, y_test_pred)

    print(f'''    Epoch {epoch}
    Training loss: {round_tensor(train_loss)} Accuracy: {round_tensor(train_acc)}
    Testing loss: {round_tensor(test_loss)} Accuracy: {round_tensor(test_acc)}''')

# If test loss is less than 0.02, then break. That result is satisfactory.
    if test_loss < 0.02:
        print("Num steps: " + str(epoch))
        break

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()


# Creating a function to evaluate our input
def AreWeBankrupt(cash_debt,
                  curr_debt,
                  int_totdebt,
                  quick_ratio,
                  de_ratio,
                  debt_assets,
                  intcov
                   ):
  t = torch.as_tensor([cash_debt,
                       curr_debt,
                       int_totdebt,
                       quick_ratio,
                       de_ratio,
                       debt_assets,
                       intcov
                       ]) \
    .float() \
    .to(device)
  output = net(t)
  return output.ge(0.5).item()


AreWeBankrupt(cash_debt=0,curr_debt=0,int_totdebt=0,quick_ratio=0,de_ratio=0,debt_assets=0,intcov=0)
AreWeBankrupt(cash_debt=1,curr_debt=1,int_totdebt=1,quick_ratio=0,de_ratio=1,debt_assets=1,intcov=1)