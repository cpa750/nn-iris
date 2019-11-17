import pandas as pd
import numpy as np
from model import Model
from torch import nn, optim, tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer

from tabulate import tabulate

NUM_EPOCHS = 10

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
a = pd.DataFrame(dataset)
a.columns = ["sw", "sl", "pw", "pl", "classname"]

le = LabelEncoder()

a.classname = le.fit_transform(a.classname.values.reshape(-1,1))

data = a.values

data_length = data.shape[0]

split = int(0.8 * data_length)
idx_list = list(range(data_length))
train_idx, valid_idx = idx_list[split:], idx_list[:split]

x_tr = tensor(data[:split, :4])
y_tr = tensor(data[:split, 4])
x_val = tensor(data[split:, :4])
y_val = tensor(data[split:, 4])

train = TensorDataset(x_tr, y_tr)
train_loader = DataLoader(train, batch_size=150)

valid = TensorDataset(x_val, y_val)
valid_loader = DataLoader(valid, batch_size=150)

m = Model()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(m.parameters(), lr=.01, momentum=0.9, nesterov=True)

for epoch in range(1, NUM_EPOCHS+1):
    m.train()

    train_loss, val_loss = [], []
    for data, target in train_loader:
        print(tabulate(data))
        optimizer.zero_grad() # Setting gradients to 0
        output = m(data.float()) # feed forward
        loss = loss_function(output, target.long()) # calculating loss
        loss.backward() # backprop
        optimizer.step() # changing optimizer according to momentum
        train_loss.append(loss.item())

    m.eval()
    for data, target in valid_loader:
        output = m(data.float())
        loss = loss_function(output, target.long())
        val_loss.append(loss.item())

    print("Epoch", epoch, "Training Loss", np.mean(train_loss), "Validation Loss", val_loss, sep=":")

