import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from sklearn import preprocessing
from sklearn.model_selection import KFold
from data import load_data_raw
from data import load_data_standardization
from torch.optim.lr_scheduler import ExponentialLR


input_size = 5
hidden_size = 64
output_size = 1
drop_rate = 0.5
learning_rate = 0.001
num_epochs = 2000


# Neural Network Model (2 hidden layers MLP)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_rate):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.drop = nn.Dropout(p=drop_rate)
        # self.lr_schedule = lr_schedule

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc4(out)
        return out

if __name__ == "__main__":

    #load data into numpy arrays
    #inputs, labels = load_data_raw()

    #load data with zero mean and unit variance
    inputs, labels = load_data_standardization()


    # training input and labels for Varible
    inputs_torch = torch.from_numpy(inputs).float()
    inputs_variable = Variable(inputs_torch)

    labels_torch = torch.from_numpy(labels).float()
    labels_variable = Variable(labels_torch, requires_grad=False)


    # shuffled_inputs, shuffled_labels = randomize(inputs, labels)

    # divide into 3 folders
    kf = KFold(n_splits=3, shuffle=True)

    total_loss = np.zeros((1,))
    # print(total_loss)

    for train, val in kf.split(labels):
        # Subsets of training data and validation data after cross validation split
        X_train, y_train, X_val, y_val = inputs[train], labels[train], inputs[val], labels[val]
        #train_inputs, test_inputs, train_labels, test_labels = inputs[train], inputs[test], labels[train], labels[test]


        # Loss and Optimer
        mlp = MLP(input_size, hidden_size, output_size, drop_rate=0.5)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = ExponentialLR(optimizer, gamma = 0.5)

        for epoch in range(num_epochs):

            optimizer.zero_grad()

            # torch to variable
            X_train_variable = Variable(torch.from_numpy(X_train).float())
            y_train_variable = Variable(torch.from_numpy(y_train).float())

            X_val_variable = Variable(torch.from_numpy(X_val).float())
            y_val_variable = Variable(torch.from_numpy(y_val).float())

            # train model
            mlp.train()
            outputs = mlp(X_train_variable)
            loss = criterion(outputs, y_train_variable)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 100 == 0:
                print('Epoch [%d/%d], Validation Loss: %.4f'
                      % (epoch + 1, num_epochs, loss.data[0]))

        # validate model
        mlp.eval()  # for the case of dropout-layer and Batch-normalization
        outputs = mlp(X_val_variable)
        validation_loss = criterion(outputs, y_val_variable)


        #print(validation_loss.data[0])
        #total_loss += validation_loss.data[0]
        #print(total_loss)
        print("End of one cross validation subset")
