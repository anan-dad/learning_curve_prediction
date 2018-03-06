import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable
from sklearn import preprocessing
from sklearn.model_selection import KFold
from data import load_data_raw
from data import load_data_standardization
from data import load_data_learning_curve
from torch.optim.lr_scheduler import ExponentialLR


HIDDEN_SIZE = 64
OUTPUT_SIZE = 1
DROP_RATE = 0.5
BATCH_SIZE = 32
LERANING_RATE = 0.001
NUM_EPOCHS = 200


class RNN(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, n_epochs=5):
        super(RNN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc4 = nn.LSTM(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, num_layers=2)
        self.fc5 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.drop = nn.Dropout(p=DROP_RATE)

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
        out, (h_n, h_c) = self.fc4(out)
        out = self.fc5(out[:,:,-1])
        #out = self.fc5(out[:,-1,:])
        return out



if __name__ == "__main__":

    #N_EPOCHS = np.random.randint(5, 20)
    N_EPOCHS = 5
    INPUT_SIZE = N_EPOCHS + 5
    print('N_EPOCHS is', N_EPOCHS)

    inputs, labels = load_data_learning_curve(n_epochs=N_EPOCHS)


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

        train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

        # Loss and Optimer
        rnn = RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, n_epochs=N_EPOCHS)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(rnn.parameters(), lr=LERANING_RATE, weight_decay=0.01)
        scheduler = ExponentialLR(optimizer, gamma = 0.5)

        for epoch in range(NUM_EPOCHS):

            optimizer.zero_grad()

            # torch to variable
            X_train_variable = Variable(torch.from_numpy(X_train).float())
            y_train_variable = Variable(torch.from_numpy(y_train).float())

            X_val_variable = Variable(torch.from_numpy(X_val).float())
            y_val_variable = Variable(torch.from_numpy(y_val).float())

            # train model
            rnn.train()
            outputs = rnn(X_train_variable)
            loss = criterion(outputs, y_train_variable)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 100 == 0:
                print('Epoch [%d/%d], Validation Loss: %.4f'
                      % (epoch + 1, NUM_EPOCHS, loss.data[0]))

        # validate model
        rnn.eval()  # for the case of dropout-layer and Batch-normalization
        outputs = rnn(X_val_variable)
        validation_loss = criterion(outputs, y_val_variable)


        #print(validation_loss.data[0])
        #total_loss += validation_loss.data[0]
        #print(total_loss)
        print("This is the end of rnn one cv subset  ")