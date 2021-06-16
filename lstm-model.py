future_num = 144 #何足先を予測するか
feature_num = 5 #volume, open, high, low, closeの5項目
batch_size = 100
time_steps = 50 # lstmのtimesteps
moving_average_num = 500 # 移動平均を取るCandle数
n_epocs = 30 
#データをtrain, testに分割するIndex
val_idx_from = 80000
test_idx_from = 100000

lstm_hidden_dim = 16
target_dim = 1

import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
from scipy.integrate import odeint, simps
from torch.optim import SGD
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

class LSTMClassifier(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, target_dim):
        super(LSTMClassifier, self).__init__()
        self.input_dim = lstm_input_dim
        self.hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(input_size=lstm_input_dim, 
                            hidden_size=lstm_hidden_dim,
                            num_layers=1, #default
                            #dropout=0.2,
                            batch_first=True
                            )
        self.dense = nn.Linear(lstm_hidden_dim, target_dim)

    def forward(self, X_input):
        _, lstm_out = self.lstm(X_input)
        # LSTMの最終出力のみを利用する。
        linear_out = self.dense(lstm_out[0].view(X_input.size(0), -1))
        return torch.sigmoid(linear_out)


def duffing(var, t, gamma, a, b, F0, omega, delta):
    """
    var = [x, p]
    dx/dt = p
    dp/dt = -gamma*p + 2*a*x - 4*b*x**3 + F0*cos(omega*t + delta)
    """
    x_dot = var[1]
    p_dot = -gamma * var[1] + 2 * a * var[0] - 4 * b * var[0]**3 + F0 * np.cos(omega * t + delta)

    return np.array([x_dot, p_dot])

def create_duffing():
    F0, gamma, omega, delta = 10, 0.1, np.pi / 3, 1.5 * np.pi
    a, b = 1 / 4, 1 / 2
    var, var_lin = [[0, 1]] * 2

    # timescale
    t = np.arange(0, 20000, 2 * np.pi / omega)
    t_lin = np.linspace(0, 1000, 10000)

    # solve
    var = odeint(duffing, var, t, args=(gamma, a, b, F0, omega, delta))
    var_lin = odeint(duffing, var_lin, t_lin, args=(gamma, a, b, F0, omega, delta))

    x_lin, p_lin = var_lin.T[0], var_lin.T[1]
    return x_lin, t_lin, p_lin

def _load_data(data, n_prev = 100):  
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1, n_prev = 100):  
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)
    X_train = X_train.astype('int')
    y_train = y_train.astype('float')
    X_test = X_test.astype('int')
    y_test = y_test.astype('float')
    return (X_train, y_train), (X_test, y_test)

class Predictor(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Predictor, self).__init__()

        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True)
        self.output_layer = nn.Linear(hiddenDim, outputDim)
    
    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.rnn(inputs, hidden0)
        output = self.output_layer(output[:, -1, :])

        return output

def mkRandomBatch(train_x, train_t, batch_size=10):
    """
    train_x, train_tを受け取ってbatch_x, batch_tを返す。
    """
    batch_x = []
    batch_t = []

    for _ in range(batch_size):
        idx = np.random.randint(0, len(train_x) - 1)
        batch_x.append(train_x[idx])
        batch_t.append(train_t[idx])
    
    return torch.stack(batch_x, dim=0), torch.stack(batch_t, dim=0)


def main():
    training_size = 10000 #traning dataのデータ数
    epochs_num = 30 #traningのepoch回数
    hidden_size = 5 #LSTMの隠れ層の次元数

    x_lin, t_lin, p_lin = create_duffing()
    (train_x, train_t),(test_x,test_t) = train_test_split(pd.DataFrame(x_lin)) #Datasetの作成
    train_x = torch.from_numpy(train_x).float()
    train_t = torch.from_numpy(train_t).float()
    test_x = torch.from_numpy(test_x).float()
    test_t = torch.from_numpy(test_t).float() 
    model = Predictor(1, hidden_size, 1) #modelの宣言

    criterion = nn.MSELoss() #評価関数の宣言
    optimizer = SGD(model.parameters(), lr=0.01) #最適化関数の宣言

    predict=[]
    predict_t=[]
    true_label=[]
    for epoch in range(epochs_num):
        # training
        running_loss = 0.0
        training_accuracy = 0.0
        for i in range(int(training_size / batch_size)):
            optimizer.zero_grad()

            data, label = mkRandomBatch(train_x, train_t, batch_size)
            true_label = label

            output = model(data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()
            training_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.1) #outputとlabelの誤差が0.1以内なら正しいとみなす。
            predict=output.data
            predict_t=train_t
        training_accuracy /= training_size
        print('%d loss: %.3f, training_accuracy: %.5f' % (epoch + 1, running_loss, training_accuracy))
    dataf =  pd.DataFrame(predict.to('cpu').detach().numpy().copy())
    print(true_label)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    dataf.columns = ["predict"]
    dataf["label"] = true_label.to('cpu').detach().numpy().copy()
    plt.plot(range(len(dataf)),dataf["predict"], color='b',)
    plt.plot(range(len(dataf)),dataf["label"], color='#e46409')
    plt.savefig("image.png")


if __name__ == '__main__':
    main()