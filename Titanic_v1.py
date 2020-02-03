# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:49:40 2020

@author: LEE
"""

import numpy
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

# torch.manual_seed(1)

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

# data preparation


def set_missing_ages(df):
    
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])
    
    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

train_data = pd.read_csv("E:\Kaggle\Titanic\Train.csv")
train_data, rfr = set_missing_ages(train_data)
train_data = set_Cabin_type(train_data)

test_data = pd.read_csv('E:\Kaggle\Titanic\Test.csv')
test_data.loc[ (test_data.Fare.isnull()), 'Fare' ] = 0
test_data, rfr = set_missing_ages(test_data)
test_data = set_Cabin_type(test_data)

#ax = train_data.Age.plot.hist(bins=12, alpha=0.5)
'''
Agerange=train_data.Age.max()-train_data.Age.min()
for idx,age in enumerate(train_data.Age):
    if pd.notnull(age):
        train_data.Age.at[idx]=int(age/Agerange*20+1)
    else:
        train_data.Age.at[idx]=0
'''
        
train_data['Sex'].replace('female', 0,inplace=True)
train_data['Sex'].replace('male', 1,inplace=True)
train_data['Cabin'].replace('No', 0,inplace=True)
train_data['Cabin'].replace('Yes', 1,inplace=True)
dummies_Embarked = pd.get_dummies(train_data['Embarked'], prefix= 'Embarked')
train_label=torch.tensor(train_data.Survived)
train_data=train_data.drop(columns=['PassengerId','Name','Ticket','Embarked','Survived'])
train_data = pd.concat([train_data, dummies_Embarked], axis=1)
x=train_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled=min_max_scaler.fit_transform(x)
train_data=pd.DataFrame(x_scaled)
tmp = train_data.values
train_data = torch.from_numpy(tmp)


test_data['Sex'].replace('female', 0,inplace=True)
test_data['Sex'].replace('male', 1,inplace=True)
test_data['Cabin'].replace('No', 0,inplace=True)
test_data['Cabin'].replace('Yes', 1,inplace=True)
dummies_Embarked = pd.get_dummies(test_data['Embarked'], prefix= 'Embarked')
test_data=test_data.drop(columns=['PassengerId','Name','Ticket','Embarked'])
test_data = pd.concat([test_data, dummies_Embarked], axis=1)
test_x=test_data.values
min_max_scaler = preprocessing.MinMaxScaler()
testx_scaled=min_max_scaler.fit_transform(test_x)
test_data=pd.DataFrame(testx_scaled)
testtmp = test_data.values
test_data = torch.from_numpy(testtmp)
testx = test_data.type(torch.FloatTensor)
testx=Variable(testx)

# torch.manual_seed(1)    # reproducible

# make fake data
'''
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)   
'''             # class1 y data (tensor), shape=(100, 1)
x = train_data.type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = train_label.type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
x, y = Variable(x), Variable(y)

#plt.scatter(x.data.numpy()[:, 6], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
#plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x

net = Net(n_feature=10, n_hidden=32, n_output=2)     # define the network
print(net)  # net architecture

optimizer = torch.optim.Adagrad(net.parameters(), lr=0.08)
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

#plt.ion()   # something about plotting

for t in range(500):
    out = net(x)                 # input x and predict based on x
    loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 2 == 0:
        # plot and show learning process
        #plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        #plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        print(accuracy)
        #plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        #plt.pause(0.1)

#plt.ioff()
#plt.show()

out=net(testx)
prediction = torch.max(out, 1)[1]
pred_test_y = prediction.data.numpy()
d = {'PassengerId': numpy.linspace(892,1309,1309-892+1,dtype = int), 'Survived': pred_test_y}
df = pd.DataFrame(data=d)
df.to_csv(r'E:\Kaggle\Titanic\test_result.csv',index=False)
