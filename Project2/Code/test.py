import framework as fw
import torch as t
import math as m
import copy as c

# Generating x test and train set
x_train = t.rand((1000,2))
x_test = t.rand((1000,2))

# Generate y test and train set
radius = 1/((2*m.pi)**(1/2))
limit_up = 0.5+radius
limit_down = 0.5-radius
y_train = t.tensor([0 if ((pair[0]>limit_up or pair[1]>limit_up) or (pair[0]<limit_down or pair[1]<limit_down)) else 1 for pair in x_train])
y_train = y_train.view(y_train.size()[0],1) #training y must be in size (n,1) and not (n)
y_test = t.tensor([0 if ((pair[0]>limit_up or pair[1]>limit_up) or (pair[0]<limit_down or pair[1]<limit_down)) else 1 for pair in x_test])
y_test = y_test.view(y_train.size()[0],1)





#Example use of framework with max batch size and no accumulate gradient



### Init NN

# nn = fw.NeuralNetwork()
# nn.add(nn.Linear(2,25),nn.reLU(),nn.Linear(25,25),nn.tanH(),nn.Linear(25,1),nn.reLU())
#
# nn.train(x_train,y_train,1000,6000,0.02)
#
#
# y = nn.forward(x_test)
# print("ytrain ",t.transpose(y_train[0:20],0,1))
# print("ytest: ",t.transpose(y_test[0:20],0,1))
# print("pred: ",y[0][0:20])
# print("round: ",nn.round(y)[0][0:20])
