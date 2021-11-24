import framework as fw
import torch as t
import math as m


# Generating x test and train set
x_train = t.rand((1000,2))
x_test = t.rand((1000,2))

# Generate y test and train set
radius = 1/((2*m.pi)**(1/2))
limit_up = 0.5+radius
limit_down = 0.5-radius
y_train = t.tensor([0 if ((pair[0]>limit_up or pair[1]>limit_up) or (pair[0]<limit_down or pair[1]<limit_down)) else 1 for pair in x_train])
y_train = y_train.view(y_train.size()[0],1) #training y must be in size (n,1)
y_test = t.tensor([0 if ((pair[0]>limit_up or pair[1]>limit_up) or (pair[0]<limit_down or pair[1]<limit_down)) else 1 for pair in x_test])
y_test = y_test.view(y_train.size()[0],1)


#Example use of framework
nn = fw.NeuralNetwork()
nn.add(nn.Linear(2,25),nn.reLU(),nn.Linear(25,25),nn.tanH(),nn.Linear(25,1),nn.tanH())
dataset = t.rand((1000,2))
operations = nn.train(dataset,y_train,1000,100)

# Compute accuracy
y = nn.forward(x_test)
prediction = (y.flatten()==y_test.flatten()).tolist()
N = len(prediction)
true_positif = prediction.count(True)
print("Accuracy = ",true_positif/N,"%")