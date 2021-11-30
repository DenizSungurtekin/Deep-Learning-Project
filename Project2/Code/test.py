import framework as fw
import torch as t
import math as m


def generateData():
    # Generating x test and train set
    x_train = t.rand((1000,2))
    x_test = t.rand((1000,2))

    # Generate y test and y train set
    radius = 1/((2*m.pi)**(1/2)) # radius of the circle
    limit_up = 0.5+radius # Limit up and down of the circle
    limit_down = 0.5-radius

    y_train = t.tensor([0 if ((pair[0]>limit_up or pair[1]>limit_up) or (pair[0]<limit_down or pair[1]<limit_down)) else 1 for pair in x_train])
    y_train = y_train.view(y_train.size()[0],1) #training y must be in size (n,1) and not (n)
    y_test = t.tensor([0 if ((pair[0]>limit_up or pair[1]>limit_up) or (pair[0]<limit_down or pair[1]<limit_down)) else 1 for pair in x_test])
    y_test = y_test.view(y_train.size()[0],1)

    return x_train,y_train,x_test,y_test


x_train,y_train,x_test,y_test = generateData()
# Load a model
nn = fw.NeuralNetwork()
myNN = nn.loadModel("TRR3HL25b50")

# Compute accuracy
batch_size = 50 #Needed because the model was trained with this batch size and the dimension of the bias depend on this batch size
myNN.computeAcc(x_test,y_test,batch_size) # A batch size is specified but the accuracy is computed on the entier dataset