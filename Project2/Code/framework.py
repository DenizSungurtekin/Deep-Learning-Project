import torch as t
import copy as c
import pickle
import warnings


class NeuralNetwork(object):

    def __init__(self):
        self.operations = []
        self.memory = []
        self.fls = [] #pile

    class Linear():
        def __init__(self,in_features,out_features,bias=True):
            self.in_features = in_features
            self.out_features = out_features
            self.bias = bias
            self.bool = False # indicate if bias has been broadcasted
            self.type = 1 # int to know which operation we use during backward

            if self.bias:
                self.bias = t.tensor([1/in_features]) # pytorch initialize the bias like this, we will broadcast this value to obtain the correct shape to add it
            else:
                self.bias = t.tensor([0])

            self.weights = t.rand((out_features,in_features))

        def evaluation(self,input):
            mul = t.matmul(self.weights,input)
            if not self.bool:
                self.bias = t.full(mul.size(),self.bias[0])
                self.bool = True

            output = t.add(mul,self.bias) # Wx' + b = output
            return output

    def add(self,*operation):
        for op in operation:
            self.operations.append(op)

    class reLU():
        def __init__(self):
            self.type = 2

        def evaluation(self,input):
            return input.apply_(lambda x: (max(0, x)))

        def mapFunction(self,x): #used in derivative to derive Relu
            if x > 0:
                res = 1
            elif x <= 0:
                res = 0
            return res

        def derivative(self,input):
            return input.apply_(lambda x: (self.mapFunction(x)))

    class noActivation(): # because the backpropagation supose that after the linear transformation we have an activation function, this class represent a "non activation function" if we dont want to use activation function
        def __init__(self):
            self.type = 2

        def evaluation(self,input):
            return input

        def derivative(self,input):
            size = input.size()
            return t.ones(size)

    class sigmoid():
        def __init__(self):
            self.type = 2

        def evaluation(self,input):
            return 1/(1+t.exp(-input))

        def derivative(self,input):
            size = input.size()
            ones = t.ones(size)
            return t.mul(self.evaluation(input),(ones-self.evaluation(input)))

    class tanH():
        def __init__(self):
            self.type = 2

        def evaluation(self,input):
            return (t.exp(input)-t.exp(-input) )/ (t.exp(input) + t.exp(-input))

        def derivative(self,input):
            return (1 - (self.evaluation(input)**2))

    def forward(self,input):#add y_train to see precision at each step
        input = t.transpose(input,0,1) #transpose
        self.memory = [] # reset memory for each FF # In example = [x0,W1x0,x1,w2x1]
        self.fls = []
        for operation in self.operations:

            self.memory.append(input) #Store x before FL or z before activation function
            if operation.type == 1:
                self.fls.append(operation)
            input = operation.evaluation(input)

        size = input.size() # Just to check if we dont have a zeros tensor as output because with reLu its mean the neural is dead -> relu(0) = 0, relu'(0) = 0
        zero = t.zeros(size)
        if t.equal(input,zero):
            print("Warning: your output is zero and might not be learning with reLU -> Try to lower your learning step.")

        return input

    def round(self,input):
        return input.apply_(lambda x: (int(round(x))))

    def MSE(self,y_train,output):
        return (y_train-output)**2

    def MSE_derivative(self,y_train,output):
        y_train = t.transpose(y_train,0,1)
        N = y_train.size()[1]
        return (-2/N) * (y_train - output)

    def MSE_scalar(self,y_train, output):
        y_train = t.flatten(y_train).tolist()
        output = t.flatten(output).tolist()
        N = len(y_train)
        res = 0
        for x, y in zip(y_train, output):
            res += (x - y) ** 2
        return res / N

    def SGD(self,operations,gradiants,step,bias): #update weights
        j = 0  # j is an index corresponding to the ith gradiants (1/2 operation is linear so we have 1/2 * len(operations) gradient
        for i in range(len(operations)):
            if operations[i].type == 1:
                operations[i].weights -= step * gradiants[j]["w"]
                if bias:
                    operations[i].bias -= step * gradiants[j]["b"]
                j+=1

    def backward(self,output,operation,y_train): # cf nn2.pdf
        gradiants = []
        deltas = []
        derivative = self.MSE_derivative(y_train,output)
        i = 0
        operations = c.deepcopy(operation)
        operations = operations[::-1] # Inverse list order

        for op in operations:
            if op.type == 2:
                if i == 0:
                    derivative = t.mul(derivative, op.derivative(self.memory.pop()))  # compute of delta if i == 0
                    deltas.append(derivative)
                else:
                    w=self.fls.pop().weights #taking the weight of next layer to compute derivative c.f report
                    derivative = t.mul(t.matmul(t.transpose(w,0,1),deltas.pop()),op.derivative(self.memory.pop()))
                    deltas.append(derivative)

            if op.type == 1:
                gradiant = {}
                gradiant['b'] = derivative # doesnt change because if we derive wrt b we obtain 1 -> last computed derivative = delta_i
                gradiant['w'] = t.matmul(derivative,t.transpose(self.memory.pop(),0,1)) #delta2 x_i-1 ' , in example: for W2 -> x1 -> if we derive wrt W we obtain x
                gradiants.append(gradiant)
            i += 1
        return gradiants[::-1] #reverse to have the gradiants in the good order

    def train(self,x_train,y_train,batch_size,epochs,training_step=0.01,grad_accumulate=0,bias=True):
        count = grad_accumulate
        N = x_train.size()[0]
        indexs = [i for i in range(0,N,batch_size)]
        batchs = [x_train[i:i+batch_size] for i in indexs]
        y_trains = [y_train[i:i+batch_size] for i in indexs]
        countfirst = True
        for i in range(epochs):
            if i%5000 == 0:
                print("Start of epochs ",i)
            for batch,y_train in zip(batchs,y_trains):
                output = self.forward(batch)
                gradiants = self.backward(output,self.operations,y_train)
                if grad_accumulate == 0:
                    self.SGD(self.operations,gradiants,training_step,bias)
                    grad_accumulate = count
                else:
                    if countfirst: # The first iteration we initialize the accumulate gradient
                        acc_grads = gradiants
                        countfirst = False
                    else:
                        for acc_grad,grad in zip(acc_grads,gradiants):
                            acc_grad["w"] += grad["w"]
                            acc_grad["b"] += grad["b"]
                    grad_accumulate -= 1

    def computeAcc(self,x_test,y_test,batch_size):
        warnings.filterwarnings("ignore")
        N = y_test.size()[0]
        y_preds = []
        indexs = [i for i in range(0, N, batch_size)]
        for i in indexs:
            y_pred = self.round(self.forward(x_test[i:i+batch_size]))
            y_preds.append(y_pred)

        #flatten the list (list of list of list)
        y_preds = [x for sub in y_preds for x in sub]
        y_preds = [x for sub in y_preds for x in sub]
        y_preds = t.FloatTensor(y_preds)

        prediction = (t.tensor(y_preds).flatten() == y_test.flatten()).tolist()
        true_positif = prediction.count(True)
        print("Accuracy = ", true_positif / N)
        return true_positif / N

    def saveModel(self,model,name): #Save model in folder "models" with specified name
        string = "models/"
        string += name
        file = open(string,"wb")
        pickle.dump(model,file)
        file.close()

    def loadModel(self, name):
        string = "models/"
        string += name
        file = open(string, "rb")
        load_model = pickle.load(file)
        file.close()
        return load_model

