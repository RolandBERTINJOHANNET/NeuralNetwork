import numpy as np
import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import pickle
import gzip
import pandas as pd

import os
os.chdir("NeuralNetwork")


def msecost(yhat,y):
    return (y-yhat)**2

def msecostprime(yhat,y):
    return (yhat-y)*2

def cecost(y,yhat):
    return -( y*np.log(yhat) + (1-y)*np.log(1-yhat) )

def cecostprime(yhat,y):
    return -( (y/yhat) - ((1-y)/(1-yhat)) )

def sigmoid(a):
    return 1./(1.+np.exp(-a))

def sigmoid_prime(a):
    return sigmoid(a)*(1.-sigmoid(a))

class Network:
    def __init__(self,in_dim,hidden_dim,out_dim):
        self.weights = []
        self.weights.append(np.reshape(0.25*np.random.randn((hidden_dim*in_dim)),(hidden_dim,in_dim)))   #matrix of hidden*in weights
        self.weights.append(np.reshape(0.25*np.random.randn((out_dim*hidden_dim)),(out_dim,hidden_dim)))  #matrix of hidden*out weights
        self.biases = []
        self.biases.append(np.reshape(0.25*np.random.randn((hidden_dim)),(hidden_dim,)))   #vector of hidden_dim biases
        self.biases.append(np.reshape(0.25*np.random.randn((out_dim)),(out_dim,)))   #vector of hidden_dim biases
        self.gradw = None
        self.gradB = None

    def feedforward(self,input):
        #this isn't production code so I'm storing linear activations for every step
        self.result = input
        if len(input) !=self.weights[0].shape[1]:
            print("problem in network : feeding an input of dim ",
                    input.shape,"but dim",self.weights[0].shape[1],"is needed")
        else:
            self.linact = []
            self.linact.append(input)          #store input as if an activation, helps for backprop
            for w,b,i in zip(self.weights,self.biases,range(len(self.weights))):
                self.result = np.matmul(w,self.result)
                self.result = self.result + b
                self.linact.append(self.result)
                self.result = np.vectorize(sigmoid)(self.result)
                
    def predict(self,input):
        self.feedforward(input)
        return np.argmax(self.result)

    def predict_batch(self,input):
        return [self.predict(x) for x in input]
        
    def backprop(self,input,label):
        self.errors = [np.zeros(self.biases[0].shape),np.zeros(self.biases[1].shape)]
        self.gradw = [np.zeros(self.weights[0].shape),np.zeros(self.weights[1].shape)]
        self.gradB = [np.zeros(self.biases[0].shape),np.zeros(self.biases[1].shape)]
        self.feedforward(input) #retrieve activations
        yhat = self.result
        #compute out layer error
        #print("errors before : ",[elt.shape for elt in self.errors])
        self.errors[-1] = np.vectorize(sigmoid_prime)(self.linact[-1])   #derivative of activation function
        self.errors[-1] = self.errors[-1] * np.vectorize(cecostprime)(yhat,label) #by the derivative of cost fun
        #print("errors after : ",[elt.shape for elt in self.errors])
        #compute error for every active neuron
        for i in np.arange(len(self.errors)-2,-1,-1):
            self.errors[i] = self.errors[i] + np.matmul(self.weights[i+1].T,self.errors[i+1])
            self.errors[i] = self.errors[i] * np.vectorize(sigmoid_prime)(self.linact[i+1])
        #compute the gradient
        self.gradB = self.errors
        for i in range(len(self.gradw)):
            self.gradw[i] = np.outer(self.errors[i],np.vectorize(sigmoid)(self.linact[i]))

    def test(self,X,Y):
        #test cost over a batch and print it out
        pred = self.predict_batch(X)
        y_transformed =[np.argmax(elt)for elt in Y]
        correct = sum([int(p==y) for (p,y) in zip(pred,y_transformed)]) 
        return float(correct)/float(len(Y))

    def update_minibatch(self,data,lr):
        #initialize gradient with first backprop
        x,y = list(data)[0]
        self.backprop(x,y)
        deltaw,deltab = (self.gradw,self.gradB)
        #sum the gradient over the rest of the samples
        for x,y in list(data)[1:]:
            self.backprop(x,y)
            deltaw = [delta+grad for (delta,grad) in zip(deltaw,self.gradw)]
            deltab = [delta+grad for (delta,grad) in zip(deltab,self.gradB)]
        #average it
        deltab = [elt/len(list(data)) for elt in deltab]
        deltaw = [elt/len(list(data)) for elt in deltaw]
        #update the weights
        self.weights = [elt - lr * delta for (delta,elt) in zip(deltaw,self.weights)]
        self.biases = [elt - lr * delta for (delta,elt) in zip(deltab,self.biases)]


    def learn(self,dataX,dataY,lr,epochs,batch_size,testX,testY):
        k=0
        data = list(zip(dataX,dataY))
        minibatches = [data[idx:idx+batch_size] for idx in range(0,len(list(data)),batch_size)]
        for i in tqdm.trange(epochs):
            for minibatch,i in zip(minibatches,range(len(minibatches))):
                self.update_minibatch(minibatch,lr)
                print("minibatch",i )
                if i%10==0:
                    print(self.test(testX,testY))

nn = Network(784,100,10)


with gzip.open("mnist.pkl.gz","rb") as file:
    (X_train,y_train),(X_test,y_test),_ = pickle.load(file,encoding="latin-1")


#one-hot-encode the labels
y_train = [[0]*(elt)+[1]+[0]*(9-elt) for elt in y_train]
y_test = [[0]*(elt)+[1]+[0]*(9-elt) for elt in y_test]

#X_cos = np.reshape(np.arange(-10,10,20./20.),(20,1,1))
#fun = np.vectorize(lambda x:(np.cos(x)+2)/4.)
#Y_cos = fun(X_cos)

nn.learn(X_train,y_train,0.15,100,512,X_test,y_test)