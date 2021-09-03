import gzip as gz
import torch, geotorch, torch.nn as nn
import numpy as np
import random
from itertools import product
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
import csv
import matplotlib.pyplot as plt
import pickle
from hyper_parallel import HyperLinear
torch.set_printoptions(precision=8)

"""
N Number of particles
M Number of sites
& noisy 1RDM
"""
Steps = 200           # Number of steps
Range = 50          # Range

class Functional(nn.Module):
    def __init__(self, batch_size, size):
        super(Functional, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(batch_size, size, size),requires_grad=True)
    def forward(self, S,L):
        DD = torch.einsum('lij,ljk->lik', S, self.weight)
        return torch.einsum('lij,lij,lij->l', DD, DD, L)

class Model(nn.Module):
      def __init__(self, batch_size, size):
          super(Model, self).__init__()
          self.functional = Functional(batch_size, size)
          geotorch.orthogonal(self.functional, "weight")
      def forward(self, S,L):
          return self.functional(S,L)


list1 = [[] for i in range(Range)]
list2 = [[] for i in range(Range)]
size = 4
data = pickle.load(gz.open('data_4_400.pickle.gz','rb'))
torch.set_default_tensor_type(torch.cuda.DoubleTensor)
eta = torch.tensor(data['eta']).cuda().double() #[0:60]
model =  Model(len(eta), size).cuda().double()
optim = torch.optim.Adam(model.parameters(), lr=0.001)
RDM = torch.tensor(data['rdms']).cuda().double()
#eta = torch.tensor(data['eta']).cuda().double()[0:30]
Lambda =  (torch.tensor(data['Lambdas']).cuda().double()*1.0) #[0:60]
USigma = torch.tensor(data['USigma']).cuda().double() #[0:60]
print(RDM.shape, eta.shape, Lambda.shape, USigma.shape)
print(eta)
#last_loss = 1000000000
eta = (eta - torch.mean(eta))/torch.std(eta)
best_loss = torch.ones(eta.shape)*10000
for i in range(100000):
    loss =model(USigma,Lambda)
    best_loss =  torch.where(loss<best_loss, loss, best_loss)
    print(loss)
    loss_b = torch.sum(torch.log(loss))
    optim.zero_grad()
    loss_b.backward()
    optim.step()

#    if (i+2)%100==0:
#        if torch.abs(last_loss-loss)<0.0000001 and i>1000:
#            print('break', )
#            break
#        last_loss = loss
 


