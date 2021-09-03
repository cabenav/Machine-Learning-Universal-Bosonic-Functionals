import torch, geotorch, torch.nn as nn
import numpy as np
import random
from itertools import product
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
import csv
import matplotlib.pyplot as plt
import pickle

#def the_func1(S,L,VV):
 #  DD = np.einsum('ij,jk->ik', S, VV)
  # return np.einsum('ij,ij,ij->', DD, DD, L)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def expon(x):
   if x < 1:
      return 2.1
    #else:
     #  return 4

class Functional(nn.Module):
    def __init__(self, size):
        super(Functional, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(size, size),requires_grad=True)
    def forward(self, S,L):
        DD = torch.einsum('ij,jk->ik', S, self.weight)
        return torch.einsum('ij,ij,ij->', DD, DD, L)

class Model(nn.Module):
      def __init__(self, size):
          super(Model, self).__init__()
          self.functional = Functional(size)
          geotorch.orthogonal(self.functional, "weight")
      def forward(self, S,L):
          return self.functional(S,L)

"""
N Number of particles
M Number of sites
& noisy 1RDM
"""
Steps = 200         # Number of steps
Range = 40          # Range

list1 = [[] for i in range(Range)]
list2 = [[] for i in range(Range)]

for y in range(39,Range):
# Generate the 1RDM (symmetric and positive definite)
   N, M = (y+1), (y+1)       # Number of particles, number of sites
   BEC = N*(N-1)/M         # BEC energy 
   E0 = (N**2/M)-N         # 0 energy
   # Hilbert space of N -1 particles 
   BEC = N*(N-1)/M         # BEC energy 
   E0 = (N**2/M)-N         # 0 energy
   print(E0,BEC)
   lista = list(range(0,int(N/M)+1))
   #Lambda = [i for i in product(lista, repeat=M) if sum(i)==N-1]
   #Lambda = np.matrix.transpose(np.array(Lambda))
   Lambda  = np.ones((M,M))
   for i in range(M):
      Lambda[i][i] += -1 
   print(N,M,Lambda.shape)
   for x in range(100,Steps):
      eta = (x+1)*(N/M)/Steps
      RDM = [[eta**(expon(eta)) for x in range(M)] for y in range(M)] 
      for i in range(M-1):
         RDM[i][i] = N/M
         RDM[i][i+1] = eta
         RDM[i+1][i] = eta
      RDM[M-1][M-1] = N/M
      if is_pos_def(RDM) == True:                     # Is positive definite?
         list1[y].insert(x,1/Steps*(x+1))
         aa, U = np.linalg.eigh(RDM)
         Sigma = np.diagflat(np.sqrt(aa))
         #print("eigen",aa)
         if Lambda.shape[1]-M > 0:
            Sigma = np.concatenate((Sigma,np.zeros((Lambda.shape[1]-M,M ))))
         Sigma2 = np.matrix.transpose(Sigma)
         USigma = np.matmul(U, Sigma2)
         #print("shapes =", USigma.shape,Lambda.shape,Lambda.shape[1])
         model = Model(Lambda.shape[1]).double()
         optim = torch.optim.Adam(model.parameters(), lr=0.001)
         torch.set_default_tensor_type(torch.DoubleTensor)
         USigma = (torch.tensor(USigma)*1.0).double()
         Lambda = (torch.tensor(Lambda)*1.0).double()
         for i in range(8000):
            loss =model(USigma,Lambda)
            #print(loss)
            optim.zero_grad()
            loss.backward()
            optim.step()
         list2[y].insert(x,loss.item())
         print(eta,loss.item())
   list2[y] = np.array(list2[y])-E0
 
pickle.dump(list1, open( "list1.p", "wb" ) )
pickle.dump(list2, open( "list2.p", "wb" ) )


plt.plot(list1[1], list2[1], 'blue',list1[2], list2[2], 'red',list1[3], list2[3], 'green',list1[4], list2[4], 'green',list1[5], list2[5], 'green')

plt.show()
