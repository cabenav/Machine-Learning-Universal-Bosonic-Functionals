# Uses code from https://github.com/vsitzmann/scene-representation-networks
import torch, geotorch, torch.nn as nn
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.utils
from torch import matrix_exp as expm
import numpy as np

import math
import numbers

import functools


def partialclass(cls, *args, **kwds):

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls


class FCLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(), #Tanh
#            nn.ELU()
        )

    def forward(self, input):
        return self.net(input)


class FCBlock(nn.Module):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=False):
        super().__init__()

        self.net = []
        self.net.append(
            FCLayer(
                in_features=in_features,
                out_features=hidden_ch))

        for i in range(num_hidden_layers):
            self.net.append(
                FCLayer(
                    in_features=hidden_ch,
                    out_features=hidden_ch))

        if outermost_linear:
            self.net.append(
                nn.Linear(
                    in_features=hidden_ch,
                    out_features=out_features))
        else:
            self.net.append(
                FCLayer(
                    in_features=hidden_ch,
                    out_features=out_features))

        self.net = nn.Sequential(*self.net)
        self.net.apply(self.init_weights)

    def __getitem__(self, item):
        return self.net[item]

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight,
                a=0.0,
                nonlinearity='leaky_relu',
                mode='fan_in')

    def forward(self, input):
        return self.net(input)


class HyperLayer(nn.Module):
    '''A hypernetwork that predicts a single Dense Layer, including LayerNorm and a ReLU.'''

    def __init__(self,
                 in_ch,
                 out_ch,
                 hyper_in_ch,
                 hyper_num_hidden_layers,
                 hyper_hidden_ch):
        super().__init__()

        self.hyper_linear = HyperLinear(
            in_ch=in_ch,
            out_ch=out_ch,
            hyper_in_ch=hyper_in_ch,
            hyper_num_hidden_layers=hyper_num_hidden_layers,
            hyper_hidden_ch=hyper_hidden_ch)
        self.norm_nl = nn.Sequential(
            nn.LayerNorm([out_ch], elementwise_affine=False),
            #            nn.ReLU(inplace=True)
            nn.Tanh()
        )

    def forward(self, hyper_input):
        '''
        :param hyper_input: input to hypernetwork.
        :return: nn.Module; predicted fully connected network.
        '''
        return nn.Sequential(self.hyper_linear(hyper_input), self.norm_nl)


class HyperFC(nn.Module):
    '''Builds a hypernetwork that predicts a fully connected neural network.
    '''

    def __init__(self,
                 hyper_in_ch,
                 hyper_num_hidden_layers,
                 hyper_hidden_ch,
                 hidden_ch,
                 num_hidden_layers,
                 in_ch,
                 out_ch,
                 outermost_linear=False):
        super().__init__()

#        PreconfHyperLinear = partialclass(HyperLinear,
#                                          hyper_in_ch=hyper_in_ch,
#                                          hyper_num_hidden_layers=hyper_num_hidden_layers,
#                                          hyper_hidden_ch=hyper_hidden_ch)
#        PreconfHyperLayer = partialclass(HyperLayer,
#                                          hyper_in_ch=hyper_in_ch,
#                                          hyper_num_hidden_layers=hyper_num_hidden_layers,
#                                          hyper_hidden_ch=hyper_hidden_ch)

        self.layers = nn.ModuleList()
        self.layers.append(
            HyperLayer(
                in_ch=in_ch,
                out_ch=hidden_ch,
                hyper_in_ch=hyper_in_ch,
                hyper_num_hidden_layers=hyper_num_hidden_layers,
                hyper_hidden_ch=hyper_hidden_ch))

        for i in range(num_hidden_layers):
            self.layers.append(
                HyperLayer(
                    in_ch=hidden_ch,
                    out_ch=hidden_ch,
                    hyper_in_ch=hyper_in_ch,
                    hyper_num_hidden_layers=hyper_num_hidden_layers,
                    hyper_hidden_ch=hyper_hidden_ch))

        if outermost_linear:
            self.layers.append(
                HyperLinear(
                    in_ch=hidden_ch,
                    out_ch=out_ch,
                    hyper_in_ch=hyper_in_ch,
                    hyper_num_hidden_layers=hyper_num_hidden_layers,
                    hyper_hidden_ch=hyper_hidden_ch))
        else:
            self.layers.append(
                HyperLayer(
                    in_ch=hidden_ch,
                    out_ch=out_ch,
                    hyper_in_ch=hyper_in_ch,
                    hyper_num_hidden_layers=hyper_num_hidden_layers,
                    hyper_hidden_ch=hyper_hidden_ch))

    def forward(self, hyper_input):
        '''
        :param hyper_input: Input to hypernetwork.
        :return: nn.Module; Predicted fully connected neural network.
        '''
        net = []
        for i in range(len(self.layers)):
            net.append(self.layers[i](hyper_input))

        return nn.Sequential(*net)


def last_hyper_layer_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(
            m.weight,
            a=0.0,
            nonlinearity='leaky_relu',
            mode='fan_in')
        m.weight.data *= 1e-1


class HyperLinear(nn.Module):
    '''A hypernetwork that predicts a single linear layer (weights & biases).'''

    def __init__(self,
                 in_ch,
                 out_ch,
                 hyper_in_ch,
                 hyper_num_hidden_layers,
                 hyper_hidden_ch):

        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.hypo_params = FCBlock(in_features=hyper_in_ch,
                                   hidden_ch=hyper_hidden_ch,
                                   num_hidden_layers=hyper_num_hidden_layers,
                                   out_features=(in_ch * out_ch) + out_ch,
                                   outermost_linear=True)
        self.hypo_params[-1].apply(last_hyper_layer_init)

    def forward(self, hyper_input):
        hypo_params = self.hypo_params(hyper_input.cuda())
        weights = hypo_params[..., :self.in_ch * self.out_ch]
        weights = weights.view(*(weights.size()[:-1]), self.out_ch, self.in_ch)
        weights = weights -  weights.transpose(2,1)
        weights = expm(weights)
        return Model(weights=weights)
        
        

class BatchLinear(nn.Module):
    def __init__(self,
                 weights):
        '''Implements a batch linear layer.
        :param weights: Shape: (batch, out_ch, in_ch)
        :param biases: Shape: (batch, 1, out_ch)
        '''
        super().__init__()

        self.weights = weights

    def __repr__(self):
        return "BatchLinear(in_ch=%d, out_ch=%d)" % (
            self.weights.shape[-1], self.weights.shape[-2])

    def forward(self, S,L):
#        print(self.weights.shape, S.shape, L.shape)
        DD = torch.einsum('lij,ljk->lik', S, self.weights)
        return torch.einsum('lij,lij,lij->l', DD, DD, L)

class Model(nn.Module):
      def __init__(self, weights):
          super(Model, self).__init__()
          self.functional = BatchLinear(weights)
      def forward(self, S,L):
          return self.functional(S,L)



