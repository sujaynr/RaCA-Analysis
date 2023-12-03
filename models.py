import import_ipynb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
import torch
import torch.nn as nn
import torch.optim as optim
import json
import gc
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import pdb
import pickle


from utils import *

import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, M, K, hidden_size):
        super(EncoderRNN, self).__init__()
        
        # RNN layer
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, K)
        
        # Softmax for K predictions
        self.smax = nn.Softmax(dim=1)
    
    def forward(self, y):
        # Initialize the hidden state of the RNN
        device = "cuda:0"
        h0 = torch.zeros(1, y.size(0), self.rnn.hidden_size).to(device)
        
        # Apply RNN layer
        y_rnn, _ = self.rnn(y.unsqueeze(2), h0)
        
        # Get the last time step output of the RNN
        y_last = y_rnn[:, -1, :]
        
        # Fully connected layers
        y_fc1 = self.fc1(y_last)
        y_fc2 = self.fc2(y_fc1)
        
        # Apply softmax to ensure predictions sum to 1
        ms = self.smax(y_fc2)
        
        return ms

class EncoderTransformer(nn.Module):
    def __init__(self, M, K, hidden_size, num_heads, num_layers):
        super(EncoderTransformer, self).__init__()

        # Parameters
        self.M = M
        self.K = K
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Positional encoding
        self.positional_encoding = nn.Embedding(M, hidden_size)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected layer for final prediction
        self.fc = nn.Linear(hidden_size, K)

    def forward(self, y):
        # Calculate positional encodings
        # pdb.set_trace()
        y = y.long()
        positional_encodings = self.positional_encoding(y)  # Shape: (batch_size, M, hidden_size)
        y = y.unsqueeze(-1).expand(-1, -1, self.hidden_size)

        # Add positional encodings to input
        y = y + positional_encodings

        # Reshape y for the transformer
        y = y.permute(1, 0, 2)  # Shape: (M, batch_size, hidden_size)

        # Transformer layers
        y_transformed = self.transformer(y) #memory error here
        # pdb.set_trace()
        # Fully connected layer
        y_fc = self.fc(y_transformed[-1])  # Use the last layer's output

        # Apply softmax to ensure predictions sum to 1
        ms = nn.functional.softmax(y_fc, dim=1)  # Shape: (batch_size, K)

        return ms

class EncoderConv1D(nn.Module):
    def __init__(self, M, K, hidden_size, kernel_size):
        super(EncoderConv1D, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * (M - 2 * kernel_size + 2), 128)
        self.fc2 = nn.Linear(128, K)
        
        # Softmax for 90 predictions
        self.smax = nn.Softmax(dim=1)
    
    def forward(self, y):
        # Apply Conv1D layers
        y_conv1 = self.conv1(y.unsqueeze(1))  # Add a channel dimension
        y_conv2 = self.conv2(y_conv1)
        
        # Flatten the output
        y_flat = y_conv2.view(y_conv2.size(0), -1)
        
        # Fully connected layers
        y_fc1 = self.fc1(y_flat)
        y_fc2 = self.fc2(y_fc1)
        
        # Apply softmax to ensure predictions sum to 1
        ms = self.smax(y_fc2)
        
        return ms


class EncoderConv1DWithUncertainty(nn.Module):
    def __init__(self, M, K, hidden_size, kernel_size):
        super(EncoderConv1DWithUncertainty, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * (M - 2 * kernel_size + 2), 128)
        
        # Output layers for predictions and uncertainties
        self.fc2_predictions = nn.Linear(128, K)
        self.fc2_uncertainties = nn.Linear(128, K)
        
    def forward(self, y):
        device = "cuda:0"
        # Apply Conv1D layers
        y_conv1 = self.conv1(y.unsqueeze(1))  # Add a channel dimension
        y_conv2 = self.conv2(y_conv1)
        
        # Flatten the output
        y_flat = y_conv2.view(y_conv2.size(0), -1)
        
        # Fully connected layers
        y_fc1 = self.fc1(y_flat)
        
        # Separate fully connected layers for predictions and uncertainties
        mean = self.fc2_predictions(y_fc1)
        sd = torch.sigmoid(self.fc2_uncertainties(y_fc1))
        # pdb.set_trace()
        # Apply a Gaussian function to add uncertainties
        sample = (torch.randn(sd.size(), device=device) * sd + mean).clip(min=0, max=1)
        
        return sample, sd, mean

class LinearMixingEncoder(nn.Module):
    def __init__(self, M, K, hidden_size):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(M),
            nn.Linear(M, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            
            # Collection of hidden layers
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            
            # Convert to vector of mass abundances.
            nn.Linear(hidden_size, K), 
            
            # No subsequent BatchNorm on last layer.
            
            # Use leaky ReLU: has gradient !=0 at large negative values
            # so that very small abundances (large neg. values at this layer)
            # sit in a region of nonvanishing gradient
            # nn.ELU()
            # nn.SiLU()
            nn.LeakyReLU()    
            # nn.ReLU()        
        )
        
        # Softmax to ensure abundances add up to 1
        self.smax = nn.Softmax() 
        
    def forward(self, y):
        # pdb.set_trace()
        y_mlp = self.mlp(y);
        ms = self.smax(y_mlp);
        return ms

class LinearMixingDecoder(nn.Module):
    def __init__(self, seedFs, seedMs, rhorad):
        super().__init__()
        
        # fixed quantities
        self.rhorad = rhorad[:-1]
        
        # model parameters
        self.Fs     = nn.Parameter(seedFs)
        self.rrsoc  = nn.Parameter(rhorad[-1])
        
    def forward(self, Ms):
        rrFull = torch.cat((self.rhorad,self.rrsoc.unsqueeze(0)))
        Ihat   = torch.matmul(torchA(Ms,rrFull).float(),self.Fs.float())
        
        return Ihat
                
    def computeLagrangeLossFactor(self) :
        # Add in a fake Lagrange multiplier to discourage Fs < 0 and Fs > 1
        oobsF = 1.0 * torch.sum((self.Fs < 0.0).float() * (self.Fs ** 2)) 
        oobsF = oobsF + 1.0 * torch.sum((self.Fs > 1.0).float() * (1.0 - self.Fs) **2)
        
        # Add in 1st derivative loss to smooth the curves
        diffloss = torch.sum(torch.diff(self.Fs) ** 2)
        diffloss += torch.sum(torch.diff(torch.diff(self.Fs)) ** 2)
        
        # Compute the multiplicative factor for our fake Lagrange multipliers
        # return (1 + 100.0* diffloss + 1000.0*oobsF) 
        return (1 + 1000.0*oobsF)

        
class LinearMixingModel(nn.Module):
    def __init__(self, seedFs, seedFsoc, seedMs, rhorad, seedrrsoc, nepochs):
        super().__init__()
        # fixed quantities
        self.rhorad = rhorad
        self.fs     = seedFs
        
        # model parameters
        self.fsoc   = nn.Parameter(seedFsoc)
        self.rrsoc  = nn.Parameter(seedrrsoc)
        self.ms     = nn.Parameter(seedMs)
        
        # model output
        self.Ihat   = 0;
        
        # variables for tracking optimization
        self.epoch = 0;
        self.nepochs = nepochs;
        
        self.lsq = np.zeros(nepochs);
        self.loss = np.zeros(nepochs);
        self.bdsALoss = np.zeros(nepochs);
        self.bdsFLoss = np.zeros(nepochs);
        self.omrsLoss = np.zeros(nepochs);
        self.diffloss1 = np.zeros(nepochs);
        self.difflossfull = np.zeros(nepochs);
        
        
    def forward(self, y):
        msocs,Is,Imax = y
        rrFull    = torch.cat((self.rhorad,self.rrsoc))
        mFull     = torch.cat((self.ms,msocs.unsqueeze(1)),dim=1)
        mFull     = (mFull.t() / torch.sum(mFull,axis=1)).t()
        fFull     = torch.cat((self.fs,self.fsoc.unsqueeze(0)),dim=0)
        self.Ihat = torch.matmul(torchA(mFull,rrFull).float(),fFull.float())
                
        # Add in a fake Lagrange multiplier to discourage abundances < 0.001 or > 0.999
        oobsA = torch.sum((mFull < 0.001).float() * (mFull - 0.001)**2) 
        oobsA = oobsA + torch.sum((mFull > 0.999).float() * (mFull + 0.001 - 1.0) **2)

        # Add in a fake Lagrange multiplier to discourage Fsoc < 0 and Fsoc > 1
        oobsF = 1.0 * torch.sum((self.fsoc < 0.0).float() * (self.fsoc ** 2)) 
        oobsF = oobsF + 1.0 * torch.sum((self.fsoc > 1.0).float() * (1.0 - self.fsoc) **2)
        
        # Add in 1st derivative loss to smooth the curves
        diffloss = torch.sum(torch.diff(self.fsoc) ** 2)
        self.diffloss1[self.epoch] = diffloss.detach().item();
        
        diffloss += torch.sum(torch.diff(torch.diff(self.fsoc)) ** 2)
        
        # Compute the loss function, which is the mean-squared error between data and prediction,
        # with a multiplicative factor for our fake Lagrange multipliers
        lsq = torch.sum((Is - self.Ihat) ** 2)
        # loss = lsq * (1 + 100.0* diffloss + 100.0*oobsA + 1000.0*oobsF) # + 10000.0*omrs
        loss = lsq * (1 + 100.0*oobsA + 1000.0*oobsF) # + 10000.0*omrs
        
        # Report optimization statistics
        self.lsq[self.epoch]  = lsq.detach().item()
        self.loss[self.epoch] = loss.detach().item();
        self.bdsALoss[self.epoch] = oobsA.detach().item();
        self.bdsFLoss[self.epoch] = oobsF.detach().item();
        self.difflossfull[self.epoch] = diffloss.detach().item();
        
        self.epoch += 1;
        
        return loss
class LinearMixingSOCPredictor(nn.Module):
    def __init__(self, seedFs, seedMs, trueMsoc, rhorad, seedrrsoc, nepochs):
        super().__init__()
        # fixed quantities
        self.rhorad = rhorad;
        self.fs     = seedFs;
        self.truemsoc = trueMsoc;
        
        # model parameters
        self.rrsoc  = nn.Parameter(seedrrsoc);
        self.ms     = nn.Parameter(seedMs);
        
        # model output
        self.Ihat   = 0;
        
        # variables for tracking optimization
        self.epoch = 0;
        self.nepochs = nepochs;
        
        self.lsq = np.zeros(nepochs);
        self.loss = np.zeros(nepochs);
        self.socbias = np.zeros(nepochs);
        self.bdsALoss = np.zeros(nepochs);
        self.diffloss1 = np.zeros(nepochs);
        self.difflossfull = np.zeros(nepochs);
        
        
    def forward(self, y):
        rrFull    = torch.cat((self.rhorad,self.rrsoc.unsqueeze(0)))
        mFull     = (self.ms.t() / torch.sum(self.ms)).t()
        self.Ihat = torch.matmul(torchA(mFull,rrFull).float(),self.fs.float())
                
        # Add in a fake Lagrange multiplier to discourage abundances < 0.001 or > 0.999
        oobsA = torch.sum((mFull < 0.001).float() * (mFull - 0.001)**2) 
        oobsA = oobsA + torch.sum((mFull > 0.999).float() * (mFull + 0.001 - 1.0) **2)
        
        # Add in 1st derivative loss to smooth the curves
        diffloss = torch.sum(torch.diff(self.Ihat) ** 2)
        self.diffloss1[self.epoch] = diffloss.detach().item();
        
        diffloss += torch.sum(torch.diff(torch.diff(self.Ihat)) ** 2)
        
        # Compute the loss function, which is the mean-squared error between data and prediction,
        # with a multiplicative factor for our fake Lagrange multipliers
        lsq = torch.sum((y - self.Ihat) ** 2)
        # loss = lsq * (1 + 100.0* diffloss + 100.0*oobsA)
        loss = lsq * (1 + 100.0*oobsA)
        
        # Report optimization statistics
        self.lsq[self.epoch]  = lsq.detach().item()
        self.loss[self.epoch] = loss.detach().item();
        self.socbias[self.epoch]  = self.truemsoc - mFull[-1];
        self.bdsALoss[self.epoch] = oobsA.detach().item();
        self.difflossfull[self.epoch] = diffloss.detach().item();
        
        self.epoch += 1;
        
        return loss
