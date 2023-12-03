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
import pickle
plt.rcParams['text.usetex'] = True


def remove_outliers(data, threshold=3):
    z_scores = np.abs((data - data.mean()) / data.std())
    return data[z_scores < threshold]
def postProcessSpectrum(xin,xout,refin): # Linear interpolation
    return np.interp(xout, xin, refin)
def gaus(mu, sigma, N=1) :
    return np.random.randn(N) * sigma + mu;
def genSeedMs(endMemList, endMemMap, NPoints, KEndmembers, msoc) :
    seedMsDict = {}
    for endMem in endMemList : #for every endmember, if avg abundance exists, get seed alpha
        if "Coarse measure of average abundance" in endMemMap[endMem] :
            seedMsDict[endMem] = float(endMemMap[endMem]["Coarse measure of average abundance"])/100.
        else :
            print("\t - Mineral",endMem,"Does not have any abundance set in the JSON file.")

    # collect constant seeds from the definitions above
    seedMs = [seedMsDict[x] if x in seedMsDict else 0.0 for x in endMemList] #puts seed alphas in other list and 0 is no value
    seedMs = seedMs + [0.0] # for background

    # make seedMs an N x K matrix
    seedMs = np.ones([NPoints,KEndmembers]) * seedMs #N x K matrix, where N is the number of points (spectra) and K is the number of endmembers
    seedMs[:,KEndmembers-1] = msoc #last col is msoc
    seedMrem = 1.0 - np.sum(seedMs,axis=1)

    # if seedM remainder is < 0, SOC exceeds pre-defined content,
    # so rescale pre-defined values, fix SOC to prior value, and set rest to 0
    seedMNegRem = (seedMs.T * (seedMrem < 0.0).astype('float32') / np.sum(seedMs[:,:-1],axis=1) * (1.0 - msoc)).T #make sure add to 1
    seedMNegRem[:,-1] = msoc * (seedMrem < 0.0).astype('float32')
    seedMs = (seedMs.T * (seedMrem >= 0.0).astype('float32')).T + seedMNegRem #why redistribute?
    
    # get seeds that haven't been filled in yet, excluding SOC
    # if remaining mass fraction is < 0, exclude from further partitioning
    seedMZeros = ((seedMs == 0.0).astype('float32').T * (seedMrem > 0.0).astype('float32')).T
    
    # sample random remaining seeds and renormalize to the appropriate remainder
    seedMZeros = seedMZeros * np.random.random([NPoints,KEndmembers])
    seedMZeros[:,-1] = 0.0 # SOC set to 0 for these
    
    seedMZeros = (seedMZeros.T / (np.sum(seedMZeros,axis=1)+0.00000000001)).T
    seedMrem = 1.0 - np.sum(seedMs,axis=1)
    seedMZeros = (seedMZeros.T * seedMrem).T
    seedMZeros[:,-1] = 0.0 # SOC set to 0 for these

    # add remaining seeds into seedMs
    seedMs = seedMs + seedMZeros 

    del seedMZeros, seedMrem
    
    return seedMs, seedMsDict
def fakeTrough(x,mu,sigma) :
    return 0.1*np.exp(-(x-mu)**2/2.0/sigma)
def A(ms,rhorads) :
    tA = ms / rhorads
    return (tA.T / np.sum(tA,axis=1)).T
def torchA(ms,rhorads) :
    tA = ms / rhorads
    return (tA.t() / torch.sum(tA,axis=1)).t()
def calculate_accuracy(predictions, ground_truth):
    # Define your accuracy calculation logic here
    # For example, if you are doing classification:
    correct = (predictions == ground_truth).sum().item()
    total = len(ground_truth)
    accuracy = correct / total
    return accuracy