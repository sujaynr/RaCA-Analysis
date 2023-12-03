# Standard libraries
import json
import gc
import pdb
import time
import argparse
import pickle

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
from scipy.stats import truncnorm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from tqdm.notebook import tqdm
import wandb

# Custom modules
from models import (
    EncoderConv1DWithUncertainty, 
    EncoderRNN, 
    EncoderTransformer, 
    EncoderConv1D, 
    LinearMixingEncoder, 
    LinearMixingDecoder, 
    LinearMixingModel, 
    LinearMixingSOCPredictor
)
from utils import (
    remove_outliers, 
    postProcessSpectrum, 
    gaus, 
    genSeedMs, 
    fakeTrough, 
    A, 
    torchA, 
    calculate_accuracy
)

# Other
import import_ipynb

# python3 updatedTrain.py [model (s,t,r,c1, c1u)] [epochs] [val_ratio] [batch_size] [test? -t for true, blank for false]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for different models")
    parser.add_argument("model", choices=["s", "c1", "c1u", "r", "t"], type=str, help="Choose a model: s = Standard Linear, c1 = 1D CNN, r = RNN, t = Transformer")
    parser.add_argument("epochs", type=int, help="Number of training epochs")
    parser.add_argument("trainval_ts", type=float, help="Train-validation split ratio (as a percentage)")
    parser.add_argument("batch", type=int, help="Batch Size")
    parser.add_argument("-t", "--test", action="store_true", default=False, help="Use this flag to indicate setting aside the test data")

    args = parser.parse_args()

    model_choices = {
        "s": "Standard Linear",
        "c1": "1D CNN",
        "c1u": "1D CNN with Uncertainties",
        "r": "RNN",
        "t": "Transformer"
    }

    print(f"Model: {model_choices[args.model]}")
    print(f"Epochs: {args.epochs}")
    print(f"Validation Ratio: {args.trainval_ts}")
    print(f"Batch Size: {args.batch}")
    if args.test:
        print("Setting Aside Test: True")
    else:
        print("Setting Aside Test: False")

wandb.init(
    # set the wandb project where this run will be logged
    project="SOC_ML_Stage2",
    name="oneoptimNov29"  # Change the run name
)




# (XF, dataI, sample_soc, sample_socsd) = torch.load('../RaCA-data-first100.pt')
data = np.loadtxt("/home/sujaynair/RaCA-spectra-raw.txt", delimiter=",",dtype=str)
sample_bd = data[1:,2158].astype('float32') # bulk density
sample_bdsd = data[1:,2159].astype('float32') # not used
sample_soc = data[1:,2162].astype('float32')
sample_socsd = data[1:,2163].astype('float32') # not used
print("Loaded txt")

XF = np.array([x for x in range(350,2501)]);

# linearly interpolated, fixed bad pixels
with open('/home/sujaynair/RaCA-data.pkl', 'rb') as file:
    dataI = pickle.load(file)
del data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('/home/sujaynair/RaCA-SOC-a-spectrum-analysis/RaCA-SOC-a/Step5/Step5FULLMODEL.pkl', 'rb') as file:
        (model,_,_,_,_,_,dataIndices,msoc,_,_,_) = pickle.load(file)

# pdb.set_trace()
ttrrFull = torch.cat((model.rhorad,model.rrsoc))
ttmFull  = torch.cat((model.ms,torch.tensor(msoc.tolist()).unsqueeze(1)),dim=1)
ttmFull  = (ttmFull.t() / torch.sum(ttmFull,axis=1)).t()
ttfFull  = torch.cat((model.fs,model.fsoc.unsqueeze(0)),dim=0)
ttIhat   = torch.matmul(torchA(ttmFull,ttrrFull).float(),ttfFull.float())
# pdb.set_trace()
rrFullRaCAFit = ttrrFull.detach().numpy()
msFullRaCAFit = ttmFull.detach().numpy() # used
FsFullRaCAFit = ttfFull.detach().numpy()
IhFullRaCAFit = ttIhat.detach().numpy() # used
del model
del ttrrFull, ttmFull, ttfFull, ttIhat

KEndmembers = 90
NPoints = IhFullRaCAFit.shape[0]
MSpectra = 2151

# Truth-level outputs: regressed abundances from an LMM, these are used for decoder model
tFs      = torch.tensor(FsFullRaCAFit.tolist()).to(device)
tMs      = torch.tensor(msFullRaCAFit.tolist()).to(device)
trhorads = torch.tensor(rrFullRaCAFit.tolist()).to(device)

# Truth-level inputs: Individual spectra
# tIs = torch.tensor(IhFullRaCAFit.tolist()).to(device)
tmsoc    = torch.tensor(sample_soc[dataIndices.astype('int')].tolist()).to(device)
tmsoc    = tmsoc / 100.
tIs      = torch.tensor(dataI[dataIndices.astype('int')].tolist()).to(device)
####^ EXACTLY WHAT ARE THESE VARIABLES DOING, ADD COMMENTS

# Split the data into a test set (5%) and the remaining data
if args.test:

    X_temp, X_test, y_temp, y_test = train_test_split(IhFullRaCAFit, msFullRaCAFit, test_size=0.05, random_state=42)

    # Split the remaining data into training (80%) and validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=args.trainval_ts, random_state=42)
else:
    X_train, X_val, y_train, y_val = train_test_split(IhFullRaCAFit, msFullRaCAFit, test_size=args.trainval_ts, random_state=42)
# pdb.set_trace()
# Convert the numpy arrays to PyTorch tensors and move them to the appropriate device
tMs_train = torch.tensor(y_train.tolist())
tIs_train = torch.tensor(X_train.tolist())

tMs_val = torch.tensor(y_val.tolist())
tIs_val = torch.tensor(X_val.tolist())

# Create data loaders for training and validation
train_dataset = torch.utils.data.TensorDataset(tIs_train, tMs_train)
val_dataset = torch.utils.data.TensorDataset(tIs_val, tMs_val)
# pdb.set_trace()
batch_size = args.batch  # You can adjust this as needed
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 8)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = 8)

# Training settings, optimizer declarations
nepochs = args.epochs

# Set up encoder model and optimizer
try:
    if args.model == "s":
        print("Using Standard Linear Model")
        encoder_model = LinearMixingEncoder(MSpectra, KEndmembers, 512).to(device)
    elif args.model == "c1":
        print("Using 1D Conv Model")
        encoder_model = EncoderConv1D(MSpectra, KEndmembers, 32, 15).to(device) # (M, K, hidden_size, kernel_size)
    elif args.model == "c1u":
        print("Using 1D Conv Model with Uncertainties")
        encoder_model = EncoderConv1DWithUncertainty(MSpectra, KEndmembers, 32, 15).to(device) # (M, K, hidden_size, kernel_size)
    elif args.model == "r":
        print("Using RNN Model")
        encoder_model = EncoderRNN(MSpectra, KEndmembers, 64).to(device) # (M, K, hidden_size)
    elif args.model == "t":
        print("Using Transformer Model")
        # pdb.set_trace()
        encoder_model = EncoderTransformer(MSpectra, KEndmembers, 64, 4, 2) # (M, K, hidden_size, num_heads, num_layers)
    else:
        raise ValueError("Model error, please choose s (standard linear), c1 (1D conv), r (RNN), or t (Transformer)")
except ValueError as e:
    print(e)
encoder_model = encoder_model.to(device)
# pdb.set_trace()

# encoder_optimizer = optim.Adam(encoder_model.parameters(), lr=0.000001, betas=(0.99, 0.999))

# Set up decoder model and optimizer
decoder_model = LinearMixingDecoder(tFs, tMs, trhorads).to(device)

# decoder_optimizer = optim.Adam(decoder_model.parameters(), lr=0.000001, betas=(0.99, 0.999))
combined_optimizer = optim.Adam(list(encoder_model.parameters()) + list(decoder_model.parameters()), lr=0.000001, betas=(0.99, 0.999))

# Rest of the code, but replace 'seedEncoderModel' with 'encoder_model' and 'decoder_model'

lossTrackingEncoder = np.zeros(nepochs);
lossTrackingDecoder = np.zeros(nepochs);
lossTrackingEncoderV = np.zeros(nepochs)  # Validation losses
lossTrackingDecoderV = np.zeros(nepochs)  # Validation losses
lossTrackingDecoderLagrangeFactor = np.zeros(nepochs);
rrTracking = np.zeros(nepochs);

encoderPreds=[]
decoderPreds=[]
means=[]


# train_tIs, val_tIs, train_tmsoc, val_tmsoc = train_test_split(tIs, tmsoc, test_size=args.trainval_ts, random_state=42) # not used
all_uncertainties = []

#encoderPreds is now samples, encoderUncertainties is now sds
#maybe add prior to keep predictions close to unit gaus
# Training loop
for epoch in tqdm(range(nepochs)):
    # Log rrsoc
    rrTracking[epoch] = decoder_model.rrsoc.detach().item()

    # Initialize loss variables for this epoch
    total_encoder_loss = 0.0
    total_decoder_loss = 0.0
    total_kl_Loss = 0.0
    all_uncertainties = []

    # Batching and training
    for batch_data in train_loader:
        # Extract batch data
        batch_tIs, batch_tmsoc = batch_data
        batch_tIs = batch_tIs.to(device)
        batch_tmsoc = batch_tmsoc.to(device)

        # Get abundance predictions from the encoder for the batch
        encoderPreds, encoderUncertainties, mean = encoder_model(batch_tIs) #edit to only have 2nd arg if model is c1u

        all_uncertainties.extend(encoderUncertainties.detach().cpu().numpy())

        # Get spectrum predictions from the decoder for the batch
        # decoderPreds = decoder_model(encoderPreds)
        decoderPreds = decoder_model(mean)
        # pdb.set_trace()
        # Compute encoder loss: sqerr from true Msoc values for the batch
        # encoder_loss = 1000 * torch.mean((encoderPreds[:, -1] - batch_tmsoc[:, -1]) ** 2)
        encoder_loss = 1000 * torch.mean((mean[:, -1] - batch_tmsoc[:, -1]) ** 2)

        # Add decoder loss: sqerr from true RaCA spectra for the batch
        decoder_loss = torch.mean((decoderPreds - batch_tIs) ** 2)

        # Multiply decoder loss by the Lagrange factor
        decoder_loss = decoder_loss * decoder_model.computeLagrangeLossFactor()
        # pdb.set_trace()
        prior_mean = 1/90
        prior_sd = 0.1
        kl_Loss = torch.log(prior_sd / (torch.sqrt(encoderUncertainties) + 1e-6)) + (encoderUncertainties + (mean - prior_mean)**2) / (2 * prior_sd**2) - 0.5
        kl_Loss = -0.5 * torch.sum(kl_Loss)
        # Calculate the combined loss
        # pdb.set_trace()
        # loss = encoder_loss + decoder_loss + kl_Loss * 0.1 # adjust weight
        loss = encoder_loss + decoder_loss # adjust weight

        # Backpropagate the gradients for both models
        combined_optimizer.zero_grad()
        loss.backward()
        combined_optimizer.step()

        # Accumulate batch losses
        total_encoder_loss += encoder_loss.item()
        total_decoder_loss += decoder_loss.item()
        # total_kl_Loss += kl_Loss.item()

    # Calculate the average loss for this epoch
    avg_encoder_loss = total_encoder_loss / len(train_loader)
    avg_decoder_loss = total_decoder_loss / len(train_loader)

    # Log and print the average losses
    wandb.log({"Encoder_Training Loss": avg_encoder_loss, "Decoder_Training Loss": avg_decoder_loss})
    print("Epoch {}: Encoder Loss: {:.4f}, Decoder Loss: {:.4f}".format(epoch, avg_encoder_loss, avg_decoder_loss))

    # Validation Loss
    with torch.no_grad():
        # Similar batching process for validation data
        total_encoder_lossV = 0.0
        total_decoder_lossV = 0.0

        for batch_dataV in val_loader:
            batch_val_tIs, batch_val_tmsoc = batch_dataV
            batch_val_tIs = batch_val_tIs.to(device)
            batch_val_tmsoc = batch_val_tmsoc.to(device)
            encoderPredsV, encoderUncertaintiesV, meanV = encoder_model(batch_val_tIs)
            all_uncertainties.extend(encoderUncertaintiesV.detach().cpu().numpy())
            # decoderPredsV = decoder_model(encoderPredsV)
            # encoder_lossV = 1000 * torch.mean((encoderPredsV[:, -1] - batch_val_tmsoc[:, -1]) ** 2)
            decoderPredsV = decoder_model(meanV)
            encoder_lossV = 1000 * torch.mean((meanV[:, -1] - batch_val_tmsoc[:, -1]) ** 2)
            decoder_lossV = torch.mean((decoderPredsV - batch_val_tIs) ** 2)
            total_encoder_lossV += encoder_lossV.item()
            total_decoder_lossV += decoder_lossV.item()

        avg_encoder_lossV = total_encoder_lossV / len(val_loader)
        avg_decoder_lossV = total_decoder_lossV / len(val_loader)

        wandb.log({"Encoder_Validation Loss": avg_encoder_lossV, "Decoder_Validation Loss": avg_decoder_lossV})
        print("Validation - Encoder Loss: {:.4f}, Decoder Loss: {:.4f}".format(avg_encoder_lossV, avg_decoder_lossV))
pdb.set_trace()
from scipy.stats import norm


# # Convert to numpy array
# all_uncertainties_array = np.array(all_uncertainties)

# # Scale uncertainties to be between 0 and 1
# uncertainties_scaled = (all_uncertainties_array - all_uncertainties_array.min()) / (all_uncertainties_array.max() - all_uncertainties_array.min())

# # Calculate the average for each of the 90 columns across all rows
# average_uncertainties = np.mean(uncertainties_scaled, axis=0)
# average_uncertainties_n = np.mean(all_uncertainties_array, axis = 0)
# # Convert to list
# average_uncertainties_list = average_uncertainties.tolist()
# average_uncertainties_nlist = average_uncertainties_n.tolist()


# # Plot histogram
# plt.hist(average_uncertainties_list, bins=30, color='blue', edgecolor='k', alpha=0.7)

# # Add labels and title
# plt.xlabel('Uncertainty')
# plt.ylabel('Frequency')
# plt.title('Histogram of Average Uncertainties')

# # Show the plot
# plt.show()




total_params = sum(p.numel() for p in encoder_model.parameters())
print("Total parameters:", total_params)
total_paramsD = sum(p.numel() for p in decoder_model.parameters())
print("Total parameters decoder:", total_paramsD)

wandb.finish()