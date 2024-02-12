# Standard libraries
import pdb
import time
import argparse
import random

# Third-party libraries
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import h5py
from tqdm import tqdm
import wandb

from sklearn.metrics import r2_score

# Custom modules
from models import (
    #EncoderConv1DWithUncertainty, 
    EncoderRNN, 
    EncoderTransformer, 
    EncoderConv1D, 
    LinearMixingEncoder, 
    ANNDecoder,
    LinearMixingDecoder
)


""" ############################################################################################
    ### HDF5Dataset

    Description:

        A streaming dataset handler using h5py.

        Example usage:
        dataset = HDF5Dataset('filename.h5')
        data_loader = data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
"""
class HDF5Dataset(data.Dataset):
    def __init__(self, filename, gpu=True, scanonly=False):
        super(HDF5Dataset, self).__init__()
        self.file_mag = h5py.File(filename, 'r')

        self.scanonly = '' if not scanonly else '_scanonly'

        if gpu :
            self.dataset = torch.tensor(self.file_mag['data'+self.scanonly][:]).to(device)
        else :
            self.dataset = torch.tensor(self.file_mag['data'+self.scanonly][:])

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        return self.dataset[index+self.scanonly,:]


""" ############################################################################################
    To run this script, use the following command:
        
        python3 updatedTrain.py [model (s,t,r,c1, c1u)] [epochs] [val_ratio] [batch_size]
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for different models")
    parser.add_argument("--encoderModel", choices=["s", "c1", "c1u", "r", "t"], type=str, help="Choose a model: s = Standard Linear, c1 = 1D CNN, r = RNN, t = Transformer")
    parser.add_argument("--crossValidationRegion", type=int, default=-1, help="RaCA region number for Leave-One-Region-Out cross-validation.")   
    parser.add_argument("--bootstrapIndex",        type=int, default=-1, help="Number of pedons to use in each bootstrap.")
    parser.add_argument("--epochs",     type=int, default=10,       help="Number of training epochs")
    parser.add_argument("--batch",      type=int, default=75,       help="Batch Size")
    parser.add_argument("--logName",    type=str, default="test",   help="Base name for output files.") 
    parser.add_argument("--trainValSplit", type=float, default=-1, help="Fraction of each region to use for validation. Negative number means use site-based cross-validation.")

    parser.add_argument("--noDecoder",      default=False, action='store_true', help="Flag to disable decoder model and only consider end-to-end encoder performance.") 
    parser.add_argument("--disableRhorads", default=False, action='store_true', help="Flag to disable conversion of mass abundance to area abundance via rhorads.") 
    parser.add_argument("--decoderModel",   default=False, action='store_true', help="Flag to implement an ANN decoder model in place of the linear mixing model.") 
    parser.add_argument("--fullFit",        default=False, action='store_true', help="Flag to fit the entire dataset, without validation.") 
    parser.add_argument("--regularizationTest", default=False, action='store_true', help="Flag to test regularization technique.")
    parser.add_argument("--fixRandomSeed", default=False, action='store_true', help="Flag to fix random seed for reproducibility.")
    parser.add_argument("--setRandomSeedTo", type=int, default=0, help="Set random seed to this value.")

    parser.add_argument("--spectraSOCLocation",         type=str, default="data_utils/ICLRDataset_RaCASpectraAndSOC_v3.h5", help="File name for soil spectra and SOC numbers.") 
    parser.add_argument("--splitIndicesLocation",       type=str, default="data_utils/ICLRDataset_splitIndices_v3.h5", help="File name for soil spectrum index, split by region number.") 
    parser.add_argument("--endmemberSpectraLocation",   type=str, default="data_utils/ICLRDataset_USGSEndmemberSpectra.h5", help="File name for pure endmember spectra and rhorads.") 

    parser.add_argument("--lr",  type=float, default=0.00005, help="Learning rate for Adam optimizer.")
    parser.add_argument("--b1",  type=float, default=0.99,    help="Beta1 for Adam optimizer.")
    parser.add_argument("--b2",  type=float, default=0.999,  help="Beta2 for Adam optimizer.")

    parser.add_argument("--finetuneEpochs",     type=int, default=10000,       help="Number of training epochs for the fine-tuning step.")

    
    args = parser.parse_args()

    if args.trainValSplit > 0 and (not args.fullFit and not args.fixRandomSeed) :
        raise ValueError("Error: fullFit and fixRandomSeed must be True if trainValSplit is > 0.")

    if args.fixRandomSeed :
        torch.manual_seed(args.setRandomSeedTo)
        random.seed(args.setRandomSeedTo)
        np.random.seed(args.setRandomSeedTo)

    runName = f"{args.logName}_{args.encoderModel}_{args.crossValidationRegion}"
    runName += f"_{args.bootstrapIndex}_nD{args.noDecoder}_dR{args.disableRhorads}"
    runName += f"_dM{args.decoderModel}_ff{args.fullFit}_rS{args.setRandomSeedTo}"

    wandb.init(
        project="ICLR_SOC_Analysis_2024",
        name=runName,
        config={
            "encoderModel": args.encoderModel,
            "crossValidationRegion": args.crossValidationRegion,
            "bootstrapIndex": args.bootstrapIndex,
            "epochs": args.epochs,
            "batch": args.batch,
            "noDecoder": args.noDecoder,
            "disableRhorads": args.disableRhorads,
            "decoderModel": args.decoderModel,
            "fullFit": args.fullFit,
            "regularizationTest": args.regularizationTest,
            "trainValSplit": args.trainValSplit,
            "fixRandomSeed": args.fixRandomSeed,
            "setRandomSeedTo": args.setRandomSeedTo,
            "spectraSOCLocation": args.spectraSOCLocation,
            "splitIndicesLocation": args.splitIndicesLocation,
            "endmemberSpectraLocation": args.endmemberSpectraLocation,
            "lr": args.lr,
            "b1": args.b1,
            "b2": args.b2,
            "finetuneEpochs": args.finetuneEpochs
        }
    )
        
    """ ############################################################################################
        Load datasets
    """
    dataset = HDF5Dataset(args.spectraSOCLocation)
    dataset_scanonly = HDF5Dataset(args.spectraSOCLocation, scanonly=True)

    ###
    # Get training data and validation data indices.
    # Load the indices file:
    indices_file = h5py.File(args.splitIndicesLocation, 'r')

    val_indices = None
    vso_indices = None
    train_indices = None
    tso_indices = None

    if args.trainValSplit < 0 :
        # Get the validation indices for the specified cross-validation region:
        val_indices = indices_file[f'{args.crossValidationRegion}/indices'][:] if not args.fullFit else None
        vso_indices = indices_file[f'{args.crossValidationRegion}/indices_scanonly'][:] if not args.fullFit else None

        # Train indices should cover the remaining RaCA regions:
        for i in range(1,19) :
            if i == 17 or i == args.crossValidationRegion : continue
            train_indices = torch.tensor(indices_file[f'{i}/indices'][:]) if train_indices is None else torch.cat((train_indices,torch.tensor(indices_file[f'{i}/indices'][:])))
            tso_indices = torch.tensor(indices_file[f'{i}/indices_scanonly'][:]) if train_indices is None else torch.cat((tso_indices,torch.tensor(indices_file[f'{i}/indices_scanonly'][:])))
    
    else :

        for i in range(1,19) :
            if i == 17 : continue
            region_indices = indices_file[f'{i}/indices'][:]
            reg_inds_scanonly = indices_file[f'{i}/indices_scanonly'][:]

            # split region by train/val split
            val_size = int(args.trainValSplit * len(region_indices))
            vso_size = int(args.trainValSplit * len(reg_inds_scanonly))
            
            region_indices = np.random.permutation(region_indices)
            reg_inds_scanonly = np.random.permutation(reg_inds_scanonly)

            val_indices   = torch.tensor(region_indices[:val_size]) if val_indices   is None else torch.cat((val_indices,  torch.tensor(region_indices[:val_size])))
            vso_indices   = torch.tensor(reg_inds_scanonly[:vso_size]) if vso_indices   is None else torch.cat((vso_indices,  torch.tensor(reg_inds_scanonly[:vso_size])))
            train_indices = torch.tensor(region_indices[val_size:]) if train_indices is None else torch.cat((train_indices,torch.tensor(region_indices[val_size:])))
            tso_indices   = torch.tensor(reg_inds_scanonly[vso_size:]) if tso_indices   is None else torch.cat((tso_indices,  torch.tensor(reg_inds_scanonly[vso_size:])))

    ###
    # Load training and validation datasets and prepare batch loaders:
    training_dataset   = torch.utils.data.Subset(dataset,          train_indices)
    tso_dataset        = torch.utils.data.Subset(dataset_scanonly, tso_indices)
    validation_dataset = torch.utils.data.Subset(dataset,          val_indices) if not args.fullFit else None
    vso_dataset        = torch.utils.data.Subset(dataset_scanonly, vso_indices) if not args.fullFit else None

    training_data_loader    = data.DataLoader(training_dataset,   batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    tso_data_loader         = data.DataLoader(tso_dataset,        batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    validation_data_loader  = data.DataLoader(validation_dataset, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True) if not args.fullFit else None
    vso_data_loader         = data.DataLoader(vso_dataset,        batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True) if not args.fullFit else None

    ###
    # Load bootstrap dataset
    bootstrap_indices,bootstrap_inds_scanonly,bootstrap_dataset,finetune_data_loader,fso_data_loader  = None, None, None, None, None
    if not args.fullFit :

        boot_keys = indices_file[f'{args.crossValidationRegion}'].keys()
        boot_keys = [x for x in boot_keys if 'indices' not in x]

        # select args.bootstrapIndex keys at random from the list of keys
        pedon_keys = np.random.choice(boot_keys, args.bootstrapIndex, replace=False)

        for pid in pedon_keys :
            aux_keys = [x for x in indices_file[f'{args.crossValidationRegion}/{pid}'].keys() if 'indices' not in x]

            new_boot_inds = torch.tensor(indices_file[f'{args.crossValidationRegion}/{pid}/indices'][:])
            new_boot_inds_scanonly = None

            for auxid in aux_keys :
                aux_inds = torch.tensor(indices_file[f'{args.crossValidationRegion}/{pid}/{auxid}/indices_scanonly'][:])
                new_boot_inds_scanonly = aux_inds if bootstrap_indices is None else torch.cat((new_boot_inds_scanonly,aux_inds))
            
            bootstrap_indices = new_boot_inds if bootstrap_indices is None else torch.cat((bootstrap_indices,new_boot_inds))
            bootstrap_inds_scanonly = new_boot_inds_scanonly if bootstrap_inds_scanonly is None else torch.cat((bootstrap_inds_scanonly,new_boot_inds_scanonly))

        bootstrap_dataset = torch.utils.data.Subset(dataset, bootstrap_indices)
        finetune_data_loader = data.DataLoader(bootstrap_dataset, batch_size=len(bootstrap_dataset), shuffle=True, num_workers=0, drop_last=True)

        bootstrap_scanonly_dataset = torch.utils.data.Subset(dataset_scanonly, bootstrap_inds_scanonly)
        fso_data_loader = data.DataLoader(bootstrap_scanonly_dataset, batch_size=len(bootstrap_scanonly_dataset), shuffle=True, num_workers=0, drop_last=True)

    ###
    # Load the endmember spectra and rhorads:
    endmember_file  = h5py.File(args.endmemberSpectraLocation, 'r')
    seedFs          = torch.tensor(endmember_file['Fs'][:]).to(device)
    seedrrs         = torch.tensor(endmember_file['rhorads'][:]).to(device)

    # If we disable rhorads, set them all to 1 so they have no effect
    if args.disableRhorads : 
        seedrrs = torch.ones(seedrrs.shape).to(device)

    # Move relevant datasets to the GPU
    trainingGTmsoc = dataset.dataset[train_indices,-5].to(device)
    trainingGTspec = dataset.dataset[train_indices,:-5].to(device)
    trainingSOspec = dataset_scanonly.dataset[tso_indices,:].to(device)

    validationGTmsoc = None if args.fullFit and args.trainValSplit < 0 else dataset.dataset[val_indices,-5].to(device)
    validationGTspec = None if args.fullFit and args.trainValSplit < 0 else dataset.dataset[val_indices,:-5].to(device)
    validationSOspec = None if args.fullFit and args.trainValSplit < 0 else dataset_scanonly.dataset[vso_indices,:].to(device)

    finetuneGTmsoc = None if args.fullFit else dataset.dataset[bootstrap_indices,-5].to(device)
    finetuneGTspec = None if args.fullFit else dataset.dataset[bootstrap_indices,:-5].to(device)
    finetuneSOspec = None if args.fullFit else dataset_scanonly.dataset[bootstrap_inds_scanonly,:].to(device)

    valb_indices = None if args.fullFit else [i for i in val_indices if i not in bootstrap_indices]
    valso_indices = None if args.fullFit else [i for i in vso_indices if i not in bootstrap_inds_scanonly]

    finetuneValGTmsoc = None if args.fullFit else dataset.dataset[valb_indices,-5].to(device)
    finetuneValGTspec = None if args.fullFit else dataset.dataset[valb_indices,:-5].to(device)
    finetuneValSOspec = None if args.fullFit else dataset_scanonly.dataset[valso_indices,:].to(device)

    """ ############################################################################################
        Prepare datasets
    """

    # Save the training and validation indices for this run
    # if we are doing a full fit with train/val split 
    if args.fullFit and args.trainValSplit < 0 :
        np.save(f"models/{runName}_training_indices.npy", train_indices)
        np.save(f"models/{runName}_validation_indices.npy", val_indices)

    """ ############################################################################################
        Prepare models
    """

    # Basic model parameters
    KEndmembers = seedrrs.shape[0]
    MSpectra = seedFs.shape[1]
    NSpectra = len(dataset)

    # Wavelength axis
    XF = torch.tensor([x for x in range(365,2501)]);

    # Generate priors for physical parameters and nuisances
    seedrrSOC = torch.mean(seedrrs[:-1])
    seedFsoc = torch.ones((MSpectra)) * 0.5

    seedrrs[-1] = seedrrSOC
    seedFs[-1,:] = seedFsoc

    # Set up encoder model and optimizer
    try:
        if args.encoderModel == "s":
            print("Using Standard Linear Model")
            encoder_model = LinearMixingEncoder(MSpectra, KEndmembers, 512).to(device)
        elif args.encoderModel == "c1":
            print("Using 1D Conv Model")
            encoder_model = EncoderConv1D(MSpectra, KEndmembers, 8, 10).to(device) # (M, K, hidden_size, kernel_size)
        elif args.encoderModel == "c1u":
            print("Using 1D Conv Model with Uncertainties")
            raise ValueError("Model error, please choose s (standard linear), c1 (1D conv), r (RNN), or t (Transformer)")
            #encoder_model = EncoderConv1DWithUncertainty(MSpectra, KEndmembers, 32, 15).to(device) # (M, K, hidden_size, kernel_size)
        elif args.encoderModel == "r":
            print("Using RNN Model")
            encoder_model = EncoderRNN(MSpectra, KEndmembers, 64).to(device) # (M, K, hidden_size)
        elif args.encoderModel == "t":
            print("Using Transformer Model")
            encoder_model = EncoderTransformer(MSpectra, KEndmembers, 64, 4, 2).to(device) # (M, K, hidden_size, num_heads, num_layers)
        else:
            raise ValueError("Model error, please choose s (standard linear), c1 (1D conv), r (RNN), or t (Transformer)")
    except ValueError as e:
        print(e)

    # print number of encoder model parameters
    print(f"Number of encoder model parameters: {sum(p.numel() for p in encoder_model.parameters() if p.requires_grad)}")

    # Set up decoder model and optimizer
    if args.noDecoder:
        decoder_model = None
    elif args.decoderModel :
        decoder_model = ANNDecoder(MSpectra, KEndmembers, 512).to(device)
    else :
        decoder_model = LinearMixingDecoder(seedFs, seedrrs).to(device)

    # Freeze the rhorads parameter if we are not using it
    if args.disableRhorads and not args.noDecoder and not args.decoderModel : 
        decoder_model.rrsoc.requires_grad = False

    # Set up optimizer
    combined_optimizer = optim.Adam(list(encoder_model.parameters()) + list([] if args.noDecoder else decoder_model.parameters()), lr=args.lr, betas=(args.b1, args.b2))


    """ ############################################################################################
        Train models
    """

    # Initialize best validation loss
    best_encoder_lossV = torch.tensor(float("inf"))

    for epoch in tqdm(range(args.epochs)):

        # Initialize loss variables for this epoch
        total_encoder_loss = 0.0
        total_decoder_loss = 0.0
        maxllf = 0.0
        total_loss_factor = 0.0

        # Batching and training
        for batch_data in training_data_loader:
            # Extract batch data
            batch_tIs, batch_tmsoc = batch_data[:,:-1].to(device), batch_data[:,-1].to(device)

            # Get abundance predictions from the encoder for the batch
            encoderPreds = encoder_model(batch_tIs)

            # Get spectrum predictions from the decoder for the batch
            decoderPreds = None if args.noDecoder else decoder_model(encoderPreds)

            # Compute encoder loss: sqerr from true Msoc values for the batch
            encoder_loss = torch.mean((encoderPreds[:, -1] - batch_tmsoc[:]) ** 2)

            # Add decoder loss: sqerr from true RaCA spectra for the batch
            decoder_loss = 0.0 if args.noDecoder else torch.mean((decoderPreds - batch_tIs) ** 2)

            # Multiply decoder loss by the Lagrange factor
            llf = 1.0 if args.noDecoder else decoder_model.computeLagrangeLossFactor()
            if llf > maxllf : maxllf = llf

            # Calculate the combined loss
            loss = (encoder_loss/(0.0041**2) + decoder_loss/(0.01**2)) * llf

            lossfactor = 1.0 
            if not args.noDecoder and args.regularizationTest :
                lossfactor += 100.0*torch.mean((encoderPreds[:,-1] - batch_tmsoc[:])**2 / 0.0041**2 / (torch.mean((decoderPreds - batch_tIs)**2 / 0.01**2, axis=1) + 50.0))

            loss = loss * lossfactor

            # Backpropagate the gradients for both models
            combined_optimizer.zero_grad()
            loss.backward()
            combined_optimizer.step()

            # Accumulate batch losses
            total_encoder_loss += encoder_loss.item()
            total_decoder_loss += decoder_loss if args.noDecoder else decoder_loss.item()
            total_loss_factor += lossfactor if args.noDecoder or not args.regularizationTest else lossfactor.item()

        # Calculate the average loss for this epoch
        avg_encoder_loss = total_encoder_loss / len(training_data_loader)
        avg_decoder_loss = total_decoder_loss / len(training_data_loader)
        avg_loss_factor = total_loss_factor / len(training_data_loader)

        wandb.log({"Encoder_Training_Loss": avg_encoder_loss, 
                   "Decoder_Training_Loss": avg_decoder_loss, 
                   "Total_Training_Loss": avg_encoder_loss/(0.0041**2) + avg_decoder_loss/(0.01**2),
                   "Max_LagrangeLossFactor": maxllf,
                   "LossFactor": avg_loss_factor}, step=epoch)

        """ ############################################################################################
            Validate models
        """
        # Validation Loss
        with torch.no_grad():

            if not args.fullFit:
                # Similar batching process for validation data
                total_encoder_lossV = 0.0
                total_decoder_lossV = 0.0

                for batch_dataV in validation_data_loader:

                    batch_val_tIs, batch_val_tmsoc = batch_dataV[:,:-1].to(device), batch_dataV[:,-1].to(device)

                    encoderPredsV = encoder_model(batch_val_tIs)
                    decoderPredsV = None if args.noDecoder else decoder_model(encoderPredsV)
                    
                    encoder_lossV = torch.mean((encoderPredsV[:, -1] - batch_val_tmsoc[:]) ** 2)
                    decoder_lossV = 0 if args.noDecoder else torch.mean((decoderPredsV - batch_val_tIs) ** 2)

                    total_encoder_lossV += encoder_lossV.item()
                    total_decoder_lossV += decoder_lossV if args.noDecoder else decoder_lossV.item()

                avg_encoder_lossV = total_encoder_lossV / len(validation_data_loader)
                avg_decoder_lossV = total_decoder_lossV / len(validation_data_loader)

                wandb.log({"Encoder_Validation_Loss": avg_encoder_lossV, 
                           "Decoder_Validation_Loss": avg_decoder_lossV,
                           "Total_Validation_Loss": avg_encoder_lossV/(0.0041**2) + avg_decoder_lossV/(0.01**2)}, step=epoch)
        
            if epoch == 0 or epoch % 50 == 0 or epoch >= args.epochs - 11:
                # Log in wandb the following on the training and validation sets:
                #   - Mean Square Error of Performance (MSEP) for encoder model
                #   - MSEP for decoder model
                #   - R^2 Score for SOC predictions
                #   - Bias for SOC predictions
                #   - Ratio of performance to deviation (RPD) for SOC predictions
                
                # Get preds on full training set
                trainingEncoderPreds = encoder_model(trainingGTspec)
                trainingDecoderPreds = None if args.noDecoder else decoder_model(trainingEncoderPreds)

                # Compute metrics for training set
                trainingRMSEP = torch.sqrt(torch.mean((trainingEncoderPreds[:, -1] - trainingGTmsoc) ** 2))
                trainingR2 = r2_score(trainingGTmsoc, trainingEncoderPreds[:, -1])
                trainingBias = torch.mean(trainingEncoderPreds[:, -1] - trainingGTmsoc)
                trainingRPD = torch.std(trainingEncoderPreds[:, -1]) / torch.std(trainingGTmsoc)

                trainingDecoderRMSEP = 0 if args.noDecoder else torch.sqrt(torch.mean((trainingDecoderPreds - trainingGTspec) ** 2))
                trainingDecoderMRMSE = 0 if args.noDecoder else torch.mean(torch.sqrt(torch.mean((trainingDecoderPreds - trainingGTspec) ** 2,axis=1)))

                # Log metrics in wandb
                wandb.log({"Encoder_Training_RMSEP": trainingRMSEP,
                            "Encoder_Training_R2": trainingR2,
                            "Encoder_Training_Bias": trainingBias,
                            "Encoder_Training_RPD": trainingRPD,
                            "Decoder_Training_RMSEP": trainingDecoderRMSEP,
                            "Decoder_Training_MRMSE": trainingDecoderMRMSE}, step=epoch)

                # Compute metrics for validation set
                if not args.fullFit or args.trainValSplit > 0 :
                    # Get preds on full val set
                    validationEncoderPreds = encoder_model(validationGTspec)
                    validationDecoderPreds = None if args.noDecoder else decoder_model(validationEncoderPreds)

                    validationRMSEP = torch.sqrt(torch.mean((validationEncoderPreds[:, -1] - validationGTmsoc) ** 2))
                    validationR2 = r2_score(validationGTmsoc, validationEncoderPreds[:, -1])
                    validationBias = torch.mean(validationEncoderPreds[:, -1] - validationGTmsoc)
                    validationRPD = torch.std(validationEncoderPreds[:, -1]) / torch.std(validationGTmsoc)

                    validationDecoderRMSEP = 0 if args.noDecoder else torch.sqrt(torch.mean((validationDecoderPreds - validationGTspec) ** 2))
                    validationDecoderMRMSE = 0 if args.noDecoder else torch.mean(torch.sqrt(torch.mean((validationDecoderPreds - validationGTspec) ** 2,axis=1)))

                    wandb.log({"Encoder_Validation_RMSEP": validationRMSEP,
                            "Encoder_Validation_R2": validationR2,
                            "Encoder_Validation_Bias": validationBias,
                            "Encoder_Validation_RPD": validationRPD,
                            "Decoder_Validation_RMSEP": validationDecoderRMSEP,
                            "Decoder_Validation_MRMSE": validationDecoderMRMSE}, step=epoch)
                
                # Log decoder model parameters in wandb
                if not args.noDecoder :
                    if not args.decoderModel :
                        wandb.log({"rrSOC": decoder_model.rrsoc.detach().item()}, step=epoch)

                        # Log fsoc graph in wandb
                        tfsoc = decoder_model.fsoc.detach()
                        fsocTableDat = [[x, y] for (x, y) in zip(XF,tfsoc)]
                        fsocTable = wandb.Table(data=fsocTableDat, columns=["Wavelength", "SOC Reflectance"])
                        wandb.log(
                            {
                                "Fsoc": wandb.plot.line(
                                    fsocTable, "Wavelength", "SOC Reflectance", title="Regressed SOC Spectrum"
                                )
                            }, step=epoch
                        )

                    # Log pred errors on validation set
                    predSOCerr = ((validationEncoderPreds[:, -1] - validationGTmsoc)**2/0.0041/0.0041).detach()
                    predRMSEP = torch.mean((validationDecoderPreds - validationGTspec) ** 2/0.01/0.01,axis=1)
                    predTableDat = [[x, y] for (x, y) in zip(predSOCerr,predRMSEP)]
                    predTable = wandb.Table(data=predTableDat, columns=["SOC prediction error", "Spectrum prediction RMSE"])
                    wandb.log(
                        {
                            "EvD_Val_Pred_Error": wandb.plot.line(
                                predTable, "SOC prediction error", "Spectrum prediction RMSE", title="Spectrum Prediction Error vs. SOC Prediction Error"
                            )
                        }, step=epoch
                    )

                    # Log pred errors on validation set as function of wavelength
                    vpredRMSEP = torch.sqrt(torch.mean((validationDecoderPreds - validationGTspec) ** 2,axis=0))
                    vpredTableDat = [[x, y] for (x, y) in zip(XF,vpredRMSEP)]
                    vpredTable = wandb.Table(data=vpredTableDat, columns=["Wavelength [nm]", "Spectrum prediction RMSE"])
                    wandb.log(
                        {
                            "D_Val_Pred_Error_Wavelength": wandb.plot.line(
                                vpredTable, "Wavelength [nm]", "Spectrum prediction RMSE", title="Spectrum Prediction Error vs. Wavelength"
                            )
                        }, step=epoch
                    )

                    # Log pred errors on training set as function of wavelength
                    tpredRMSEP = torch.sqrt(torch.mean((trainingDecoderPreds - trainingGTspec) ** 2,axis=0))
                    tpredTableDat = [[x, y] for (x, y) in zip(XF,tpredRMSEP)]
                    tpredTable = wandb.Table(data=tpredTableDat, columns=["Wavelength [nm]", "Spectrum prediction RMSE"])
                    wandb.log(
                        {
                            "D_Train_Pred_Error_Wavelength": wandb.plot.line(
                                tpredTable, "Wavelength [nm]", "Spectrum prediction RMSE", title="Spectrum Prediction Error vs. Wavelength"
                            )
                        }, step=epoch
                    )

        
    torch.save(encoder_model.state_dict(), f"models/{runName}_encoder_final.pt")

    if not args.noDecoder:
        torch.save(decoder_model.state_dict(), f"models/{runName}_decoder_final.pt")


    """ ############################################################################################
        Fine-tune models
    """
    if not args.fullFit :
        # Set up optimizer
        combined_optimizer = optim.Adam(list(encoder_model.parameters()) + list([] if args.noDecoder else decoder_model.parameters()), lr=args.lr, betas=(args.b1, args.b2))

        # Initialize best validation loss
        best_encoder_lossV = torch.tensor(float("inf"))

        for epoch in tqdm(range(args.finetuneEpochs)):

            # Initialize loss variables for this epoch
            total_encoder_loss = 0.0
            total_decoder_loss = 0.0
            maxllf = 0.0

            # Batching and training
            for batch_data in finetune_data_loader:
                # Extract batch data
                batch_tIs, batch_tmsoc = batch_data[:,:-1].to(device), batch_data[:,-1].to(device)

                # Get abundance predictions from the encoder for the batch
                #encoderPreds, encoderUncertainties, mean = encoder_model(batch_tIs) #edit to only have 2nd arg if model is c1u
                encoderPreds = encoder_model(batch_tIs)

                # Get spectrum predictions from the decoder for the batch
                decoderPreds = None if args.noDecoder else decoder_model(encoderPreds)

                # Compute encoder loss: sqerr from true Msoc values for the batch
                encoder_loss = torch.mean((encoderPreds[:, -1] - batch_tmsoc[:]) ** 2)

                # Add decoder loss: sqerr from true RaCA spectra for the batch
                decoder_loss = 0 if args.noDecoder else torch.mean((decoderPreds - batch_tIs) ** 2)

                # Multiply decoder loss by the Lagrange factor
                llf = 1.0 if args.noDecoder else decoder_model.computeLagrangeLossFactor()
                if llf > maxllf : maxllf = llf

                # Calculate the combined loss
                loss = (encoder_loss/(0.0041**2) + decoder_loss/(0.01**2)) * llf

                # Backpropagate the gradients for both models
                combined_optimizer.zero_grad()
                loss.backward()
                combined_optimizer.step()

                # Accumulate batch losses
                total_encoder_loss += encoder_loss.item()
                total_decoder_loss += decoder_loss if args.noDecoder else decoder_loss.item()

            # Calculate the average loss for this epoch
            avg_encoder_loss = total_encoder_loss / len(finetune_data_loader)
            avg_decoder_loss = total_decoder_loss / len(finetune_data_loader)

            wandb.log({"Encoder_Finetune_Loss": avg_encoder_loss,
                       "Decoder_Finetune_Loss": avg_decoder_loss,
                       "Total_Finetune_Loss": avg_encoder_loss/(0.0041**2) + avg_decoder_loss/(0.01**2),
                       "Max_Finetune_LagrangeLossFactor": maxllf}, step=args.epochs+epoch)
    
            """ ############################################################################################
                Validate fine-tuned model
            """
             # Validation Loss
            with torch.no_grad():

                # Similar batching process for validation data
                total_encoder_lossV = 0.0
                total_decoder_lossV = 0.0

                for batch_dataV in validation_data_loader:

                    batch_val_tIs, batch_val_tmsoc = batch_dataV[:,:-1].to(device), batch_dataV[:,-1].to(device)

                    #encoderPredsV, encoderUncertaintiesV, meanV = encoder_model(batch_val_tIs)
                    encoderPredsV = encoder_model(batch_val_tIs)

                    decoderPredsV = None if args.noDecoder else decoder_model(encoderPredsV)
                    encoder_lossV = torch.mean((encoderPredsV[:, -1] - batch_val_tmsoc[:]) ** 2)
                    decoder_lossV = 0 if args.noDecoder else torch.mean((decoderPredsV - batch_val_tIs) ** 2)

                    total_encoder_lossV += encoder_lossV.item()
                    total_decoder_lossV += decoder_lossV if args.noDecoder else decoder_lossV.item()

                avg_encoder_lossV = total_encoder_lossV / len(validation_data_loader)
                avg_decoder_lossV = total_decoder_lossV / len(validation_data_loader)

                wandb.log({"Encoder_FinetuneValidation_Loss": avg_encoder_lossV, 
                          "Decoder_FinetuneValidation_Loss": avg_decoder_lossV,
                          "Total_FinetuneValidation_Loss": avg_encoder_lossV/(0.0041**2) + avg_decoder_lossV/(0.01**2)}, step=args.epochs+epoch)
            
                if epoch % 10 == 0 or epoch >= args.finetuneEpochs - 11 :
                    # Log in wandb the following on the training and validation sets:
                    #   - Mean Square Error of Performance (RMSEP) for encoder model
                    #   - RMSEP for decoder model
                    #   - R^2 Score for SOC predictions
                    #   - Bias for SOC predictions
                    #   - Ratio of performance to deviation (RPD) for SOC predictions
                    
                    # Get preds on full training set
                    finetuneEncoderPreds = encoder_model(finetuneGTspec)
                    finetuneDecoderPreds = None if args.noDecoder else decoder_model(finetuneEncoderPreds)

                    # Compute metrics for training set
                    finetuneRMSEP = torch.sqrt(torch.mean((finetuneEncoderPreds[:, -1] - finetuneGTmsoc) ** 2))
                    finetuneR2 = r2_score(finetuneGTmsoc, finetuneEncoderPreds[:, -1])
                    finetuneBias = torch.mean(finetuneEncoderPreds[:, -1] - finetuneGTmsoc)
                    finetuneRPD = torch.std(finetuneEncoderPreds[:, -1]) / torch.std(finetuneGTmsoc)

                    finetuneDecoderRMSEP = 0 if args.noDecoder else torch.sqrt(torch.mean((finetuneDecoderPreds - finetuneGTspec) ** 2))
                    finetuneDecoderMRMSE = 0 if args.noDecoder else torch.mean(torch.sqrt(torch.mean((finetuneDecoderPreds - finetuneGTspec) ** 2,axis=1)))

                    # Log metrics in wandb
                    wandb.log({"Encoder_FinetuneTraining_RMSEP": finetuneRMSEP,
                                "Encoder_FinetuneTraining_R2": finetuneR2,
                                "Encoder_FinetuneTraining_Bias": finetuneBias,
                                "Encoder_FinetuneTraining_RPD": finetuneRPD,
                                "Decoder_FinetuneTraining_RMSEP": finetuneDecoderRMSEP,
                                "Decoder_FinetuneTraining_MRMSE": finetuneDecoderMRMSE}, step=args.epochs+epoch)

                    # Get preds on full val set
                    validationEncoderPreds = encoder_model(finetuneValGTspec)
                    validationDecoderPreds = None if args.noDecoder else decoder_model(validationEncoderPreds)

                    validationRMSEP = torch.sqrt(torch.mean((validationEncoderPreds[:, -1] - finetuneValGTmsoc) ** 2))
                    validationR2 = r2_score(finetuneValGTmsoc, validationEncoderPreds[:, -1])
                    validationBias = torch.mean(validationEncoderPreds[:, -1] - finetuneValGTmsoc)
                    validationRPD = torch.std(validationEncoderPreds[:, -1]) / torch.std(finetuneValGTmsoc)

                    validationDecoderRMSEP = 0 if args.noDecoder else torch.sqrt(torch.mean((validationDecoderPreds - finetuneValGTspec) ** 2))
                    validationDecoderMRMSE = 0 if args.noDecoder else torch.mean(torch.sqrt(torch.mean((validationDecoderPreds - finetuneValGTspec) ** 2,axis=1)))

                    wandb.log({"Encoder_FinetuneValidation_RMSEP": validationRMSEP,
                            "Encoder_FinetuneValidation_R2": validationR2,
                            "Encoder_FinetuneValidation_Bias": validationBias,
                            "Encoder_FinetuneValidation_RPD": validationRPD,
                            "Decoder_FinetuneValidation_RMSEP": validationDecoderRMSEP,
                            "Decoder_FinetuneValidation_MRMSE": validationDecoderMRMSE}, step=args.epochs+epoch)
                    
                    # Log decoder model parameters in wandb
                    if not args.noDecoder : 
                        if not args.decoderModel :
                            wandb.log({"rrSOC_finetuned": decoder_model.rrsoc.detach().item()}, step=args.epochs+epoch)

                            # Log fsoc graph in wandb
                            tfsoc = decoder_model.fsoc.detach()
                            fsocTableDat = [[x, y] for (x, y) in zip(XF,tfsoc)]
                            fsocTable = wandb.Table(data=fsocTableDat, columns=["Wavelength", "SOC Reflectance"])
                            wandb.log(
                                {
                                    "Fsoc_finetuned": wandb.plot.line(
                                        fsocTable, "Wavelength", "SOC Reflectance", title="Regressed SOC Spectrum"
                                    )
                                }, step=args.epochs+epoch
                            )

                        # Log pred errors on validation set
                        predSOCerr = (validationEncoderPreds[:, -1] - finetuneValGTmsoc).detach()
                        predRMSEP = torch.sqrt(torch.mean((validationDecoderPreds - finetuneValGTspec) ** 2,axis=1))
                        predTableDat = [[x, y] for (x, y) in zip(predSOCerr,predRMSEP)]
                        predTable = wandb.Table(data=predTableDat, columns=["SOC prediction error", "Spectrum prediction RMSE"])
                        wandb.log(
                            {
                                "EvD_Finetuned_Val_Pred_Error": wandb.plot.line(
                                    predTable, "SOC prediction error", "Spectrum prediction RMSE", title="Spectrum Prediction Error vs. SOC Prediction Error"
                                )
                            }, step=args.epochs+epoch
                        )

                        # Log pred errors on validation set as function of wavelength
                        vpredRMSEP = torch.sqrt(torch.mean((validationDecoderPreds - finetuneValGTspec) ** 2,axis=0))
                        vpredTableDat = [[x, y] for (x, y) in zip(XF,vpredRMSEP)]
                        vpredTable = wandb.Table(data=vpredTableDat, columns=["Wavelength [nm]", "Spectrum prediction RMSE"])
                        wandb.log(
                            {
                                "D_Val_Pred_Error_Wavelength": wandb.plot.line(
                                    vpredTable, "Wavelength [nm]", "Spectrum prediction RMSE", title="Spectrum Prediction Error vs. Wavelength"
                                )
                            }, step=args.epochs+epoch
                        )

        torch.save(encoder_model.state_dict(), f"models/{runName}_encoder_finetuned.pt")

        if not args.noDecoder:
            torch.save(decoder_model.state_dict(), f"models/{runName}_decoder_finetuned.pt")    


    """ ############################################################################################
        Cleanup
    """
    # Close the HDF5 files
    indices_file.close()
    endmember_file.close()

    # Finish wandb run
    wandb.finish()