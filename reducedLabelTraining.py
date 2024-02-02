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
    def __init__(self, filename, gpu=True):
        super(HDF5Dataset, self).__init__()
        self.file_mag = h5py.File(filename, 'r')
        if gpu :
            self.dataset = torch.tensor(self.file_mag['data'][:]).to(device)
        else :
            self.dataset = torch.tensor(self.file_mag['data'][:])

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        return self.dataset[index,:]


""" ############################################################################################
    To run this script, use the following command:
        
        python3 updatedTrain.py [model (s,t,r,c1, c1u)] [epochs] [val_ratio] [batch_size]
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for different models")
    parser.add_argument("--encoderModel", choices=["s", "c1", "c1u", "r", "t"], type=str, help="Choose a model: s = Standard Linear, c1 = 1D CNN, r = RNN, t = Transformer")
    parser.add_argument("--crossValidationRegion", type=int, default=-1, help="RaCA region number for Leave-One-Region-Out cross-validation.")   
    parser.add_argument("--bootstrapIndex",        type=int, default=-1, help="Bootstrap sample number for fine-tuning within validation region.")
    parser.add_argument("--epochs",     type=int, default=10,       help="Number of training epochs")
    parser.add_argument("--batch",      type=int, default=75,       help="Batch Size")
    parser.add_argument("--logName",    type=str, default="test",   help="Base name for output files.") 

    parser.add_argument("--noDecoder",      default=False, action='store_true', help="Flag to disable decoder model and only consider end-to-end encoder performance.") 
    parser.add_argument("--disableRhorads", default=False, action='store_true', help="Flag to disable conversion of mass abundance to area abundance via rhorads.") 
    parser.add_argument("--decoderModel",   default=False, action='store_true', help="Flag to implement an ANN decoder model in place of the linear mixing model.") 
    parser.add_argument("--fullFit",        default=False, action='store_true', help="Flag to fit the entire dataset, without validation.") 
    parser.add_argument("--regularizationTest", default=False, action='store_true', help="Flag to test regularization technique.")
    parser.add_argument("--fixRandomSeed", default=False, action='store_true', help="Flag to fix random seed for reproducibility.")
    parser.add_argument("--setRandomSeedTo", type=int, default=0, help="Set random seed to this value.")

    parser.add_argument("--spectraSOCLocation",         type=str, default="data_utils/ICLRDataset_RaCASpectraAndSOC.h5", help="File name for soil spectra and SOC numbers.") 
    parser.add_argument("--splitIndicesLocation",       type=str, default="data_utils/ICLRDataset_splitIndices.h5", help="File name for soil spectrum index, split by region number.") 
    parser.add_argument("--endmemberSpectraLocation",   type=str, default="data_utils/ICLRDataset_USGSEndmemberSpectra.h5", help="File name for pure endmember spectra and rhorads.") 

    parser.add_argument("--lr",  type=float, default=0.00005, help="Learning rate for Adam optimizer.")
    parser.add_argument("--b1",  type=float, default=0.99,    help="Beta1 for Adam optimizer.")
    parser.add_argument("--b2",  type=float, default=0.999,  help="Beta2 for Adam optimizer.")

    parser.add_argument("--finetuneEpochs",     type=int, default=10000,       help="Number of training epochs for the fine-tuning step.")

    
    args = parser.parse_args()

    if args.fixRandomSeed :
        torch.manual_seed(args.setRandomSeedTo)
        random.seed(args.setRandomSeedTo)
        np.random.seed(args.setRandomSeedTo)

    runName = f"{args.logName}_{args.encoderModel}_{args.crossValidationRegion}_"
    runName += f"{args.bootstrapIndex}_nD{args.noDecoder}_dR{args.disableRhorads}"
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
    dataset_size = len(dataset)

    ###
    # Get training data and validation data indices.
    # Load the indices file:
    indices_file = h5py.File(args.splitIndicesLocation, 'r')

    val_indices = None
    train_indices = None
    bootstrap_indices = None

    # Get the validation indices for the entire specified cross-validation region.
    # For fine-tuning, split the validation indices into bootstrap and non-bootstrap indices.
    val_bootstrap_indices, val_nobootstr_indices = None, None
    if not args.fullFit :
        val_indices = indices_file[f'{args.crossValidationRegion}_indices'][:]
        val_bootstrap_indices = indices_file[f'{args.crossValidationRegion}_bootstrap_{args.bootstrapIndex}'][:]

        # Take only half of the bootstrap indices for the fine-tuning dataset
        # val_bootstrap_indices = val_bootstrap_indices[:len(val_bootstrap_indices)//2]

        val_nobootstr_indices = [i for i in val_indices if i not in val_bootstrap_indices]

    # Train indices should cover the remaining RaCA regions:
    for i in range(1,19) :
        if i == 17 or i == args.crossValidationRegion : continue

        region_indices = torch.tensor(indices_file[f'{i}_indices'][:])
        region_bootstr = torch.tensor(indices_file[f'{i}_bootstrap_{args.bootstrapIndex}'][:])

        region_bootstr3 = torch.tensor(indices_file[f'{i}_bootstrap_3'][:])

        region_bootstr = torch.cat((region_bootstr,region_bootstr3))
        
        # Delete duplicates from region_bootstr
        region_bootstr = torch.unique(region_bootstr)

        # Take only half of the bootstrap indices for the fine-tuning dataset
        # region_bootstr = region_bootstr[:len(region_bootstr)//2]

        train_indices = region_indices if train_indices is None else torch.cat((train_indices,region_indices))
        bootstrap_indices = region_bootstr if bootstrap_indices is None else torch.cat((bootstrap_indices,region_bootstr))

    # Remove bootstrap indices from training indices:
    train_indices = [i for i in train_indices if i not in bootstrap_indices]

    ###
    # Load training and validation datasets and prepare batch loaders:
    training_dataset   = torch.utils.data.Subset(dataset, train_indices)
    bootstrap_dataset  = torch.utils.data.Subset(dataset, bootstrap_indices)
    validation_dataset = torch.utils.data.Subset(dataset, val_indices) if not args.fullFit else None

    # Load the fine-tuning dataset if we are not doing a full fit
    valb_dataset  = torch.utils.data.Subset(dataset, val_bootstrap_indices) if not args.fullFit else None
    valnb_dataset = torch.utils.data.Subset(dataset, val_nobootstr_indices) if not args.fullFit else None

    # Get the data loaders for the training and validation datasets
    training_data_loader    = data.DataLoader(training_dataset,   batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    num_batches = len(training_data_loader)

    boot_batch_size     = len(bootstrap_dataset) // num_batches
    bootstrap_data_loader   = data.DataLoader(bootstrap_dataset, batch_size=boot_batch_size, shuffle=True, num_workers=0, drop_last=True)

    validation_data_loader  = None if args.fullFit else data.DataLoader(validation_dataset, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    # Get the data loaders for the fine-tuning datasets
    finetuneScan_data_loader, finetuneBoot_data_loader = None, None
    if not args.fullFit :
        finetuneScan_data_loader = data.DataLoader(valnb_dataset, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
        num_val_batches = len(finetuneScan_data_loader)

        valboot_batch_size = len(valb_dataset) // num_val_batches
        finetuneBoot_data_loader = data.DataLoader(valb_dataset,  batch_size=valboot_batch_size, shuffle=True, num_workers=0, drop_last=True)

    ###
    # Load the endmember spectra and rhorads:
    endmember_file  = h5py.File(args.endmemberSpectraLocation, 'r')
    seedFs          = torch.tensor(endmember_file['Fs'][:]).to(device)
    seedrrs         = torch.tensor(endmember_file['rhorads'][:]).to(device)

    # If we disable rhorads, set them all to 1 so they have no effect
    if args.disableRhorads : 
        seedrrs = torch.ones(seedrrs.shape).to(device)

    # Move relevant datasets to the GPU
    trainingGTmsoc = dataset.dataset[train_indices,-1].to(device)
    trainingGTspec = dataset.dataset[train_indices,:-1].to(device)

    bootstrapGTmsoc = dataset.dataset[bootstrap_indices,-1].to(device)
    bootstrapGTspec = dataset.dataset[bootstrap_indices,:-1].to(device)

    validationGTmsoc = None if args.fullFit else dataset.dataset[val_indices,-1].to(device)
    validationGTspec = None if args.fullFit else dataset.dataset[val_indices,:-1].to(device)

    valbGTmsoc = None if args.fullFit else dataset.dataset[val_bootstrap_indices, -1].to(device)
    valbspec   = None if args.fullFit else dataset.dataset[val_bootstrap_indices,:-1].to(device)

    valnbGTmsoc = None if args.fullFit else dataset.dataset[val_nobootstr_indices, -1].to(device)
    valnbspec   = None if args.fullFit else dataset.dataset[val_nobootstr_indices,:-1].to(device)

    """ ############################################################################################
        Prepare models
    """

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

    for epoch in tqdm(range(args.epochs)):

        # Initialize loss variables for this epoch
        total_encoder_loss = 0.0
        total_decoder_loss = 0.0
        
        maxllf = 0.0
        
        total_loss_factor = 0.0

        # Batching and training
        bootstrap_iter = iter(bootstrap_data_loader)

        for batch_dataT in training_data_loader:

            batch_dataB = next(bootstrap_iter)

            # Extract batch data
            batch_tIsT, batch_tmsocT = batch_dataT[:,:-1].to(device), batch_dataT[:,-1].to(device)
            batch_tIsB, batch_tmsocB = batch_dataB[:,:-1].to(device), batch_dataB[:,-1].to(device)

            batchSizeT = batch_tIsT.shape[0]
            batchSizeB = batch_tIsB.shape[0]

            trainingEncoderLossNum = batchSizeB
            trainingDecoderLossNum = batchSizeT + batchSizeB

            # Get abundance predictions from the encoder for the batch
            encoderPredsT = encoder_model(batch_tIsT)
            encoderPredsB = encoder_model(batch_tIsB)

            # Get spectrum predictions from the decoder for the batch
            decoderPredsT = None if args.noDecoder else decoder_model(encoderPredsT)
            decoderPredsB = None if args.noDecoder else decoder_model(encoderPredsB)

            # Compute encoder loss: sqerr from true Msoc values for the batch
            encoder_loss = torch.sum((encoderPredsB[:, -1] - batch_tmsocB[:]) ** 2)/trainingEncoderLossNum

            # Add decoder loss: sqerr from true RaCA spectra for the batch
            decoder_loss = 0.0 
            if not args.noDecoder :
                decoder_loss += torch.mean((decoderPredsT - batch_tIsT) ** 2) * batchSizeT
                decoder_loss += torch.mean((decoderPredsB - batch_tIsB) ** 2) * batchSizeB
                decoder_loss /= trainingDecoderLossNum

            # Multiply decoder loss by the Lagrange factor
            llf = 1.0 if args.noDecoder else decoder_model.computeLagrangeLossFactor()
            if llf > maxllf : maxllf = llf

            # Calculate the combined loss
            loss = (encoder_loss/(0.0041**2) + decoder_loss/(0.01**2)) * llf

            lossfactor = 1.0 
            if not args.noDecoder and args.regularizationTest :
                lossfactor += 100.0*torch.mean((encoderPredsB[:,-1] - batch_tmsocB[:])**2 / 0.0041**2 / (torch.mean((decoderPredsB - batch_tIsB)**2 / 0.01**2, axis=1) + 50.0))

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
        
            if epoch == 0 or epoch % 10 == 0 or epoch >= args.epochs - 11:
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
                if not args.fullFit :
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

                    # Log pred errors on validation set
                    tpredSOCerr = ((trainingEncoderPreds[:, -1] - trainingGTmsoc)**2/0.0041/0.0041).detach()
                    tpredRMSEP = torch.mean((trainingDecoderPreds - trainingGTspec) ** 2/0.01/0.01,axis=1)
                    tpredTableDat = [[x, y] for (x, y) in zip(tpredSOCerr,tpredRMSEP)]
                    tpredTable = wandb.Table(data=tpredTableDat, columns=["SOC prediction error", "Spectrum prediction RMSE"])
                    wandb.log(
                        {
                            "EvD_Train_Pred_Error": wandb.plot.line(
                                tpredTable, "SOC prediction error", "Spectrum prediction RMSE", title="Spectrum Prediction Error vs. SOC Prediction Error"
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

        for epoch in tqdm(range(args.finetuneEpochs)):

            # Initialize loss variables for this epoch
            total_encoder_loss = 0.0
            total_decoder_loss = 0.0
            maxllf = 0.0

            # Batching and training
            bootstrap_iter = iter(finetuneBoot_data_loader)

            for batch_dataT in finetuneScan_data_loader:

                batch_dataB = next(bootstrap_iter)

                # Extract batch data
                batch_tIsT, batch_tmsocT = batch_dataT[:,:-1].to(device), batch_dataT[:,-1].to(device)
                batch_tIsB, batch_tmsocB = batch_dataB[:,:-1].to(device), batch_dataB[:,-1].to(device)

                batchSizeT = batch_tIsT.shape[0]
                batchSizeB = batch_tIsB.shape[0]

                trainingEncoderLossNum = batchSizeB
                trainingDecoderLossNum = batchSizeT + batchSizeB

                # Get abundance predictions from the encoder for the batch
                encoderPredsT = encoder_model(batch_tIsT)
                encoderPredsB = encoder_model(batch_tIsB)

                # Get spectrum predictions from the decoder for the batch
                decoderPredsT = None if args.noDecoder else decoder_model(encoderPredsT)
                decoderPredsB = None if args.noDecoder else decoder_model(encoderPredsB)

                # Compute encoder loss: sqerr from true Msoc values for the batch
                encoder_loss = torch.sum((encoderPredsB[:, -1] - batch_tmsocB[:]) ** 2)/trainingEncoderLossNum

                # Add decoder loss: sqerr from true RaCA spectra for the batch
                decoder_loss = 0.0 
                if not args.noDecoder :
                    decoder_loss += torch.mean((decoderPredsT - batch_tIsT) ** 2) * batchSizeT
                    decoder_loss += torch.mean((decoderPredsB - batch_tIsB) ** 2) * batchSizeB
                    decoder_loss /= trainingDecoderLossNum

                # Multiply decoder loss by the Lagrange factor
                llf = 1.0 if args.noDecoder else decoder_model.computeLagrangeLossFactor()
                if llf > maxllf : maxllf = llf

                # Calculate the combined loss
                loss = (encoder_loss/(0.0041**2) + decoder_loss/(0.01**2)) * llf

                lossfactor = 1.0 
                if not args.noDecoder and args.regularizationTest :
                    lossfactor += 100.0*torch.mean((encoderPredsB[:,-1] - batch_tmsocB[:])**2 / 0.0041**2 / (torch.mean((decoderPredsB - batch_tIsB)**2 / 0.01**2, axis=1) + 50.0))

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
            avg_encoder_loss = total_encoder_loss / len(finetuneScan_data_loader)
            avg_decoder_loss = total_decoder_loss / len(finetuneScan_data_loader)

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
            
                if epoch % 5 == 0 or epoch >= args.finetuneEpochs - 11 :
                    # Log in wandb the following on the training and validation sets:
                    #   - Mean Square Error of Performance (RMSEP) for encoder model
                    #   - RMSEP for decoder model
                    #   - R^2 Score for SOC predictions
                    #   - Bias for SOC predictions
                    #   - Ratio of performance to deviation (RPD) for SOC predictions
                    
                    # Get preds on full training set
                    finetuneEncoderPreds = encoder_model(valnbspec)
                    finetuneDecoderPreds = None if args.noDecoder else decoder_model(finetuneEncoderPreds)

                    # Compute metrics for training set
                    finetuneRMSEP = torch.sqrt(torch.mean((finetuneEncoderPreds[:, -1] - valnbGTmsoc) ** 2))
                    finetuneR2 = r2_score(valnbGTmsoc, finetuneEncoderPreds[:, -1])
                    finetuneBias = torch.mean(finetuneEncoderPreds[:, -1] - valnbGTmsoc)
                    finetuneRPD = torch.std(finetuneEncoderPreds[:, -1]) / torch.std(valnbGTmsoc)

                    finetuneDecoderRMSEP = 0 if args.noDecoder else torch.sqrt(torch.mean((finetuneDecoderPreds - valnbspec) ** 2))
                    finetuneDecoderMRMSE = 0 if args.noDecoder else torch.mean(torch.sqrt(torch.mean((finetuneDecoderPreds - valnbspec) ** 2,axis=1)))

                    # Log metrics in wandb
                    wandb.log({"Encoder_FinetuneValidation_RMSEP": finetuneRMSEP,
                                "Encoder_FinetuneValidation_R2": finetuneR2,
                                "Encoder_FinetuneValidation_Bias": finetuneBias,
                                "Encoder_FinetuneValidation_RPD": finetuneRPD,
                                "Decoder_FinetuneValidation_RMSEP": finetuneDecoderRMSEP,
                                "Decoder_FinetuneValidation_MRMSE": finetuneDecoderMRMSE}, step=args.epochs+epoch)
                    
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
                        predSOCerr = (finetuneEncoderPreds[:, -1] - valnbGTmsoc).detach()
                        predRMSEP = torch.sqrt(torch.mean((finetuneDecoderPreds - valnbspec) ** 2,axis=1))
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
                        vpredRMSEP = torch.sqrt(torch.mean((finetuneDecoderPreds - valnbspec) ** 2,axis=0))
                        vpredTableDat = [[x, y] for (x, y) in zip(XF,vpredRMSEP)]
                        vpredTable = wandb.Table(data=vpredTableDat, columns=["Wavelength [nm]", "Spectrum prediction RMSE"])
                        wandb.log(
                            {
                                "D_Finetuned_Val_Pred_Error_Wavelength": wandb.plot.line(
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