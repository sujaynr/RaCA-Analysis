# Standard libraries
import pdb
import time
import argparse
import copy

# Third-party libraries
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
            self.dataset = torch.tensor(self.file_mag['data'][:])
        else :
            self.dataset = torch.tensor(self.file_mag['data'][:])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index,:][:]


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

    parser.add_argument("--disableRhorads", default=False, action='store_true', help="Flag to indicate that the decoder model had all rhorads frozen at 1.") 
    parser.add_argument("--decoderModel", type=int, default=0, help="Flag to indicate that an ANN decoder model was used in place of the linear mixing model.") 
    parser.add_argument("--noDecoder", type=int, default=0, help="Flag to indicate that no decoder model was used.")

    parser.add_argument("--noTrainingData",     default=False, action='store_true', help="Flag to retrain models without any training data.") 
    parser.add_argument("--noBootstrapSOCData", default=False, action='store_true', help="Flag to retrain models without m_SOC data from the validation bootstrap.") 
    parser.add_argument("--noValSpectra",       default=False, action='store_true', help="Flag to retrain models without spectral data from the validation set.") 
    parser.add_argument("--trainDecoder", type=int, default=0, help="Flag to retrain decoder model.") 

    parser.add_argument("--spectraSOCLocation",   type=str, default="data_utils/ICLRDataset_RaCASpectraAndSOC.h5", help="File name for soil spectra and SOC numbers.") 
    parser.add_argument("--splitIndicesLocation", type=str, default="data_utils/ICLRDataset_splitIndices.h5", help="File name for soil spectrum index, split by region number.")      
    parser.add_argument("--endmemberSpectraLocation",   type=str, default="data_utils/ICLRDataset_USGSEndmemberSpectra.h5", help="File name for pure endmember spectra and rhorads.") 
    parser.add_argument("--encoderLocation",      type=str, default="models/testEncoder.h5", help="File location for pretrained encoder model.")
    parser.add_argument("--decoderLocation",      type=str, default="models/testDecoder.h5", help="File location for pretrained decoder model.")

    parser.add_argument("--lr",  type=float, default=0.0001, help="Learning rate for Adam optimizer.")
    parser.add_argument("--b1",  type=float, default=0.99,    help="Beta1 for Adam optimizer.")
    parser.add_argument("--b2",  type=float, default=0.999,  help="Beta2 for Adam optimizer.")

    
    args = parser.parse_args()

    model_choices = {
        "s": "Standard Linear",
        "c1": "1D CNN",
        "c1u": "1D CNN with Uncertainties",
        "r": "RNN",
        "t": "Transformer"
    }

    runName = f"{args.logName}_{args.encoderModel}_{args.crossValidationRegion}_{args.bootstrapIndex}_nD{args.noDecoder}_dR{args.disableRhorads}_dM{args.decoderModel}_ntd{args.noTrainingData}_nvs{args.noValSpectra}_nb{args.noBootstrapSOCData}_td{args.trainDecoder}"

    wandb.init(
        project="ICLR_SOC_Analysis_2024",
        name=runName,
        config={
            "encoderModel": args.encoderModel,
            "crossValidationRegion": args.crossValidationRegion,
            "bootstrapIndex": args.bootstrapIndex,
            "epochs": args.epochs,
            "batch": args.batch,
            "disableRhorads": args.disableRhorads,
            "decoderModel": args.decoderModel,
            "noDecoder": args.noDecoder,
            "noTrainingData": args.noTrainingData,
            "noBootstrapSOCData": args.noBootstrapSOCData,
            "noValSpectra": args.noValSpectra,
            "trainDecoder": args.trainDecoder,
            "spectraSOCLocation": args.spectraSOCLocation,
            "splitIndicesLocation": args.splitIndicesLocation,
            "encoderLocation": args.encoderLocation,
            "decoderLocation": args.decoderLocation,
            "lr": args.lr,
            "b1": args.b1,
            "b2": args.b2
        }
    )
        
    """ ############################################################################################
        Load datasets
    """
    ###
    # Get training data and validation data indices.
    # Load the indices file:
    indices_file = h5py.File(args.splitIndicesLocation, 'r')
    
    # Load bootstrap dataset
    bootstrap_indices = [] if args.noBootstrapSOCData else indices_file[f'{args.crossValidationRegion}_bootstrap_{args.bootstrapIndex}'][:]

    # Get the validation indices for the specified cross-validation region:
    val_indices = indices_file[f'{args.crossValidationRegion}_indices'][:]
    if len(bootstrap_indices) > 0 :
        val_indices = [i for i in val_indices if i not in bootstrap_indices]

    # Train indices should cover the remaining RaCA regions:
    train_indices = None
    for i in range(1,19) :
        if i == 17 or i == args.crossValidationRegion : continue
        train_indices = torch.tensor(indices_file[f'{i}_indices'][:]) if train_indices is None else torch.cat((train_indices,torch.tensor(indices_file[f'{i}_indices'][:])))

    ###
    # Load training and validation datasets and prepare batch loaders:
    dataset = HDF5Dataset(args.spectraSOCLocation)
    training_dataset   = torch.utils.data.Subset(dataset, train_indices)
    validation_dataset = torch.utils.data.Subset(dataset, val_indices)
    bootstrap_dataset  = None if args.noBootstrapSOCData else torch.utils.data.Subset(dataset, bootstrap_indices)

    validation_data_loader  = data.DataLoader(validation_dataset, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    num_batches = len(validation_data_loader)

    train_batch_size     = len(training_dataset) // num_batches
    training_data_loader = data.DataLoader(training_dataset,   batch_size=train_batch_size, shuffle=True, num_workers=0, drop_last=True)

    bootstrap_batch_size  = None if args.noBootstrapSOCData else len(bootstrap_dataset) // num_batches
    bootstrap_data_loader = None if args.noBootstrapSOCData else data.DataLoader(bootstrap_dataset,  batch_size=bootstrap_batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Move relevant datasets to the GPU
    trainingGTmsoc = dataset.dataset[train_indices,-1].to(device)
    trainingGTspec = dataset.dataset[train_indices,:-1].to(device)

    validationGTmsoc = dataset.dataset[list(val_indices) + list(bootstrap_indices),-1].to(device)
    validationGTspec = dataset.dataset[list(val_indices) + list(bootstrap_indices),:-1].to(device)

    """ ############################################################################################
        Prepare models
    """

    ###
    # Load the endmember spectra and rhorads:
    endmember_file  = h5py.File(args.endmemberSpectraLocation, 'r')
    seedFs          = torch.tensor(endmember_file['Fs'][:]).to(device)
    seedrrs         = torch.tensor(endmember_file['rhorads'][:]).to(device)

    # Wavelength axis
    XF = torch.tensor([x for x in range(365,2501)]);

    # If we disable rhorads, set them all to 1 so they have no effect
    if args.disableRhorads : 
        seedrrs = torch.ones(seedrrs.shape).to(device)

    KEndmembers = seedrrs.shape[0]
    MSpectra = seedFs.shape[1]

    # Generate priors for physical parameters and nuisances
    seedrrSOC = torch.mean(seedrrs[:-1])
    seedFsoc = torch.ones((MSpectra)) * 0.5

    seedrrs[-1] = seedrrSOC
    seedFs[-1,:] = seedFsoc

    # Load encoder model from file
    encoder_model = None

    # Set up encoder model and optimizer
    try:
        if args.encoderModel == "s": # Load pretrained LinearMixingEncoder from file
            print("Using Standard Linear Model")
            encoder_model = LinearMixingEncoder(MSpectra, KEndmembers, 512).to(device)
        elif args.encoderModel == "c1":
            print("Using 1D Conv Model")
            encoder_model = EncoderConv1D(MSpectra, KEndmembers, 8, 10).to(device) # (M, K, hidden_size, kernel_size)
        elif args.encoderModel == "r":
            print("Using RNN Model")
            encoder_model = EncoderRNN(MSpectra, KEndmembers, 64).to(device) # (M, K, hidden_size)
        elif args.encoderModel == "t":
            print("Using Transformer Model")
            encoder_model = EncoderTransformer(MSpectra, KEndmembers, 64, 4, 2).to(device) # (M, K, hidden_size, num_heads, num_layers)
        else:
            raise ValueError("Model error, please choose s (standard linear), c1 (1D conv), r (RNN), or t (Transformer)")
        
        encoder_model.load_state_dict(torch.load(args.encoderLocation))
    except ValueError as e:
        print(e)

    # Load decoder model from file.
    decoder_model = None

    if args.noDecoder :
        decoder_model = None
    elif args.decoderModel :
        decoder_model = ANNDecoder(MSpectra, KEndmembers, 512).to(device)
    else :
        decoder_model = LinearMixingDecoder(seedFs, seedrrs).to(device)

    if not args.noDecoder :
        decoder_model.load_state_dict(torch.load(args.decoderLocation))

    # Get the Lagrange factor for this decoder model, to preserve continuity of loss computation
    llf = 1.0 if args.noDecoder or args.decoderModel else decoder_model.computeLagrangeLossFactor().item()

    # Set up optimizer for encoder only
    encoder_optimizer = optim.Adam(list(encoder_model.parameters()) + ([] if args.noDecoder or not args.trainDecoder else list(decoder_model.parameters())) , lr=args.lr, betas=(args.b1, args.b2))

    """ ############################################################################################
        Initial metrics
    """
    # Log predicted vs. measured msoc graph in wandb
    validationEncoderPreds = encoder_model(validationGTspec)

    measVPredTableDat = [[x, y] for (x, y) in zip(validationGTmsoc, validationEncoderPreds[:, -1].detach())]
    measVPredTable = wandb.Table(data=measVPredTableDat, columns=["Measured SOC fraction", "Predicted SOC fraction"])
    wandb.log(
        {
            "measVPred_Initial": wandb.plot.scatter(
                measVPredTable, "Measured SOC fraction", "Predicted SOC fraction", title="Initial Predicted vs. Measured SOC"
            )
        }, step=0
    )

    """ ############################################################################################
        Finetune encoder model using training + (validation - SOC data) + (optional, bootstrap + SOC data)
    """

    # Initialize best validation loss
    best_encoder_lossV = torch.tensor(float("inf"))

    for epoch in tqdm(range(args.epochs)):

        
        # Initialize loss variables for this epoch
        total_encoder_loss = 0.0
        total_decoder_loss = 0.0

        total_decoder_lossV = 0.0
        total_encoder_lossV = 0.0

        # Batching and training
        val_iter = iter(validation_data_loader)
        training_iter = None if args.noTrainingData else iter(training_data_loader)
        bootstrap_iter = None if args.noBootstrapSOCData else iter(bootstrap_data_loader)

        for i in range(num_batches):

            batch_dataV = next(val_iter)

            # Iterate over batches by index in validation_data_loader and bootstrap_data_loader 
            # simultaneously. If bootstrap_data_loader is None, then this loop will only iterate
            # over the validation_data_loader.
            batch_dataT = None if args.noTrainingData else next(training_iter)
            batch_dataB = None if args.noBootstrapSOCData else next(bootstrap_iter)

            # Extract batch data
            batch_tIsV, batch_tmsocV = batch_dataV[:,:-1].to(device), batch_dataV[:,-1].to(device)
            
            batch_tIsT   = None if args.noTrainingData else batch_dataT[:,:-1].to(device)
            batch_tmsocT = None if args.noTrainingData else batch_dataT[:,-1].to(device)

            batch_tIsB   = None if args.noBootstrapSOCData else batch_dataB[:,:-1].to(device)
            batch_tmsocB = None if args.noBootstrapSOCData else batch_dataB[:,-1].to(device)

            batchSizeT = 0 if args.noTrainingData else batch_tIsT.shape[0]
            batchSizeB = 0 if args.noBootstrapSOCData else batch_tIsB.shape[0]
            batchSizeV = batch_tIsV.shape[0]

            trainingEncoderLossNum = 0
            if not args.noBootstrapSOCData :
                trainingEncoderLossNum += batchSizeB
            if not args.noTrainingData :
                trainingEncoderLossNum += batchSizeT
            
            trainingDecoderLossNum = 0
            if not args.noValSpectra :
                trainingDecoderLossNum += batchSizeV
            if not args.noTrainingData :
                trainingDecoderLossNum += batchSizeT
            if not args.noBootstrapSOCData :
                trainingDecoderLossNum += batchSizeB

            validationEncoderLossNum = batchSizeV + batchSizeB
            
            validationDecoderLossNum = batchSizeV + batchSizeB
            

            # Get abundance predictions from the encoder for the batch
            encoderPredsV = encoder_model(batch_tIsV)
            encoderPredsT = None if args.noTrainingData else encoder_model(batch_tIsT)
            encoderPredsB = None if args.noBootstrapSOCData else encoder_model(batch_tIsB)

            # Get spectrum predictions from the decoder for the batch
            decoderPredsV, decoderPredsT, decoderPredsB = None, None, None
            if not args.noDecoder :
                decoderPredsV = decoder_model(encoderPredsV)
                decoderPredsT = None if args.noTrainingData else decoder_model(encoderPredsT)
                decoderPredsB = None if args.noBootstrapSOCData else decoder_model(encoderPredsB)

            # Compute encoder loss: sqerr from true Msoc values for the batch
            encoder_loss = 0.0
            encoder_loss = 0.0 if args.noTrainingData else torch.sum((encoderPredsT[:, -1] - batch_tmsocT[:]) ** 2)
            encoder_loss += 0.0 if args.noBootstrapSOCData else torch.sum((encoderPredsB[:, -1] - batch_tmsocB[:]) ** 2)

            # Calculate validation loss
            encoder_lossV = torch.sum((encoderPredsV[:, -1] - batch_tmsocV[:]) ** 2)
            encoder_lossV += 0.0 if args.noBootstrapSOCData else torch.sum((encoderPredsB[:, -1] - batch_tmsocB[:]) ** 2)

            decoder_lossV = 0.0
            decoder_loss = 0.0
            if not args.noDecoder :
                decoder_lossV = torch.mean((decoderPredsV - batch_tIsV) ** 2)*batchSizeV
                decoder_lossV += 0.0 if args.noBootstrapSOCData else torch.mean((decoderPredsB - batch_tIsB) ** 2)*batchSizeB

                # Add decoder loss from training or bootstrap data if applicable
                decoder_loss = 0.0 if args.noValSpectra else decoder_lossV
                decoder_loss += 0.0 if args.noTrainingData else torch.mean((decoderPredsT - batch_tIsT) ** 2)*batchSizeT

                if args.trainDecoder: 
                    llf = 1.0 if args.decoderModel else decoder_model.computeLagrangeLossFactor()

            encoder_loss = encoder_loss*0. if trainingEncoderLossNum == 0 else encoder_loss/trainingEncoderLossNum
            decoder_loss = decoder_loss/trainingDecoderLossNum

            encoder_lossV = encoder_lossV/validationEncoderLossNum
            decoder_lossV = decoder_lossV/validationDecoderLossNum

            # Calculate the combined loss
            loss = (encoder_loss/(0.0041**2) + decoder_loss/(0.01**2)) * llf

            # Backpropagate the gradients for both models
            encoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()

            # Accumulate batch losses
            if not (args.noTrainingData and args.noBootstrapSOCData) :
                total_encoder_loss += encoder_loss.item()

            total_decoder_loss += decoder_loss if args.noDecoder or (args.noValSpectra and args.noTrainingData) else decoder_loss.item()

            total_encoder_lossV += encoder_lossV.item()
            total_decoder_lossV += decoder_loss if args.noDecoder else decoder_lossV.item()


        # Calculate the average loss for this epoch
        avg_encoder_loss = total_encoder_loss / len(validation_data_loader)
        avg_decoder_loss = total_decoder_loss / len(validation_data_loader)

        avg_encoder_lossV = total_encoder_lossV / len(validation_data_loader)
        avg_decoder_lossV = total_decoder_lossV / len(validation_data_loader)

        wandb.log({"Encoder_FreezeDecoderFinetune_Loss": avg_encoder_loss,
                   "Decoder_FreezeDecoderFinetune_Loss": avg_decoder_loss,
                   "Total_FreezeDecoderFinetune_Loss": avg_encoder_loss/(0.0041**2) + avg_decoder_loss/(0.01**2),
                   "Encoder_FreezeDecoderFinetuneValidation_Loss": avg_encoder_lossV, 
                   "Decoder_FreezeDecoderFinetuneValidation_Loss": avg_decoder_lossV,
                   "Total_FreezeDecoderFinetuneValidation_Loss": avg_encoder_lossV/(0.0041**2) + avg_decoder_lossV/(0.01**2)}, step=epoch)
    
        if epoch % 10 == 0 or epoch == args.epochs - 1:
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
            trainingR2 = r2_score(trainingGTmsoc, trainingEncoderPreds[:, -1].detach())
            trainingBias = torch.mean(trainingEncoderPreds[:, -1] - trainingGTmsoc)
            trainingRPD = torch.std(trainingEncoderPreds[:, -1]) / torch.std(trainingGTmsoc)

            trainingDecoderRMSEP = 0.0 if args.noDecoder else torch.sqrt(torch.mean((trainingDecoderPreds - trainingGTspec) ** 2))
            trainingDecoderMRMSE = 0.0 if args.noDecoder else torch.mean(torch.sqrt(torch.mean((trainingDecoderPreds - trainingGTspec) ** 2,axis=1)))

            # Log metrics in wandb
            wandb.log({"Encoder_FreezeDecoderFinetuneTraining_RMSEP": trainingRMSEP,
                        "Encoder_FreezeDecoderFinetuneTraining_R2": trainingR2,
                        "Encoder_FreezeDecoderFinetuneTraining_Bias": trainingBias,
                        "Encoder_FreezeDecoderFinetuneTraining_RPD": trainingRPD,
                        "Decoder_FreezeDecoderFinetuneTraining_RMSEP": trainingDecoderRMSEP,
                        "Decoder_FreezeDecoderFinetuneTraining_MRMSE": trainingDecoderMRMSE
                        }, step=epoch)

            # Compute metrics for validation set
            validationEncoderPreds = encoder_model(validationGTspec)
            validationDecoderPreds = None if args.noDecoder else decoder_model(validationEncoderPreds)

            validationRMSEP = torch.sqrt(torch.mean((validationEncoderPreds[:, -1] - validationGTmsoc) ** 2))
            validationR2 = r2_score(validationGTmsoc, validationEncoderPreds[:, -1].detach())
            validationBias = torch.mean(validationEncoderPreds[:, -1] - validationGTmsoc)
            validationRPD = torch.std(validationEncoderPreds[:, -1]) / torch.std(validationGTmsoc)

            validationDecoderRMSEP = 0.0 if args.noDecoder else torch.sqrt(torch.mean((validationDecoderPreds - validationGTspec) ** 2))
            validationDecoderMRMSE = 0.0 if args.noDecoder else torch.mean(torch.sqrt(torch.mean((validationDecoderPreds - validationGTspec) ** 2,axis=1)))

            wandb.log({"Encoder_FreezeDecoderFinetuneValidation_RMSEP": validationRMSEP,
                    "Encoder_FreezeDecoderFinetuneValidation_R2": validationR2,
                    "Encoder_FreezeDecoderFinetuneValidation_Bias": validationBias,
                    "Encoder_FreezeDecoderFinetuneValidation_RPD": validationRPD,
                    "Decoder_FreezeDecoderFinetuneValidation_RMSEP": validationDecoderRMSEP,
                    "Decoder_FreezeDecoderFinetuneValidation_MRMSE": validationDecoderMRMSE
                    }, step=epoch)

            if not args.noDecoder :
                if args.trainDecoder and not args.decoderModel :
                    wandb.log({"rrSOC_finetuned": decoder_model.rrsoc.detach().item()},step=epoch)

                    # Log fsoc graph in wandb
                    tfsoc = decoder_model.fsoc.detach()
                    fsocTableDat = [[x, y] for (x, y) in zip(XF,tfsoc)]
                    fsocTable = wandb.Table(data=fsocTableDat, columns=["Wavelength", "SOC Reflectance"])
                    wandb.log(
                        {
                            "Fsoc_finetuned": wandb.plot.line(
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

    """ ############################################################################################
        Summarial metrics
    """
    # Log predicted vs. measured msoc graph in wandb
    validationEncoderPreds = encoder_model(validationGTspec)

    measVPredTableDat = [[x, y] for (x, y) in zip(validationGTmsoc, validationEncoderPreds[:, -1].detach())]
    measVPredTable = wandb.Table(data=measVPredTableDat, columns=["Measured SOC fraction", "Predicted SOC fraction"])
    wandb.log(
        {
            "measVPred_Final": wandb.plot.scatter(
                measVPredTable, "Measured SOC fraction", "Predicted SOC fraction", title="Final Predicted vs. Measured SOC"
            )
        }, step=args.epochs
    )

    """ ############################################################################################
        Cleanup
    """
    torch.save(encoder_model.state_dict(), f"models/{runName}_encoderFinetuned.pt")

    # Close the HDF5 files
    indices_file.close()
    endmember_file.close()

    # Finish wandb run
    wandb.finish()