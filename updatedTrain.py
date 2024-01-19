# Standard libraries
import pdb
import time
import argparse

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
            self.dataset = torch.tensor(self.file_mag['data'][:]).to(device)
        else :
            self.dataset = torch.tensor(self.file_mag['data'][:])

    def __len__(self):
        return len(self.file_mag.keys())

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

    parser.add_argument("--spectraSOCLocation",         type=str, default="data_utils/ICLRDataset_RaCASpectraAndSOC.h5", help="File name for soil spectra and SOC numbers.") 
    parser.add_argument("--splitIndicesLocation",       type=str, default="data_utils/ICLRDataset_splitIndices.h5", help="File name for soil spectrum index, split by region number.") 
    parser.add_argument("--endmemberSpectraLocation",   type=str, default="data_utils/ICLRDataset_USGSEndmemberSpectra.h5", help="File name for pure endmember spectra and rhorads.") 

    parser.add_argument("--lr",  type=float, default=0.00001, help="Learning rate for Adam optimizer.")
    parser.add_argument("--b1",  type=float, default=0.99,    help="Beta1 for Adam optimizer.")
    parser.add_argument("--b2",  type=float, default=0.999,  help="Beta2 for Adam optimizer.")

    parser.add_argument("--finetuneEpochs",     type=int, default=10000,       help="Number of training epochs for the fine-tuning step.")

    
    args = parser.parse_args()

    model_choices = {
        "s": "Standard Linear",
        "c1": "1D CNN",
        "c1u": "1D CNN with Uncertainties",
        "r": "RNN",
        "t": "Transformer"
    }

    runName = f"{args.logName}_{args.encoderModel}_{args.crossValidationRegion}_{args.bootstrapIndex}_nD{args.noDecoder}_dR{args.disableRhorads}_dM{args.decoderModel}_ff{args.fullFit}"

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

    # Get the validation indices for the specified cross-validation region:
    val_indices = indices_file[f'{args.crossValidationRegion}_indices'][:] if not args.fullFit else None

    # Train indices should cover the remaining RaCA regions:
    train_indices = None
    for i in range(1,19) :
        if i == 17 or i == args.crossValidationRegion : continue
        train_indices = torch.tensor(indices_file[f'{i}_indices'][:]) if train_indices is None else torch.cat((train_indices,torch.tensor(indices_file[f'{i}_indices'][:])))

    ###
    # Load training and validation datasets and prepare batch loaders:
    training_dataset   = torch.utils.data.Subset(dataset, train_indices)
    validation_dataset = torch.utils.data.Subset(dataset, val_indices) if not args.fullFit else None

    training_data_loader    = data.DataLoader(training_dataset,   batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    validation_data_loader  = data.DataLoader(validation_dataset, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True) if not args.fullFit else None

    # Load bootstrap dataset
    bootstrap_indices = None if args.fullFit else indices_file[f'{args.crossValidationRegion}_bootstrap_{args.bootstrapIndex}'][:]
    bootstrap_dataset = None if args.fullFit else torch.utils.data.Subset(dataset, bootstrap_indices)
    finetune_data_loader  = None if args.fullFit else data.DataLoader(bootstrap_dataset, batch_size=len(bootstrap_dataset), shuffle=True, num_workers=0, drop_last=True)

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

    validationGTmsoc = None if args.fullFit else dataset.dataset[val_indices,-1].to(device)
    validationGTspec = None if args.fullFit else dataset.dataset[val_indices,:-1].to(device)

    finetuneGTmsoc = None if args.fullFit else dataset.dataset[bootstrap_indices,-1].to(device)
    finetuneGTspec = None if args.fullFit else dataset.dataset[bootstrap_indices,:-1].to(device)

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
            encoder_model = EncoderConv1D(MSpectra, KEndmembers, 32, 15).to(device) # (M, K, hidden_size, kernel_size)
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

    for epoch in tqdm(range(args.epochs+1)):

        # Initialize loss variables for this epoch
        total_encoder_loss = 0.0
        total_decoder_loss = 0.0
        maxllf = 0.0

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

            # Backpropagate the gradients for both models
            combined_optimizer.zero_grad()
            loss.backward()
            combined_optimizer.step()

            # Accumulate batch losses
            total_encoder_loss += encoder_loss.item()
            total_decoder_loss += decoder_loss if args.noDecoder else decoder_loss.item()

        # Calculate the average loss for this epoch
        avg_encoder_loss = total_encoder_loss / len(training_data_loader)
        avg_decoder_loss = total_decoder_loss / len(training_data_loader)

        wandb.log({"Encoder_Training_Loss": avg_encoder_loss, 
                   "Decoder_Training_Loss": avg_decoder_loss, 
                   "Total_Training_Loss": avg_encoder_loss/(0.0041**2) + avg_decoder_loss/(0.01**2),
                   "Max_LagrangeLossFactor": maxllf})

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
                           "Total_Validation_Loss": avg_encoder_lossV/(0.0041**2) + avg_decoder_lossV/(0.01**2)})
        
            if epoch % 100 == 0:
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

                # Log metrics in wandb
                wandb.log({"Encoder_Training_RMSEP": trainingRMSEP,
                            "Encoder_Training_R2": trainingR2,
                            "Encoder_Training_Bias": trainingBias,
                            "Encoder_Training_RPD": trainingRPD,
                            "Decoder_Training_RMSEP": trainingDecoderRMSEP})

                # Compute metrics for validation set
                if not args.fullFit:
                    # Get preds on full val set
                    validationEncoderPreds = encoder_model(validationGTspec)
                    validationDecoderPreds = None if args.noDecoder else decoder_model(validationEncoderPreds)

                    validationRMSEP = torch.sqrt(torch.mean((validationEncoderPreds[:, -1] - validationGTmsoc) ** 2))
                    validationR2 = r2_score(validationGTmsoc, validationEncoderPreds[:, -1])
                    validationBias = torch.mean(validationEncoderPreds[:, -1] - validationGTmsoc)
                    validationRPD = torch.std(validationEncoderPreds[:, -1]) / torch.std(validationGTmsoc)

                    validationDecoderRMSEP = 0 if args.noDecoder else torch.sqrt(torch.mean((validationDecoderPreds - validationGTspec) ** 2))

                    wandb.log({"Encoder_Validation_RMSEP": validationRMSEP,
                            "Encoder_Validation_R2": validationR2,
                            "Encoder_Validation_Bias": validationBias,
                            "Encoder_Validation_RPD": validationRPD,
                            "Decoder_Validation_RMSEP": validationDecoderRMSEP})
                
                # Log decoder model parameters in wandb
                if not args.noDecoder and not args.decoderModel :
                    wandb.log({"rrSOC": decoder_model.rrsoc.detach().item()})

                    # Log fsoc graph in wandb
                    tfsoc = decoder_model.fsoc.detach()
                    fsocTableDat = [[x, y] for (x, y) in zip(XF,tfsoc)]
                    fsocTable = wandb.Table(data=fsocTableDat, columns=["Wavelength", "SOC Reflectance"])
                    wandb.log(
                        {
                            "Fsoc": wandb.plot.line(
                                fsocTable, "Wavelength", "SOC Reflectance", title="Regressed SOC Spectrum"
                            )
                        }
                    )

                # Save the model if it is the best so far
                if epoch % 1000 == 0 and not args.fullFit and avg_encoder_lossV < best_encoder_lossV and epoch < args.epochs - 1:
                    
                    best_encoder_lossV = avg_encoder_lossV

                    torch.save(encoder_model.state_dict(), f"models/{runName}_encoder_minValMSEP.pt")

                    if not args.noDecoder:
                        torch.save(decoder_model.state_dict(), f"models/{runName}_decoder_minValMSEP.pt")

        
    torch.save(encoder_model.state_dict(), f"models/{runName}_encoder_final.pt")

    if not args.noDecoder:
        torch.save(decoder_model.state_dict(), f"models/{runName}_decoder_final.pt")


    """ ############################################################################################
        Fine-tune models (TODO)
    """
    if not args.fullFit :
        # # Load the best encoder model
        # encoder_model.load_state_dict(torch.load(f"models/{runName}_encoder_minValMSEP.pt"))

        # # Load the best decoder model
        # if not args.noDecoder:
        #     decoder_model.load_state_dict(torch.load(f"models/{runName}_decoder_minValMSEP.pt"))

        # Set up optimizer
        combined_optimizer = optim.Adam(list(encoder_model.parameters()) + list([] if args.noDecoder else decoder_model.parameters()), lr=args.lr, betas=(args.b1, args.b2))

        # Initialize best validation loss
        best_encoder_lossV = torch.tensor(float("inf"))

        for epoch in tqdm(range(args.finetuneEpochs+1)):

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
                       "Max_Finetune_LagrangeLossFactor": maxllf})
    
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
                          "Total_FinetuneValidation_Loss": avg_encoder_lossV/(0.0041**2) + avg_decoder_lossV/(0.01**2)})
            
                if epoch % 100 == 0:
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

                    # Log metrics in wandb
                    wandb.log({"Encoder_Finetune_RMSEP": finetuneRMSEP,
                                "Encoder_Finetune_R2": finetuneR2,
                                "Encoder_Finetune_Bias": finetuneBias,
                                "Encoder_Finetune_RPD": finetuneRPD,
                                "Decoder_Finetune_RMSEP": finetuneDecoderRMSEP})

                    # Get preds on full val set
                    validationEncoderPreds = encoder_model(validationGTspec)
                    validationDecoderPreds = None if args.noDecoder else decoder_model(validationEncoderPreds)

                    validationRMSEP = torch.sqrt(torch.mean((validationEncoderPreds[:, -1] - validationGTmsoc) ** 2))
                    validationR2 = r2_score(validationGTmsoc, validationEncoderPreds[:, -1])
                    validationBias = torch.mean(validationEncoderPreds[:, -1] - validationGTmsoc)
                    validationRPD = torch.std(validationEncoderPreds[:, -1]) / torch.std(validationGTmsoc)

                    validationDecoderRMSEP = 0 if args.noDecoder else torch.sqrt(torch.mean((validationDecoderPreds - validationGTspec) ** 2))

                    wandb.log({"Encoder_FinetuneValidation_RMSEP": validationRMSEP,
                            "Encoder_FinetuneValidation_R2": validationR2,
                            "Encoder_FinetuneValidation_Bias": validationBias,
                            "Encoder_FinetuneValidation_RPD": validationRPD,
                            "Decoder_FinetuneValidation_RMSEP": validationDecoderRMSEP})
                    
                    # Log decoder model parameters in wandb
                    if not args.noDecoder and not args.decoderModel :
                        wandb.log({"rrSOC_finetuned": decoder_model.rrsoc.detach().item()})

                        # Log fsoc graph in wandb
                        tfsoc = decoder_model.fsoc.detach()
                        fsocTableDat = [[x, y] for (x, y) in zip(XF,tfsoc)]
                        fsocTable = wandb.Table(data=fsocTableDat, columns=["Wavelength", "SOC Reflectance"])
                        wandb.log(
                            {
                                "Fsoc_finetuned": wandb.plot.line(
                                    fsocTable, "Wavelength", "SOC Reflectance", title="Regressed SOC Spectrum"
                                )
                            }
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