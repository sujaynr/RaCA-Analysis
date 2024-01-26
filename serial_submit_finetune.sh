#!/bin/bash

#########################################################################################
# Bash script for local serial run of all jobs on one machine.
# Remember to make script executable by user via `chmod u+x serial_submit.sh`.
#########################################################################################

# Loading modules
#conda activate base

export WANDB_MODE=online
export currWorkingDir=$(pwd)

# Set global variables
export nEpochs=100
export batchSize=150
export bootstrapIndex=2
export spectraSOCLocation="${currWorkingDir}/data_utils/ICLRDataset_RaCASpectraAndSOC_v2.h5"
export splitIndicesLocation="${currWorkingDir}/data_utils/ICLRDataset_SplitIndices_v2.h5"
export endmemberSpectraLocation="${currWorkingDir}/data_utils/ICLRDataset_USGSEndmemberSpectra.h5"

export basename=$1


#########################################################################################

# Loop over all integers from 1 to 18
for regionNumber in 4 6
do 
    for dM in {0..1}
    do
        for tD in {0..1}
        do

            export dMText="False"
            if [ $dM -eq 1 ]
            then
                export dMText="True"
            fi

            #########################################################################################
            # Skip job 17, as there is no RaCA region associated to it
            if [ $regionNumber -eq 17 ]
            then
                continue
            fi
            
            echo "Running region $regionNumber with tD=$tD, dM=$dM"
            
            #########################################################################################
            # Submit jobs for end to end prediction with decoder and physical modeling
            export encoderLoc="./models/glasswing_s_${regionNumber}_${bootstrapIndex}_nDFalse_dRFalse_dM${dMText}_ffFalse_encoder_final.pt"
            export decoderLoc="./models/glasswing_s_${regionNumber}_${bootstrapIndex}_nDFalse_dRFalse_dM${dMText}_ffFalse_decoder_final.pt"

            echo " - Finetuning for region $regionNumber, with no training data"
            python freezeDecoderFinetuneExperiment.py \
                                    --encoderModel s \
                                    --trainDecoder ${tD} \
                                    --decoderModel ${dM} \
                                    --noTrainingData \
                                    --crossValidationRegion $regionNumber \
                                    --bootstrapIndex $bootstrapIndex \
                                    --epochs $nEpochs \
                                    --batch $batchSize \
                                    --spectraSOCLocation $spectraSOCLocation \
                                    --splitIndicesLocation $splitIndicesLocation \
                                    --endmemberSpectraLocation $endmemberSpectraLocation \
                                    --encoderLocation $encoderLoc \
                                    --decoderLocation $decoderLoc \
                                    --logName $basename

            #########################################################################################

            echo " - Finetuning for region $regionNumber, with no training or bootstrap data"
            python freezeDecoderFinetuneExperiment.py \
                                    --encoderModel s \
                                    --trainDecoder ${tD} \
                                    --decoderModel ${dM} \
                                    --noTrainingData \
                                    --noBootstrapSOCData \
                                    --crossValidationRegion $regionNumber \
                                    --bootstrapIndex $bootstrapIndex \
                                    --epochs $nEpochs \
                                    --batch $batchSize \
                                    --spectraSOCLocation $spectraSOCLocation \
                                    --splitIndicesLocation $splitIndicesLocation \
                                    --endmemberSpectraLocation $endmemberSpectraLocation \
                                    --encoderLocation $encoderLoc \
                                    --decoderLocation $decoderLoc \
                                    --logName $basename

            #########################################################################################

            echo " - Finetuning for region $regionNumber, with training and bootstrap data"
            python freezeDecoderFinetuneExperiment.py \
                                    --encoderModel s \
                                    --trainDecoder ${tD} \
                                    --decoderModel ${dM} \
                                    --crossValidationRegion $regionNumber \
                                    --bootstrapIndex $bootstrapIndex \
                                    --epochs $nEpochs \
                                    --batch $batchSize \
                                    --spectraSOCLocation $spectraSOCLocation \
                                    --splitIndicesLocation $splitIndicesLocation \
                                    --endmemberSpectraLocation $endmemberSpectraLocation \
                                    --encoderLocation $encoderLoc \
                                    --decoderLocation $decoderLoc \
                                    --logName $basename


            echo "\t\t - Running model with normal settings"
        done
    done
done