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
export nEpochs=150
export batchSize=300
export bootstrapIndex=2
export spectraSOCLocation="${currWorkingDir}/data_utils/ICLRDataset_RaCASpectraAndSOC_v2.h5"
export splitIndicesLocation="${currWorkingDir}/data_utils/ICLRDataset_SplitIndices_v2.h5"
export endmemberSpectraLocation="${currWorkingDir}/data_utils/ICLRDataset_USGSEndmemberSpectra.h5"

export basename=$1


#########################################################################################

# Loop over all integers from 1 to 18
for regionNumber in {1..18}
do 
    for dM in {0..2}
    do
        for tD in 1
        do

            export dMText="False"
            export nDText="True"
            export nD=1

            if [ $dM -eq 1 ]
            then
                export dMText="True"
                export nDText="False"
                export nD=0

            fi

            if [ $dM -eq 2 ]
            then
                export dMText="False"
                export nDText="False"
                export nD=0
                export dM=0
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
            export encoderLoc="./models/final_s_${regionNumber}_${bootstrapIndex}_nD${nDText}_dRFalse_dM${dMText}_ffFalse_encoder_final.pt"
            export decoderLoc="./models/final_s_${regionNumber}_${bootstrapIndex}_nD${nDText}_dRFalse_dM${dMText}_ffFalse_decoder_final.pt"

            echo " - Finetuning for region $regionNumber, with no training data, only val spectra + bootstrap all"
            python freezeDecoderFinetuneExperiment.py \
                                    --encoderModel s \
                                    --trainDecoder ${tD} \
                                    --decoderModel ${dM} \
                                    --noDecoder ${nD} \
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

            echo " - Finetuning for region $regionNumber, with no training or bootstrap data, only val spectra"
            python freezeDecoderFinetuneExperiment.py \
                                    --encoderModel s \
                                    --trainDecoder ${tD} \
                                    --decoderModel ${dM} \
                                    --noDecoder ${nD} \
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

            # ########################################################################################

            echo " - Finetuning for region $regionNumber, with no training data or val spectra, only bootstrap data"
            python freezeDecoderFinetuneExperiment.py \
                                    --encoderModel s \
                                    --trainDecoder ${tD} \
                                    --decoderModel ${dM} \
                                    --noDecoder ${nD} \
                                    --noTrainingData \
                                    --noValSpectra \
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
        done
    done
done