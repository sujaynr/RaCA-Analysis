#!/bin/bash

#########################################################################################
# Bash script for local serial run of all jobs on one machine.
# Remember to make script executable by user via `chmod u+x serial_submit.sh`.
#########################################################################################

export WANDB_MODE=online
export currWorkingDir=$(pwd)

# Set global variables
export nEpochs=1000
export batchSize=300
export spectraSOCLocation="${currWorkingDir}/data_utils/ICLRDataset_RaCASpectraAndSOC_v2.h5"
export splitIndicesLocation="${currWorkingDir}/data_utils/ICLRDataset_SplitIndices_v2.h5"
export endmemberSpectraLocation="${currWorkingDir}/data_utils/ICLRDataset_USGSEndmemberSpectra.h5"
export basename=$1


#########################################################################################
# Submit jobs for train-val split over all regions
for modelType in "$2" # "s" "c1" "r"
do
    echo "\t\t - Running model $modelType with no decoder"
    python updatedTrain.py --encoderModel $modelType \
                            --noDecoder \
                            --fullFit \
                            --fixRandomSeed \
                            --trainValSplit 0.4 \
                            --epochs $nEpochs \
                            --batch $batchSize \
                            --spectraSOCLocation $spectraSOCLocation \
                            --splitIndicesLocation $splitIndicesLocation \
                            --endmemberSpectraLocation $endmemberSpectraLocation \
                            --logName $basename 


    echo "\t\t - Running model $modelType with physical model"
    python updatedTrain.py --encoderModel $modelType \
                            --fullFit \
                            --fixRandomSeed \
                            --trainValSplit 0.4 \
                            --epochs $nEpochs \
                            --batch $batchSize \
                            --spectraSOCLocation $spectraSOCLocation \
                            --splitIndicesLocation $splitIndicesLocation \
                            --endmemberSpectraLocation $endmemberSpectraLocation \
                            --logName $basename


    echo "\t\t - Running model $modelType with ANN decoder"
    python updatedTrain.py --encoderModel $modelType \
                            --decoderModel \
                            --fullFit \
                            --fixRandomSeed \
                            --trainValSplit 0.4 \
                            --epochs $nEpochs \
                            --batch $batchSize \
                            --spectraSOCLocation $spectraSOCLocation \
                            --splitIndicesLocation $splitIndicesLocation \
                            --endmemberSpectraLocation $endmemberSpectraLocation \
                            --logName $basename
done