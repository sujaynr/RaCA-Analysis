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
export modelType="s"

#########################################################################################
# Submit jobs for train-val split over all regions

for randomSeed in 100390439 23039 4054950 534
do
    echo "\t\t - Running model $modelType with no decoder"
    python updatedTrain.py --encoderModel $modelType \
                            --noDecoder \
                            --fullFit \
                            --fixRandomSeed \
                            --setRandomSeedTo $randomSeed \
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
                            --setRandomSeedTo $randomSeed \
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
                            --setRandomSeedTo $randomSeed \
                            --trainValSplit 0.4 \
                            --epochs $nEpochs \
                            --batch $batchSize \
                            --spectraSOCLocation $spectraSOCLocation \
                            --splitIndicesLocation $splitIndicesLocation \
                            --endmemberSpectraLocation $endmemberSpectraLocation \
                            --logName $basename
done