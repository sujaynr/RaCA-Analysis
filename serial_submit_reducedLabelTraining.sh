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
export nFineTuneEpochs=100
export bootstrapIndex=2
export spectraSOCLocation="${currWorkingDir}/data_utils/ICLRDataset_RaCASpectraAndSOC_v2.h5"
export splitIndicesLocation="${currWorkingDir}/data_utils/ICLRDataset_SplitIndices_v2.h5"
export endmemberSpectraLocation="${currWorkingDir}/data_utils/ICLRDataset_USGSEndmemberSpectra.h5"
export basename=$1


#########################################################################################
# Submit jobs for train-val split over all regions
# Loop over all integers from 1 to 18
for regionNumber in {1..18}
do 

    for modelType in "$2"
    do
        echo "\t\t - Running model $modelType with no decoder"
        python reducedLabelTraining.py --encoderModel $modelType \
                                        --crossValidationRegion $regionNumber \
                                        --bootstrapIndex $bootstrapIndex \
                                        --epochs $nEpochs \
                                        --batch $batchSize \
                                        --spectraSOCLocation $spectraSOCLocation \
                                        --splitIndicesLocation $splitIndicesLocation \
                                        --endmemberSpectraLocation $endmemberSpectraLocation \
                                        --logName $basename \
                                        --finetuneEpochs $nFineTuneEpochs

    done

    exit
done