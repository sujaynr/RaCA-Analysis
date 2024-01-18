#!/bin/bash

#########################################################################################
# Bash script for local serial run of all jobs on one machine.
# Remember to make script executable by user via `chmod u+x serial_submit.sh`.
#########################################################################################

# Loading modules
#conda activate base

export WANDB_MODE=offline
export currWorkingDir=$(pwd)

# Set global variables
export nEpochs=3
export batchSize=75
export nFineTuneEpochs=5
export bootstrapIndex=1
export spectraSOCLocation="${currWorkingDir}/data_utils/ICLRDataset_RaCASpectraAndSOC_v2.h5"
export splitIndicesLocation="${currWorkingDir}/data_utils/ICLRDataset_SplitIndices_v2.h5"
export endmemberSpectraLocation="${currWorkingDir}/data_utils/ICLRDataset_USGSEndmemberSpectra.h5"
export basename=$1

#########################################################################################

# Loop over all integers from 1 to 18
for regionNumber in {1..18}
do 
    #########################################################################################
    # Skip job 17, as there is no RaCA region associated to it
    if [ $regionNumber -eq 17 ]
    then
        continue
    fi

    echo "Running region $regionNumber"

    #########################################################################################
    # Submit jobs for end to end prediction with no decoder
    for modelType in "s" "c1" "r"
    do
        echo "\t\t - Running model $modelType with no decoder"
        python updatedTrain.py --encoderModel $modelType \
                               --noDecoder \
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

    #########################################################################################
    # Submit jobs for end to end prediction with decoder and physical modeling
    for modelType in "s" "c1" "r"
    do
        echo "\t\t - Running model $modelType with normal settings"
        python updatedTrain.py --encoderModel $modelType \
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

    #########################################################################################
    # Submit jobs for end to end prediction with decoder and no physical modeling
    for modelType in "s" "c1" "r"
    do
        echo "\t\t - Running model $modelType with no rhorads"
        python updatedTrain.py --encoderModel $modelType \
                               --disableRhorads \
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

    #########################################################################################   
    # Submit jobs for end to end encoder and ANN decoder
    for modelType in "s" # "c1" "r"
    do
        echo "\t\t - Running model $modelType with ANN decoder"
        python updatedTrain.py --encoderModel $modelType \
                               --decoderModel \
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
done

#########################################################################################
# Perform full analysis with no validation set to find complete SOC spectrum    
for modelType in "s" "c1" "r"
do
    echo "\t\t - Running full fit for model $modelType"
    python updatedTrain.py --encoderModel $modelType \
                            --fullFit \
                            --epochs $nEpochs \
                            --batch $batchSize \
                            --spectraSOCLocation $spectraSOCLocation \
                            --splitIndicesLocation $splitIndicesLocation \
                            --endmemberSpectraLocation $endmemberSpectraLocation \
                            --logName $basename
done
