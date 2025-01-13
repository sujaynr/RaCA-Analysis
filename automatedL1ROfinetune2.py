import subprocess
import random

# Define parameter values
learning_rates = [0.00005]
hidden_layer_sizes = [1517]
batch_sizes = [75]
cross_validation_regions = [1,2,3,4,5,6,7,8,9,10,11, 12, 13, 14, 15, 16, 18]
random_seeds = [192021, 42, 123, 456]  # Generate 5 random seeds
models = ['--decoderModel', '--noDecoder']  # Include None for the LMM decoder case

# Script path and base arguments
script_path = '/home/sujaynair/RaCA-Analysis/updatedTrain.py'
common_args_base = ['python3', script_path, '--trainValSplit', '-0.4', '--encoderModel', 's', '--fullFit', '--epochs', '500', '--allScans', '--finetuneEpochs', '1000']

# Loop through all combinations of parameters and configurations
for lr in learning_rates:
    for hl in hidden_layer_sizes:
        for bs in batch_sizes:
            for cvr in cross_validation_regions:
                for seed in random_seeds:
                    for model in models:
                        # Determine the model-specific command
                        model_args = []
                        if model is not None:
                            model_args.append(model)
                        
                        # Modify the log name based on the model
                        if model == '--decoderModel':
                            model_suffix = "ann"
                        elif model == '--noDecoder':
                            model_suffix = "ND"
                        else:
                            model_suffix = "lmm"  # Default LMM case
                            
                        log_name = f"CROSSVAL_10FINETUNE_{model_suffix}_lr{lr}_hl{hl}_bs{bs}_cvr{cvr}_seed{seed}"
                        
                        # Construct the command
                        command = common_args_base + model_args + [
                            '--batch', str(bs), '--hl', str(hl), '--lr', str(lr),
                            '--crossValidationRegion', str(cvr), '--setRandomSeedTo', str(seed),
                            '--logName', log_name
                        ]
                        
                        # Run the subprocess
                        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        
                        # Output the process details and results
                        print(f"Running with lr={lr}, hl={hl}, bs={bs}, cvr={cvr}, seed={seed}, model={model_suffix}, logName={log_name}")
                        print(f"Standard Output: {result.stdout}")
                        if result.stderr:
                            print(f"Standard Error: {result.stderr}")
                        
                        if result.returncode == 0:
                            print(f"Execution succeeded for lr={lr}, hl={hl}, bs={bs}, cvr={cvr}, seed={seed}, model={model_suffix}")
                        else:
                            print(f"Execution failed for lr={lr}, hl={hl}, bs={bs}, cvr={cvr}, seed={seed}, model={model_suffix}")
