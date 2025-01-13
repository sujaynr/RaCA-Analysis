import subprocess

learning_rates = [0.00005]
hidden_layer_sizes = [1517]
batch_sizes = [75]
seeds = [42, 123]  # Example seeds
scanSplits = [5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50]

script_path = '/home/sujaynair/RaCA-Analysis/updatedTrain.py'
common_args_base = ['python3', script_path, '--trainValSplit', '0.4', '--encoderModel', 's', '--fullFit', '--epochs', '500', '--decoderModel']

for lr in learning_rates:
    for hl in hidden_layer_sizes:
        for bs in batch_sizes:
            for seed in seeds:
                for scanSplit in scanSplits:
                    log_name = f"scanSplit{scanSplit}_ann_lr{lr}_hl{hl}_bs{bs}_seed{seed}"
                    common_args = common_args_base + ['--batch', str(bs)]  # Include batch size in common args
                    # Construct command with current settings and log name
                    command = common_args + ['--logName', log_name, '--hl', str(hl), '--lr', str(lr), '--setRandomSeedTo', str(seed), '--scanSplit', str(scanSplit)]

                    # Run the subprocess with the constructed command
                    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                    # Output information about the current run
                    print(f"Running with lr={lr}, hl={hl}, bs={bs}, seed={seed}, scanSplit={scanSplit}, logName={log_name}")
                    print(f"Standard Output: {result.stdout}")
                    if result.stderr:
                        print(f"Standard Error: {result.stderr}")

                    # Check the result and print appropriate message
                    if result.returncode == 0:
                        print(f"Execution succeeded for lr={lr}, hl={hl}, bs={bs}, seed={seed}, scanSplit={scanSplit}")
                    else:
                        print(f"Execution failed for lr={lr}, hl={hl}, bs={bs}, seed={seed}, scanSplit={scanSplit}")
