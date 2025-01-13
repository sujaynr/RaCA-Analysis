import subprocess


learning_rates = [0.00005]
hidden_layer_sizes = [1517]
batch_sizes = [1024]
noises = [0.0, 0.001, 0.1, 0.0001, 0.5]

script_path = '/home/sujaynair/RaCA-Analysis/updatedTrain.py'
common_args = ['python3', script_path, '--trainValSplit', '0.4', '--fixRandomSeed', '--encoderModel', 's', '--fullFit', '--epochs', '500', '--decoderModel']

for lr in learning_rates:
    for hl in hidden_layer_sizes:
        for bs in batch_sizes:
            for noise in noises:
                log_name = f"dataAUGsweep_lr{lr}_hl{hl}_bs{bs}_noise{noise}"

                # Construct command with current hidden layer size, learning rate, batch size, noise level, and log name
                command = common_args + ['--logName', log_name, '--hl', str(hl), '--lr', str(lr), '--batch', str(bs), '--noise', str(noise)]

                # Run the subprocess with the constructed command
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Output information about the current run
                print(f"Running with lr={lr}, hl={hl}, bs={bs}, noise={noise}, logName={log_name}")
                print(f"Standard Output: {result.stdout}")
                if result.stderr:
                    print(f"Standard Error: {result.stderr}")

                # Check the result and print appropriate message
                if result.returncode == 0:
                    print(f"Execution succeeded for lr={lr}, hl={hl}, bs={bs}, noise={noise}")
                else:
                    print(f"Execution failed for lr={lr}, hl={hl}, bs={bs}, noise={noise}")
