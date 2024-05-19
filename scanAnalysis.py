import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pdb


data = pd.read_csv('/Users/sujaynair/Downloads/wandb_export_2024-05-15T17_27_41.566-07_00.csv')
data_array = data.to_numpy()

ND_data = data_array[data_array[:, 3] == True]
annNOVAL_data = data_array[(data_array[:, 3] == False) & (data_array[:, 4] == False)]
annVAL_data = data_array[(data_array[:, 3] == False) & (data_array[:, 4] == True)]

unique_scan_splits = np.unique(data_array[:, 1])

# Initialize and calculate averages
ND_averages = [np.mean(ND_data[ND_data[:, 1] == split, 2].astype(float)) for split in unique_scan_splits[1:]]
annNOVAL_averages = [np.mean(annNOVAL_data[annNOVAL_data[:, 1] == split, 2].astype(float)) for split in unique_scan_splits[1:]]
annVAL_averages = [np.mean(annVAL_data[annVAL_data[:, 1] == split, 2].astype(float)) for split in unique_scan_splits[1:]]

scan_splits = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Plot average R^2 values
plt.figure(figsize=(12, 8))
plt.plot(scan_splits, ND_averages, marker='o', linestyle='-', color='blue', label='Encoder Only')
plt.plot(scan_splits, annNOVAL_averages, marker='s', linestyle='--', color='green', label='Encoder + Decoder (No Val Scans)')
plt.plot(scan_splits, annVAL_averages, marker='^', linestyle='-.', color='red', label='Encoder + Decoder')
plt.title('Validation R^2 vs % Labeled Data', fontsize=16)
plt.xlabel('% Labeled Data', fontsize=14)
plt.ylabel('Validation R^2', fontsize=14)
plt.xticks(scan_splits)
plt.grid(True)
plt.legend(title='Model Type', fontsize=12)
plt.savefig('/Users/sujaynair/Documents/average_r2_plot.png')  # Save the figure
plt.show()

# Running Avgs
def calculate_running_average(data):
    cumulative_sum = np.cumsum(data)
    counts = np.arange(1, len(data) + 1)
    return cumulative_sum / counts

ND_running_avg = calculate_running_average(ND_averages)
annNOVAL_running_avg = calculate_running_average(annNOVAL_averages)
annVAL_running_avg = calculate_running_average(annVAL_averages)

# Plot running average R^2 values
plt.figure(figsize=(12, 8))
plt.plot(scan_splits, ND_running_avg, marker='o', linestyle='-', color='blue', label='Encoder Only')
plt.plot(scan_splits, annNOVAL_running_avg, marker='s', linestyle='--', color='green', label='Encoder + Decoder (No Val Scans)')
plt.plot(scan_splits, annVAL_running_avg, marker='^', linestyle='-.', color='red', label='Encoder + Decoder')
plt.title('Running Average Validation R^2 vs % Labeled Data', fontsize=16)
plt.xlabel('% Labeled Data', fontsize=14)
plt.ylabel('Running Average Validation R^2', fontsize=14)
plt.xticks(scan_splits)
plt.grid(True)
plt.legend(title='Model Type', fontsize=12)
plt.savefig('/Users/sujaynair/Documents/running_avg_r2_plot.png')  # Save the figure
plt.show()
