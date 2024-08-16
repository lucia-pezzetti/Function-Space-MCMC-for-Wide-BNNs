import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm


def load_samples(checkpoint_dir):
    """Load all samples from checkpoint files in a directory."""
    filepath = os.path.join(checkpoint_dir, 'samples')
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

def compute_ess_for_columns(samples):
    """Compute ESS for each column of a sample matrix."""
    ess_values = []
    print(samples.shape[0])
    for i in range(samples.shape[1]):
        column_samples = samples[:, i]
        ess = tfp.mcmc.effective_sample_size(column_samples, cross_chain_dims=None)
        ess_values.append(ess.numpy()/samples.shape[0])  # Assuming computation on a single chain
    return np.array(ess_values)

def plot_ess_statistics(DIMS, ess_values, fig, subplot_index):
    """Plot the mean, max, and min of ESS values on given subplot index."""
    labels = ['LMC', 'pCN', 'pCNL']
    colors = ['blue', 'darkorange', 'green']
    light_colors = ['lightblue', 'sandybrown', 'lightgreen']
    markers = ['s', 'o', 'o']
    
    ax = fig.add_subplot(1, 2, subplot_index)

    for i in range(ess_values.shape[0]):
        mean_ess = np.mean(ess_values[i, :, :], axis=1)
        max_ess = np.max(ess_values[i, :, :], axis=1)
        min_ess = np.min(ess_values[i, :, :], axis=1)
        
        ax.plot(DIMS, mean_ess, marker=markers[i], color=colors[i], linewidth=2, markersize=4, label=labels[i])
        ax.plot(DIMS, max_ess, color=light_colors[i], linestyle='--')
        ax.plot(DIMS, min_ess, color=light_colors[i], linestyle='--')
        ax.fill_between(DIMS, max_ess, min_ess, color=light_colors[i], alpha=0.3)

    ax.legend(fontsize=14, loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xticks(DIMS)
    ax.set_xticklabels(DIMS, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlabel('Width', fontsize=16)
    ax.set_ylabel('ESS/step', fontsize=16)
    ax.set_title(f'ESS of {["weights", "predictions"][subplot_index-1]} per width', fontsize=16)
    

DIMS = [128, 512, 1024, 2096, 4192]
labels = [128, 512, 1024, 2048, 4096]
samplers = ['hmc', 'pcn', 'pcnl']
stepsize = 0.1
objects = ['theta', 'preds']
ess_values = np.zeros([3,5,100])
fig = plt.figure(figsize=(24, 6))

for obj in objects:
    for i, sampler in enumerate(samplers):
        print(f'sampler: {sampler}')
        for j in tqdm(range(len(DIMS))):
                checkpoint_dir = f'results/stepsize_{stepsize}_nunits_{DIMS[j]}_{sampler}'
            samples = load_samples(checkpoint_dir)
            tf_samples = tf.convert_to_tensor(samples[obj], dtype=tf.float32)

            # Compute ESS values
            ess_values[i,j,:] = compute_ess_for_columns(tf_samples)



    # Plot the ESS statistics
    if obj == 'theta':
        plot_ess_statistics(labels, ess_values, fig, 1)
    else:
        plot_ess_statistics(labels, ess_values, fig, 2)

plt.savefig("ESS_statistics.pdf", dpi=300, bbox_inches='tight', format='pdf')  # Save the figure as a PNG file with high resolution
