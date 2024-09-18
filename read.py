import numpy as np
import glob
import os
import torch
from pathlib import Path, PurePath
import sys
path = Path.cwd()
parent_path = path.parents[1]
sys.path.append(str(PurePath(parent_path, 'neuroformer')))
sys.path.append('neuroformer')
from neuroformer.model_neuroformer import load_model_and_tokenizer
from neuroformer.datasets import load_ibl_dataset, get_intervals, get_binned_spikes_from_dataset
from neuroformer.utils import bits_per_spike
parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"
folder = "models/NF.15/Visnav_VR_Expt/ibl/Neuroformer/None/(state_history=6,_state=6,_stimulus=6,_behavior=6,_self_att=6,_modalities=(n_behavior=25))/25"
file_path = glob.glob(os.path.join(folder, 'inference','*.pkl'))[0]
data = np.load(file_path, allow_pickle=True)

total_neurons = 185
print(data.keys())
print(len(data['ID']))
print(len(data['time']))
print(len(data['dt'])) 
print(len(data['Trial']))
print(len(data['true_dt']))

# gt_id is a list of tensor, convert it to numpy array
gt_id = []
gt_time = []
for i in range(len(data['true'])):
    gt_id.append(data['true'][i].cpu())
    gt_time.append(data['true_dt'][i].cpu())
neuron_ids = np.unique(gt_id)
neuron_ids = np.unique(data['ID'])
time_points = np.unique(data['dt'])
gt_time_points = np.unique(gt_time)
trials = np.unique(data['Trial'])

# a list of [0, 0.02, 0.04, 0.06, 0.08, 0.1...2.0]
time_list = [str(round((0.02 * i),2)) for i in range(0, 101)]
time_list = time_list + ['EOS', 'PAD', 'SOS']
time_index = {time: idx for idx, time in enumerate(time_list)}

def make_spike_matrix(id, dt, total_neurons):
    spiking_matrix = np.zeros((total_neurons, len(time_index)))
    for idx, neuron_id in enumerate(id):
        neuron_idx = neuron_id
        time_idx = time_index[str(dt[idx])]
        spiking_matrix[neuron_idx, time_idx] += 1  # Assuming each entry in 'ID' represents a single spike
    return spiking_matrix[:, :100]

def plot_spiking_activity(spiking_matrix, trial_id):
    import matplotlib.pyplot as plt
    plt.imshow(spiking_matrix, aspect='auto', interpolation='none')
    plt.colorbar()
    plt.xlabel('Time Points')
    plt.ylabel('Neurons')
    plt.title('Spiking Activity Matrix')
    plt.savefig(f"trial_{trial_id}_spiking_activity.png")
    plt.close()

train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset(
        cache_dir='data',
        split_method="predefined",
        num_sessions=1,
        eid='db4df448-e449-4a6f-a0e7-288711e7a75a'
        )

gt_spikes = get_binned_spikes_from_dataset(test_dataset)
print(gt_spikes.shape)
total_neurons = 185
print(meta_data)
bps_list = []
all_pred_matrix = []
for trial_id in trials:
    trial_idx = np.where(data['Trial'] == trial_id)[0].tolist()
    pred_neuron_id = np.array(data['ID'])[trial_idx]
    pred_dt = np.array(data['dt'])[trial_idx]
    pred_spiking_matrix = make_spike_matrix(pred_neuron_id, pred_dt, total_neurons)
    all_pred_matrix.append(pred_spiking_matrix)
    print(pred_spiking_matrix.shape)
    bps = bits_per_spike(pred_spiking_matrix.T, gt_spikes[trial_id])
    print(f"Trial {trial_id}: {bps}")
    bps_list.append(bps)
print(f"Mean BPS: {np.nanmean(bps_list)}")
all_pred_matrix = np.array(all_pred_matrix).mean(axis=0)
print(all_pred_matrix.shape)
gt_spikes = gt_spikes.mean(axis=0)
plot_spiking_activity(all_pred_matrix, 'all')
plot_spiking_activity(gt_spikes.T, 'gt')


    # plot_spiking_activity(pred_spiking_matrix, trial_id)


# Create a mapping from neuron ID and time to matrix indices for quick lookup
# neuron_index = {neuron_id: idx for idx, neuron_id in enumerate(neuron_ids)}
# print(neuron_index)
# exit()
# config, tokenizer, model = load_model_and_tokenizer(folder)
# print(tokenizer)
# decoded_data = tokenizer.decode(data['ID'], 'ID')
# print(len(decoded_data))
# print(decoded_data)
# decoded_dt = tokenizer.decode(data['dt'], 'dt')
# decoded_trial = tokenizer.decode(data['Trial'], 'Trial')
# # decoded_time = tokenizer.decode(data['time'], 'time')