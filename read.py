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
parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"
folder = "models/NF.15/Visnav_VR_Expt/ibl/Neuroformer/None/(state_history=6,_state=6,_stimulus=6,_behavior=6,_self_att=6,_modalities=(n_behavior=25))/25"
file_path = glob.glob(os.path.join(folder, 'inference','*.pkl'))[0]
data = np.load(file_path, allow_pickle=True)
print(data.keys())
print(len(data['ID']))
print(len(data['time']))
print(len(data['dt'])) 
print(len(data['Trial']))
print(len(data['true_dt']))

gt_id = data['true']
# gt_id is a list of tensor, convert it to numpy array
gt = []
for i in range(len(gt_id)):
    gt.append(gt_id[i].cpu())
neuron_ids = np.unique(gt)
neuron_ids = np.unique(data['ID'])
time_points = np.unique(data['dt'])
trials = np.unique(data['Trial'])
print(len(neuron_ids))
print(time_points)
print(trials)
exit()
total_neurons = 185
# Initialize a matrix of zeros with dimensions [number of neurons X number of time points]
spiking_matrix = np.zeros((total_neurons, len(time_points)))

# Create a mapping from neuron ID and time to matrix indices for quick lookup
# neuron_index = {neuron_id: idx for idx, neuron_id in enumerate(neuron_ids)}
# print(neuron_index)
# exit()
time_index = {time: idx for idx, time in enumerate(time_points)}

time2idx = {time: idx for idx, time in enumerate(time_points)}
print(data['Trial'])
# Populate the matrix
for idx, neuron_id in enumerate(data['ID']):
    neuron_idx = neuron_id
    time_idx = time_index[str(data['time'][idx])]
    spiking_matrix[neuron_idx, time_idx] += 1  # Assuming each entry in 'ID' represents a single spike

import matplotlib.pyplot as plt

plt.imshow(spiking_matrix, aspect='auto', interpolation='none')
plt.colorbar()
plt.xlabel('Time Points')
plt.ylabel('Neurons')
plt.title('Spiking Activity Matrix')
plt.savefig('spiking_activity.png')
print("Spiking activity matrix saved as 'spiking_activity.png'")
# config, tokenizer, model = load_model_and_tokenizer(folder)
# print(tokenizer)
# decoded_data = tokenizer.decode(data['ID'], 'ID')
# print(len(decoded_data))
# print(decoded_data)
# decoded_dt = tokenizer.decode(data['dt'], 'dt')
# decoded_trial = tokenizer.decode(data['Trial'], 'Trial')
# # decoded_time = tokenizer.decode(data['time'], 'time')