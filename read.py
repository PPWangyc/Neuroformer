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

config, tokenizer, model = load_model_and_tokenizer(folder)
print(tokenizer)
decoded_data = tokenizer.decode(data['ID'], 'ID')
print(len(decoded_data))
print(decoded_data)
decoded_dt = tokenizer.decode(data['dt'], 'dt')
decoded_trial = tokenizer.decode(data['Trial'], 'Trial')
# decoded_time = tokenizer.decode(data['time'], 'time')