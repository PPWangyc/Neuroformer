import sys
sys.path.append('./neuroformer')

import itertools
import torch
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
# Print all the available datasets
from huggingface_hub import list_datasets
from datasets import Dataset, DatasetInfo, load_dataset, concatenate_datasets,DatasetDict, load_from_disk
from scipy.sparse import csr_array

class DATASET_MODES:
    train = "train"
    val = "val"
    test = "test"
    trainval = "trainval"

DATA_COLUMNS = ['spikes_sparse_data', 'spikes_sparse_indices', 'spikes_sparse_indptr', 'spikes_sparse_shape','cluster_depths']
TARGET_EIDS="data/train_eids.txt"
TEST_RE_EIDS="data/test_eids.txt"

def get_train_eids():
    with open(TARGET_EIDS) as file:
        include_eids = [line.rstrip().replace("'", "") for line in file]
    return include_eids
def get_test_eids():
    with open(TEST_RE_EIDS) as file:
        include_eids = [line.rstrip().replace("'", "") for line in file]
    return include_eids

def get_sparse_from_binned_spikes(binned_spikes):
    sparse_binned_spikes = [csr_array(binned_spikes[i], dtype=np.ubyte) for i in range(binned_spikes.shape[0])]

    spikes_sparse_data_list = [csr_matrix.data.tolist() for csr_matrix in sparse_binned_spikes] 
    spikes_sparse_indices_list = [csr_matrix.indices.tolist() for csr_matrix in sparse_binned_spikes]
    spikes_sparse_indptr_list = [csr_matrix.indptr.tolist() for csr_matrix in sparse_binned_spikes]
    spikes_sparse_shape_list = [csr_matrix.shape for csr_matrix in sparse_binned_spikes]

    return sparse_binned_spikes, spikes_sparse_data_list, spikes_sparse_indices_list, spikes_sparse_indptr_list, spikes_sparse_shape_list

def get_binned_spikes_from_sparse(spikes_sparse_data_list, spikes_sparse_indices_list, spikes_sparse_indptr_list, spikes_sparse_shape_list):
    sparse_binned_spikes = [csr_array((spikes_sparse_data_list[i], spikes_sparse_indices_list[i], spikes_sparse_indptr_list[i]), shape=spikes_sparse_shape_list[i]) for i in range(len(spikes_sparse_data_list))]

    binned_spikes = np.array([csr_matrix.toarray() for csr_matrix in sparse_binned_spikes])

    return binned_spikes

def get_binned_spikes_from_dataset(dataset):
    binned_spikes_data = get_binned_spikes_from_sparse(
        dataset["spikes_sparse_data"], 
        dataset["spikes_sparse_indices"],
        dataset["spikes_sparse_indptr"],
        dataset["spikes_sparse_shape"]
        )
    return binned_spikes_data

def get_intervals(train_dataset, val_dataset, test_dataset):
    train_data = get_binned_spikes_from_dataset(train_dataset)
    val_data = get_binned_spikes_from_dataset(val_dataset)
    test_data = get_binned_spikes_from_dataset(test_dataset)
    bin_size = train_dataset['binsize'][0]
    print(bin_size)
    spikes = np.concatenate([train_data, val_data, test_data], axis=0)
    B, T, N = spikes.shape # Trial, Time, Neuron
    spikes = spikes.reshape(B * T, N).T
    wheel_speed = np.concatenate([
        np.asarray(train_dataset['wheel-speed']), 
        np.asarray(val_dataset['wheel-speed']), 
        np.asarray(test_dataset['wheel-speed'])
    ])
    
    whisker_energy = np.concatenate([
        np.asarray(train_dataset['whisker-motion-energy']), 
        np.asarray(val_dataset['whisker-motion-energy']), 
        np.asarray(test_dataset['whisker-motion-energy'])
    ])
    speed_energy = np.concatenate([wheel_speed, whisker_energy], axis=1)
    data = {
        'spikes': spikes,
        'se': speed_energy,
        'stimulus': None
    }

    # Creating intervals
    train_intervals = np.arange(0, train_data.shape[0]) * bin_size * T
    val_intervals = np.arange(train_data.shape[0], train_data.shape[0] + val_data.shape[0]) * bin_size * T
    test_intervals = np.arange(train_data.shape[0] + val_data.shape[0], train_data.shape[0] + val_data.shape[0] + test_data.shape[0]) * bin_size * T
    
    # Concatenate all intervals
    intervals = np.concatenate([train_intervals, val_intervals, test_intervals])
    return {
        "data": data,
        "intervals": intervals,
        "train_intervals": train_intervals,
        "val_intervals": val_intervals,
        "test_intervals": test_intervals,
        "callback": None
    }

# This function will fetch all dataset repositories for a given user or organization
def get_user_datasets(user_or_org_name):
    all_datasets = list_datasets()
    user_datasets = [d.id for d in all_datasets if d.id.startswith(f"{user_or_org_name}/")]
    return user_datasets

def split_data_by_interval(intervals, r_split=0.8, r_split_ft=0.1):
    chosen_idx = np.random.choice(len(intervals), int(len(intervals) * r_split))
    train_intervals = intervals[chosen_idx]
    test_intervals = np.array([i for i in intervals if i not in train_intervals])
    finetune_intervals = np.array(train_intervals[:int(len(train_intervals) * r_split_ft)])
    return train_intervals, test_intervals, finetune_intervals

def combo3_V1AL_callback(frames, frame_idx, n_frames, **args):
    """
    Shape of frames: [3, 640, 64, 112]
                     (3 = number of stimuli)
                     (0-20 = n_stim 0,
                      20-40 = n_stim 1,
                      40-60 = n_stim 2)
    frame_idx: the frame_idx in question
    n_frames: the number of frames to be returned
    """
    trial = kwargs['trial']
    if trial <= 20: n_stim = 0
    elif trial <= 40: n_stim = 1
    elif trial <= 60: n_stim = 2
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    f_idx_0 = max(0, frame_idx - n_frames)
    f_idx_1 = f_idx_0 + n_frames
    chosen_frames = frames[n_stim, f_idx_0:f_idx_1].type(torch.float32).unsqueeze(0)
    return chosen_frames

def visnav_callback(frames, frame_idx, n_frames, **args):
    """
    frames: [n_frames, 1, 64, 112]
    frame_idx: the frame_idx in question
    n_frames: the number of frames to be returned
    """
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    f_idx_0 = max(0, frame_idx - n_frames)
    f_idx_1 = f_idx_0 + n_frames
    chosen_frames = frames[f_idx_0:f_idx_1].type(torch.float32).unsqueeze(0)
    return chosen_frames

def download_data():
    print(f"Creating directory ./data and storing datasets!")
    print("Downloading data...")
    import gdown
    url = "https://drive.google.com/drive/folders/1O6T_BH9Y2gI4eLi2FbRjTVt85kMXeZN5?usp=sharing"
    gdown.download_folder(id=url, quiet=False, use_cookies=False, output="./data")

def load_V1AL(config, stimulus_path=None, response_path=None, top_p_ids=None):
    if not os.path.exists("./data"):
        download_data()

    if stimulus_path is None:
        # stimulus_path = "/home/antonis/projects/slab/git/slab/transformer_exp/code/data/SImNew3D/stimulus/tiff"
        stimulus_path = "data/Combo3_V1AL/Combo3_V1AL_stimulus.pt"
    if response_path is None:
        response_path = "data/Combo3_V1AL/Combo3_V1AL.pkl"
    
    data = {}
    data['spikes'] = pickle.load(open(response_path, "rb"))
    data['stimulus'] = torch.load(stimulus_path).transpose(1, 2).squeeze(1)

    intervals = np.arange(0, 31, config.window.curr)
    trials = list(set(data['spikes'].keys()))
    combinations = np.array(list(itertools.product(intervals, trials)))
    train_intervals, test_intervals, finetune_intervals = split_data_by_interval(combinations, r_split=0.8, r_split_ft=0.01)

    return (data, intervals,
           train_intervals, test_intervals, 
           finetune_intervals, combo3_V1AL_callback)

def load_visnav(version, config, selection=None):
    if not os.path.exists("./data"):
        download_data()
    if version not in ["medial", "lateral"]:
        raise ValueError("version must be either 'medial' or 'lateral'")
    
    if version == "medial":
        data_path = "./data/VisNav_VR_Expt/MedialVRDataset/"
    elif version == "lateral":
        data_path = "./data/VisNav_VR_Expt/LateralVRDataset/"

    spikes_path = f"{data_path}/NF_1.5/spikerates_dt_0.01.npy"
    speed_path = f"{data_path}/NF_1.5/behavior_speed_dt_0.05.npy"
    stim_path = f"{data_path}/NF_1.5/stimulus.npy"
    phi_path = f"{data_path}/NF_1.5/phi_dt_0.05.npy"
    th_path = f"{data_path}/NF_1.5/th_dt_0.05.npy"

    data = dict()
    data['spikes'] = np.load(spikes_path)
    data['speed'] = np.load(speed_path)
    data['stimulus'] = np.load(stim_path)
    data['phi'] = np.load(phi_path)
    data['th'] = np.load(th_path)

    if selection is not None:
        selection = np.array(pd.read_csv(os.path.join(data_path, f"{selection}.csv"), header=None)).flatten()
        data['spikes'] = data['spikes'][selection - 1]

    spikes = data['spikes']
    print(data['spikes'].shape)
    print(data['speed'].shape)
    print(data['stimulus'].shape)
    print(data['phi'].shape)
    print(data['th'].shape)
    # exit()
    intervals = np.arange(0, spikes.shape[1] * config.resolution.dt, config.window.curr)
    train_intervals, test_intervals, finetune_intervals = split_data_by_interval(intervals, r_split=0.8, r_split_ft=0.01)
    return data, intervals, train_intervals, test_intervals, finetune_intervals, visnav_callback


def load_ibl_dataset(cache_dir,
                     user_or_org_name='neurofm123',
                     aligned_data_dir=None,
                     train_aligned=True,
                     eid=None, # specify 1 session for training, random_split will be used
                     num_sessions=5, # total number of sessions for training and testing
                     split_method="session_based",
                     train_session_eid=[],
                     test_session_eid=[], # specify session eids for testing, session_based will be used
                     split_size = 0.1,
                     mode = "train",
                     batch_size=16,
                     use_re=False,
                     seed=42):
    if aligned_data_dir:
        dataset = load_from_disk(aligned_data_dir)
        # if dataset does not have a 'train' key, it is a single session dataset
        if "train" not in dataset:
            _dataset = dataset.train_test_split(test_size=0.2, seed=seed)
            _dataset_train, _dataset_test = _dataset["train"], _dataset["test"]
            dataset = _dataset_train.train_test_split(test_size=0.1, seed=seed)
            return dataset["train"], dataset["test"], _dataset_test
        return dataset["train"], dataset["val"], dataset["test"]
    
    user_datasets = get_user_datasets(user_or_org_name)
    print("Total session-wise datasets found: ", len(user_datasets))
    cache_dir = os.path.join(cache_dir, "ibl", user_or_org_name)
    test_session_eid_dir = []
    train_session_eid_dir = []
    if eid is not None:
        eid_dir = os.path.join(user_or_org_name, eid+"_aligned")
        if eid_dir not in user_datasets:
            raise ValueError(f"Dataset with eid: {eid} not found in the user's datasets")
        else:
            train_session_eid_dir = [eid_dir]
            user_datasets = [eid_dir]

    if len(test_session_eid) > 0:
        test_session_eid_dir = [os.path.join(user_or_org_name, eid) for eid in test_session_eid]
        print("Test session-wise datasets found: ", len(test_session_eid_dir))
        train_session_eid_dir = [eid for eid in user_datasets if eid not in test_session_eid_dir]
        print("Train session-wise datasets found: ", len(train_session_eid_dir))
        if train_aligned:
            train_session_eid_dir = [eid for eid in train_session_eid_dir if "aligned" in eid]
        else:
            train_session_eid_dir = [eid for eid in train_session_eid_dir if "aligned" not in eid]
        train_session_eid_dir = train_session_eid_dir[:num_sessions - len(test_session_eid)]
        print("Number of training sesssion datasets to be used: ", len(train_session_eid_dir))
    else:
        if len(train_session_eid) > 0:
            train_session_eid_dir = [os.path.join(user_or_org_name, eid+'_aligned') for eid in train_session_eid]
        else:
            train_session_eid_dir = user_datasets
        if train_aligned:
            train_session_eid_dir = [eid for eid in train_session_eid_dir if "aligned" in eid]
        else:
            train_session_eid_dir = [eid for eid in train_session_eid_dir if "aligned" not in eid]
    assert len(train_session_eid_dir) > 0, "No training datasets found"
    assert not (len(test_session_eid) > 0 and split_method == "random_split"), "When you have a test session, the split method should be 'session_based'"

    all_sessions_datasets = []
    if mode == "eval":
        print("eval mode: only loading test datasets...")
        for dataset_eid in tqdm(test_session_eid_dir):
            session_dataset = load_dataset(dataset_eid, cache_dir=cache_dir)["train"]
            all_sessions_datasets.append(session_dataset)
        all_sessions_datasets = concatenate_datasets(all_sessions_datasets)
        test_dataset = all_sessions_datasets.select_columns(DATA_COLUMNS)
        return None, test_dataset
    
    if split_method == 'random_split':
        print("Loading datasets...")
        for dataset_eid in tqdm(train_session_eid_dir[:num_sessions]):
            session_dataset = load_dataset(dataset_eid, cache_dir=cache_dir)["train"]
            all_sessions_datasets.append(session_dataset)
        all_sessions_datasets = concatenate_datasets(all_sessions_datasets)
        # split the dataset to train and test
        dataset = all_sessions_datasets.train_test_split(test_size=split_size, seed=seed)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif split_method == 'predefined':
        print("Loading train dataset sessions for predefined train/val/test split...")
        session_train_datasets = []
        session_val_datasets = []
        session_test_datasets = []

        num_neuron_set = set()
        eids_set = set()
        eid_list = {}
        if use_re:
            target_eids = get_train_eids()
            test_re_eids = get_test_eids()
            train_session_eid_dir = [eid for eid in train_session_eid_dir if eid.split('_')[0].split('/')[1] in target_eids]
            # remove the test_re_eids from the train_session_eid_dir
            train_session_eid_dir = [eid for eid in train_session_eid_dir if eid.split('_')[0].split('/')[1] not in test_re_eids]
        for dataset_eid in tqdm(train_session_eid_dir[:num_sessions]):
            try:
                # print("Loading dataset: ", dataset_eid)
                session_dataset = load_dataset(dataset_eid, cache_dir=cache_dir)
                train_trials = len(session_dataset["train"]["spikes_sparse_data"])
                train_trials = train_trials - train_trials % batch_size
                session_train_datasets.append(session_dataset["train"].select(list(range(train_trials))))

                val_trials = len(session_dataset["val"]["spikes_sparse_data"])
                val_trials = val_trials - val_trials % batch_size
                session_val_datasets.append(session_dataset["val"].select(list(range(val_trials))))

                test_trials = len(session_dataset["test"]["spikes_sparse_data"])
                test_trials = test_trials - test_trials % batch_size
                session_test_datasets.append(session_dataset["test"].select(list(range(test_trials))))
                binned_spikes_data = get_binned_spikes_from_sparse([session_dataset["train"]["spikes_sparse_data"][0]], 
                                                                    [session_dataset["train"]["spikes_sparse_indices"][0]],
                                                                    [session_dataset["train"]["spikes_sparse_indptr"][0]],
                                                                    [session_dataset["train"]["spikes_sparse_shape"][0]])

                num_neuron_set.add(binned_spikes_data.shape[2])
                eid_prefix = dataset_eid.split('_')[0] if train_aligned else dataset_eid
                eid_prefix = eid_prefix.split('/')[1]
                eids_set.add(eid_prefix)
                assert eid_prefix not in eid_list, f"Duplicate eid found: {eid_prefix}"
                eid_list[eid_prefix] = binned_spikes_data.shape[2]
            except Exception as e:
                print("Error loading dataset: ", dataset_eid)
                print(e)
                continue
        print("session eid used: ", eids_set)
        print("Total number of session: ", len(eids_set))
        train_dataset = concatenate_datasets(session_train_datasets)
        val_dataset = concatenate_datasets(session_val_datasets)
        test_dataset = concatenate_datasets(session_test_datasets)
        print("Train dataset size: ", len(train_dataset))
        print("Val dataset size: ", len(val_dataset))
        print("Test dataset size: ", len(test_dataset))
        num_neuron_set = list(num_neuron_set)
        meta_data = {
            "num_neurons": num_neuron_set,
            "num_sessions": len(eids_set),
            "eids": eids_set,
            "eid_list": eid_list
        }
    elif split_method == 'session_based':
        print("Loading train dataset sessions...")
        for dataset_eid in tqdm(train_session_eid_dir):
            session_dataset = load_dataset(dataset_eid, cache_dir=cache_dir)["train"]
            all_sessions_datasets.append(session_dataset)
        train_dataset = concatenate_datasets(all_sessions_datasets)

        print("Loading test dataset session...")
        all_sessions_datasets = []
        for dataset_eid in tqdm(test_session_eid_dir):
            session_dataset = load_dataset(dataset_eid, cache_dir=cache_dir)["train"]
            all_sessions_datasets.append(session_dataset)
        test_dataset = concatenate_datasets(all_sessions_datasets)
        
        train_dataset = train_dataset
        test_dataset = test_dataset
    else:
        raise ValueError("Invalid split method. Please choose either 'random_split' or 'session_based'")
    
    return train_dataset, val_dataset, test_dataset, meta_data