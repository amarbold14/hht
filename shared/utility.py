from sklearn.preprocessing import StandardScaler
import logging
import os
import numpy as np
import random
import torch
from collections import defaultdict
import time
import pandas as pd
from pygit2 import Repository


class SharedVariable:
    """
    shared variable class
    """

    def __init__(self):
        # There are a few configurations dependent on the host machine you work on
        # git_branch_name = Repository('.').head.shorthand
        self.host_machine = 'amar' # Now it automatically matches your workstation(branch)

        # Might deprecate later we will see
        # self.host_machine = "WorkStation"
        # self.host_machine = 'Steven-Local'
        # ^^^^^^^^^^^^

        self.default_train_class_number_list_multi, self.default_train_class_number_list_bin = None, None
        self._init_based_on_host_machine()

        # Class str to index mapping, and the other way around
        self.default_multi_class_map = {'H': 0, 'N': 1, 'NV': 2, 'D': 3, 'ND': 4, 'NVD': 5}
        self.default_multi_class_map_inverse = {0: 'H', 1: 'N', 2: 'NV', 3: 'D', 4: 'ND',
                                                5: 'NVD'}  # The other way round
        self.default_binary_class_map = defaultdict(lambda: 1)  # everything is 'fault' except 'H'
        self.default_binary_class_map['H'] = 0

        # Paths for data
        self.latest_column_name = "latest_column_name.csv"  # .csv where the latest column names are specified relative to the dataset directory
        self.path_loaded_raw_data = "/data/loaded_raw_data/"  # path_loaded_raw_data: path to look for loaded raw data (relative to project dir)
        self.raw_class_labels = ["H", "V", "N", "D"]  # raw class labels from the dataset, trivial
        self.path_cleaned_up_data = "/data/cleaned_up_data/"
        self.path_sq_label_data = "/data/sequence_label_data/"

        # Some where in MyData this is called to assert sufficient number of samples exist
        self.min_number_of_sample = {'train': 20000, 'val': 5000, 'test': 3000}

    def _init_based_on_host_machine(self):
        """
        :param self.path_rosa: where RoSA is located
        :param self.result_filename: the .xlsx filename of batch result
        :param self.dropbox_result_path: where to save the third copy of batch result
        """
        if self.host_machine == "Workstation":
            self.path_rosa = r"C:\git\MIT-MRL-RoSA-Data-Archive"  # MRL-Workstation
            self.default_spe = -1

            # Path to save result data to
            self.result_filename = "result_ws.xlsx"
            self.dropbox_result_path = r'C:\Users\steve\Dropbox (MIT)\MIT\Write Paper\Few-Shot\Result/'

            # Number of samples for minor classes during training
            # Will populate self.default_train_class_number_list_multi/_bin
            # self.default_minor_class_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 32, 64, 128,
            #                                    256, 512, 768,
            #                                    800, 1024, 2048, 4096, 10000, 12000, 15000, 20000, 25000]
            self.default_minor_class_counts = [768,
                                               800, 1024, 2048, 4096, 10000, 12000, 15000, 20000, 25000]

        elif self.host_machine == 'amar':
            self.path_rosa = r"C:\Users\MRL - Workstation\Documents\GitHub\MRL_Few_Shot\data" # Steven-Local
            self.default_spe = -1

            # Path to save result data to
            self.result_filename = "result_amar.xlsx"
            self.dropbox_result_path = r"C:\Users\MRL - Workstation\Documents\GitHub\MRL_Few_Shot\results/"

            # Number of samples for minor classes during training
            # Will populate self.default_train_class_number_list_multi/_bin

            # self.default_minor_class_counts = [20000]
            self.default_minor_class_counts = [12, 16, 24, 32, 64, 128, 256, 512, 768,
                                               800, 1024, 2048, 4096, 10000, 12000, 15000, 20000, 25000]



        elif self.host_machine == 'Steven-Mac':
            # TODO @yeung: modify the path for MAC
            self.path_rosa = r"C:\Users\steve\Dropbox (MIT)\PhD Research\GPAD Archive (git)\MIT-MRL-GPAD-Data-Archive"  # Steven-Local
            self.default_spe = 200

            # Path to save result data to
            self.result_filename = "result_steven.xlsx"
            self.dropbox_result_path = r"C:\Users\steve\Dropbox (MIT)\MIT\Write Paper\Few-Shot\Result/"

            # Number of samples for minor classes during training
            # Will populate self.default_train_class_number_list_multi/_bin
            self.default_minor_class_counts = [1, 2, 3, 4, 5]

        # TODO yeung: is this unsued?
        self.train_val_class_numbers = {'train': 25000, 'val': -1}  # train max is 25000

        # Populate self.default_train_class_number_list_multi/_bin
        self.default_train_class_number_list_multi, self.default_train_class_number_list_bin = self._generate_default_train_class_number_lists()

    def _generate_default_train_class_number_lists(self):
        """
        Generates the default train class number list <list of <dict>s>
        These are the default values of <train_class_number> in the main training loop.
        """
        train_class_number_list_multi, train_class_number_list_bin = [], []
        classes = ['H', 'N', 'NV', 'D']

        for minor_class_count in self.default_minor_class_counts:
            train_class_number_multi, train_class_number_bin = {}, {}

            # We might want to have a parameterized distribution
            class_counts_multi = [3 * minor_class_count, minor_class_count, minor_class_count, minor_class_count]
            class_counts_bin = [3 * minor_class_count, minor_class_count, minor_class_count, minor_class_count]

            # Parse data
            for class_idx, class_label in enumerate(classes):
                train_class_number_multi[class_label] = class_counts_multi[class_idx]
                train_class_number_bin[class_label] = class_counts_bin[class_idx]
            train_class_number_list_multi.append(train_class_number_multi)
            train_class_number_list_bin.append(train_class_number_bin)
        return train_class_number_list_multi, train_class_number_list_bin


# Test case
test = 0
if test:
    SV = SharedVariable()
    print(SV.default_train_class_number_list_bin)


def util_convert_tuple_class_label_to_class_idx(class_label_tuple, class_type, SV=SharedVariable()):
    """
    Args:
        class_label_tuple: ('H','NV','N'...)
        class_type: 'binary' vs 'multi'
        SV: SharedVariable
    Returns:
        class_label_idx: <list>
    """
    class_label_idx = []
    for class_label in class_label_tuple:
        if class_type == 'binary':
            class_label_idx.append(SV.default_binary_class_map[class_label])  # get the class index of class_label
        elif class_type == 'multi':
            class_label_idx.append(SV.default_multi_class_map[class_label])
    return class_label_idx


def define_multi_class_label(df):
    N = df['N-mode?'] == True
    V = df['V-mode?'] == True
    D = df['D-mode?'] == True
    if N and V and D:
        return 'NVD'
    elif (not N) and (not V) and (not D):
        return 'H'
    elif (not N) and not (V) and D:
        return 'D'
    elif N and (not V) and (not D):
        return 'N'
    elif V and (not D):
        return 'NV'
    elif V and D:  # same as first case
        return 'NVD'
    elif N and (not V) and D:
        return 'ND'


def normalize_numpy_and_return_scaler(X):
    """
    Input: X [number of samples, time step per sample, feature dimension]
    or Y[number of samples, feature dimension]
    :return: Normalized X, scaler
    """
    scaler = StandardScaler()
    X_shape = X.shape  # record shape of X
    X = X.reshape((-1, X.shape[-1]))  # We will have to do a reshape to [n_samples, n_features]
    scaler.fit(X)
    X_norm = scaler.transform(X)
    X = X_norm.reshape(X_shape)
    return X, scaler


def normalize_numpy_with_scaler(X, scaler):
    """
    Input: X [number of samples, time step per sample, feature dimension]
    or Y[number of samples, feature dimension]
    :return: Normalized X, scaler
    """
    X_shape = X.shape  # record shape of X
    X = X.reshape((-1, X.shape[-1]))  # We will have to do a reshape to [n_samples, n_features]
    X_norm = scaler.transform(X)
    X = X_norm.reshape(X_shape)
    return X


def util_seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=2)


def initialize_logger_and_device(logger_name, debug=False):
    # Set up shared variable
    SV = SharedVariable()
    # Set up print option
    np.set_printoptions(precision=3, suppress=True)
    # Set up logger
    logger_name = time.strftime("%m%d_%H%M%S") + "_" + logger_name
    filename_name = "../runs/" + logger_name + ".log"
    file_handler = logging.FileHandler(filename_name, mode='w')
    os.makedirs(os.path.dirname(filename_name), exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.addHandler(file_handler)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL)
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return SV, logger, device


def util_init_SV_logger_device_seed(logger_name, seed, debug=False):
    util_seed_everything(seed=seed)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    return initialize_logger_and_device(logger_name, debug=debug)


def report_converted_data_info(logger, X, Ys, type_of_data):
    """
    Will need to feed in Ys
    """
    logger.critical("Meta-Data of " + type_of_data + " set:")
    logger.critical("Shape of feature [number of sequence, window length, dimension]:{}".format(X.shape))
    logger.critical("Shape of state label [number of sequence, dimension]:{}".format(Ys.shape))

    logger.debug(
        "Checking feature, observation and state label info of the first sample:feature:{},\nstate label:{}".format(
            X[0, -1, :], Ys[0, :]))

    unique_state_labels, unique_state_labels_counts = np.unique(Ys, return_counts=True)
    logger.critical(
        "Checking unique state labels for {} data, there are {} labels, each has {} counts".format(
            type_of_data, unique_state_labels, unique_state_labels_counts))


class DictMap(object):
    """
    Usage: new_dict2 = DictMap(key_map, old_dict)
    """

    def __init__(self, key_map, old_dict):
        self.key_map = key_map
        self.old_dict = old_dict

        self.temp_dict = {}
        for keys, values in self.old_dict.items():
            self.temp_dict[self.key_map[keys]] = values

    def map(self):
        return self.temp_dict
