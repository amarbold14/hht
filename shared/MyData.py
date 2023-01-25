import os
import random
from os import listdir
from collections import deque
import pandas as pd
import time
import logging
import csv
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle as pkl
from numpy.random import default_rng
from sklearn.model_selection import train_test_split

from shared.utility import normalize_numpy_and_return_scaler, normalize_numpy_with_scaler, SharedVariable, \
    define_multi_class_label, DictMap

from collections import defaultdict
from random import choices

# Set up logger
logging.basicConfig()
logger = logging.getLogger("MyDataProcessor Default logger")

# Set up df display option
pd.set_option('display.max_columns', None)


class RawData:
    def __init__(self, path_dataset, class_labels, latest_column_name="latest_column_name.csv",
                 path_loaded_raw_data="/data/loaded_raw_data/", overwrite_data_file=False,
                 debug=False):
        """
        Input Members
        ----------
        class_labels: string from ["H","V","N","D"] i.e., uses all four classes as labels
        latest_column_name: names of valid columns that are used
        path_loaded_raw_data: path to look for loaded raw data (relative to current jupyter path)
            e.g. "/data/loaded_new_data/" if .pkl in  ~/MRL_Few_Shot/data/loaded_new_data
        path_dataset: path to RoSA dataset (absolute path)
        overwrite_data_file: whether to overwrite data file even it is processed before

        Intermediate(Private)Members
        _loaded_data_name
        _overwrite
        _dir_list: the subdirectory names in RoSA

        Public Members
        -------
        df: Stores all loaded data
        class_labels: accessible to process with CleanUpData
        path_dataset
        exp_id_list: the experiment ids <list> of that are parsed
        """
        self._set_verbosity(debug)

        self.path_dataset = path_dataset
        self._path_loaded_raw_data = path_loaded_raw_data
        self.class_labels = class_labels  # public

        self._loaded_data_name = self.__gen_loaded_data_name()
        self._latest_column_name = latest_column_name

        self.df = pd.DataFrame()  # init the df, public member
        self.exp_id_dict = {}  # init the dictionary, public member
        self._dir_list = []  # store all the directories that contains the wanted data

        self._overwrite = overwrite_data_file

        self.__main__()

    def _set_verbosity(self, debug):
        logging.basicConfig()
        self._logger = logging.getLogger('Raw Data logger')
        self._debug = debug
        if self._debug:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.CRITICAL)

    def __main__(self):
        self._logger.critical("Processing Raw Data for {} modes...".format(self.class_labels))
        start_time = time.time()

        if self._overwrite:
            self._logger.critical("Force overwriting data file, loading raw .csv files...")
            self.__process_data()
        else:
            self._logger.critical("Trying to load {}...".format(self._loaded_data_name))
            try:
                # try to look .pkl for MRL_Few_Shot/data/loaded_raw_data
                self.df = pd.read_pickle(self._loaded_data_name)
                self._logger.critical("Raw data already processed before, loading previous pickle file...")
            except:
                self._logger.critical("Raw data never processed before, loading raw .csv files...")
                self.__process_data()
        self.exp_id_list = self.df['Exp.ID'].unique().astype(int)

        self._logger.info("Dataset rows: {}".format(len(self.df)))
        self._logger.info("Took {:2f} seconds to load dataset.".format(time.time() - start_time))
        self._logger.critical("Data loaded.")

    def __gen_loaded_data_name(self):
        """
        generate the name of the file either to read from or to save to

        """
        cwd = os.getcwd()  # get current working directory

        class_labels_sorted = sorted(self.class_labels)  # automatically sort the class labels

        processed_data_name = "raw_data_"
        for class_name in class_labels_sorted:
            processed_data_name += class_name + "_"

        processed_data_name = cwd + self._path_loaded_raw_data + processed_data_name + ".pkl"
        return processed_data_name

    def __process_data(self):
        self.__import_column_names()
        self.__finddir()
        self.__loop_through_dir()
        self.__save_read_csv_as_pickle()

    def __import_column_names(self):
        """
        called by main function
        imports latest column names from RoSA dataset
        """
        self.df = pd.read_csv(self.path_dataset + "/" + self._latest_column_name)

    def __finddir(self):
        """
        called by main function
        find directory in the RoSA dataset that contains the modes we want
        """
        dir_filter_kw_1 = "Mode"
        dir_filter_kw_list = self.__label2dirname()
        self._logger.info("Looking for the correct directories...")
        dir_iterator = os.walk(self.path_dataset)  # iterate through subdirectories
        for dir_info in dir_iterator:
            dir_name = dir_info[0]  # extract dir names in the directory
            if (dir_filter_kw_1 in dir_name):  # remove all the .git dirs
                for dir_filter_kw_X in dir_filter_kw_list:
                    if dir_filter_kw_X in dir_name:
                        self._dir_list.append(dir_name)
                        break

    def __label2dirname(self):
        """
        map the class labels to kws to find the directories and files
        """
        label_2_dir_map = {"H": "Healthy", "D": "D M", "N": "N M", "V": "V M"}
        dir_to_open_identifier = deque()
        for label in self.class_labels:
            dir_to_open_identifier.append(label_2_dir_map[label])
        return dir_to_open_identifier

    def __loop_through_dir(self):
        """
        called by main function
        go through directory for every mode and do something
        """
        extension = ".csv"
        for directory in self._dir_list:  # will use another iterator, so that the previous loop doesn't get too long
            fnames_list = listdir(directory)
            csv_file_count = 0
            for fname in fnames_list:  # csv file name
                if fname.endswith(extension):
                    csv_file_count += 1
                    fname_with_dir = directory + "/" + fname  # prepend .csv files with the directory
                    exp_id = self.__load_csv(fname_with_dir)
            self._logger.info(self.exp_id_dict)
            self._logger.info("Found {} .csv files".format(csv_file_count))

    def __load_csv(self, fname_with_dir):
        """
        load .csv file and append to self.df
        """
        self._logger.info(fname_with_dir)
        local_df = pd.read_csv(fname_with_dir, index_col=None, header=0)

        exp_id = int(''.join(filter(str.isdigit, fname_with_dir)))  # extract merged_dataXX.csv: file name number
        local_df['Exp.ID'] = exp_id  # save this information as well

        self.df = self.df.append(local_df)
        return exp_id

    def __save_read_csv_as_pickle(self):
        self.df.to_pickle(self._loaded_data_name)


class CleanedUpData:
    def __init__(self, raw_data, class_map, binary_class_map, path_processed_data="/data/cleaned_up_data/",
                 valid_columns="valid_columns.csv", debug=False):
        """
        This class will clean up the Raw Data:
            - Remove deprecated columns
            - Verify valid columns
            - Fillnan

        Input Members
        ----------
        raw_data: class RawData()
        debug: set verbosity
        valid_columns: latest valid columns
        class map: maps the multi-class<str> into numerical classes<int>

        Intermediate (Private) Members
        _raw_df : df from Rawdata class
        _class_labels: inherit from Rawdata class
        _class_map: maps the combined categories into numerical classes

        Public
        -------
        self.df is the processed dataframe
            df['multi_class'] includes combined mode {H, VN, VND, N, ND, D} all six types of policies
            df['multi_class_enc'] are the df['multi_class'] that are mapped with the class map
            df['binary_class_enc'] are the df['multi_class'] that are mapped binary into 0 or 1
        self.valid_column_info: contains the str() column names divided into 'shared', 'label' and 'feature'
        self.exp_id_list: Exp id in the experiment

        """
        self._set_verbosity(debug)

        self._raw_data = raw_data
        self._path_dataset = raw_data.path_dataset
        self._class_map = class_map  # mapping of various combined categories into a numerical class
        self._binary_class_map = binary_class_map

        self.df = self._raw_data.df  # Public
        self._class_labels = self._raw_data.class_labels
        self._path_processed_data = path_processed_data
        self._valid_columns = valid_columns
        self.exp_id_list = self._raw_data.exp_id_list

        self.__main__()

    def _set_verbosity(self, debug):
        logging.basicConfig()
        self._logger = logging.getLogger('Cleanedup Data logger')
        self._debug = debug
        if self._debug:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.CRITICAL)

    def __main__(self):
        self._logger.critical("Cleaning Up Raw Data...")
        self.cleaned_up_data_name = self.__get_cleaned_up_data_name()

        try:
            self.df = pd.read_pickle(self.cleaned_up_data_name)
            self.valid_column_info = self.__identify_valid_columns()
            self._logger.critical("Data already cleaned up before, loading previous pickle file...")
            self._logger.info("Pickle file loaded.")
        except:
            self._logger.critical("Data never cleaned up before")
            self.valid_column_info = self.__identify_valid_columns()
            self.__remove_deprecated_columns()
            self.__fillnan()
            self.__parse_individual_mode_to_class_labels()
            self.__save_read_csv_as_pickle()

        self.__check_health_modes('multi_class')
        self._logger.critical("Processing execution completed.")
        self._logger.critical(
            "Cleaned data is at {}, extract data with self.df.".format(self.cleaned_up_data_name))

    def __parse_individual_mode_to_class_labels(self):
        """
        This function parses the individual columns: <V,D,N> Columns to a new column called 'multi_class'
        :return: nothing, but adds 'multi_class' and 'multi_class_enc to self.df, the values should be {H, VN, VND, N, ND, D}
        """
        self.df['multi_class'] = self.df.apply(lambda df: define_multi_class_label(df), axis=1)  # in utility.py

        # very basic encoding, see utility and you will understand
        self.df['multi_class_enc'] = self.df['multi_class'].map(self._class_map)
        self.df['binary_class_enc'] = self.df['multi_class'].map(self._binary_class_map)

    def __get_cleaned_up_data_name(self):
        """
        get the name of the file either to read from or to save to

        """
        cwd = os.getcwd()  # get current working directory

        class_labels_sorted = sorted(self._class_labels)  # automatically sort the class labels
        cleaned_up_data_name = "/cleaned_up_data_"

        for class_name in class_labels_sorted:
            cleaned_up_data_name += class_name + "_"

        cleaned_up_data_name = cwd + self._path_processed_data + cleaned_up_data_name + ".pkl"
        self._logger.critical(cleaned_up_data_name)

        return cleaned_up_data_name

    def __identify_valid_columns(self):
        """
        identify what are the valid (not deprecated) columns given the valid_columns.csv file

        in the .csv file, the first row is the feature columns, the second rows are the label columns

        RETURN:
        a dictionary containing the valid column information, including all columns, feature columns and label columns
        """
        fname = self._path_dataset + "/" + self._valid_columns  # read valid column name from RoSA repo
        self._logger.critical("Parsing valid column data...")
        self._logger.debug("Valid column file: {}".format(fname))

        valid_column_info = {}
        with open(fname, encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)

            for row, names in enumerate(csv_reader):
                if row == 0:
                    self._logger.debug("Parsing shared data...")
                    shared_names = []
                    for column_name in names:  # remove empty label names ("")
                        if column_name:
                            shared_names.append(column_name)
                    valid_column_info['shared'] = shared_names
                elif row == 1:
                    self._logger.debug("Parsing feature space data...")
                    valid_column_info['feature'] = names
                elif row == 2:
                    self._logger.debug("Parsing label space data...")
                    label_names = []
                    for column_name in names:  # remove empty label names ("")
                        if column_name:
                            label_names.append(column_name)
                    valid_column_info['label'] = label_names

        valid_column_info['all'] = valid_column_info['shared'] + valid_column_info['feature'] + valid_column_info[
            'label']
        self._logger.debug("Valid column info: {}".format(valid_column_info))
        return valid_column_info

    def __remove_deprecated_columns(self):
        """
        remove deprecated columns
        """
        self._logger.critical(
            "Removing deprecated columns, column count before removal: {}".format(len(self.df.columns)))
        self.df = self.df[self.df.columns.intersection(self.valid_column_info['all'])]
        self._logger.critical("Removed deprecated columns, column count after removal: {}".format(len(self.df.columns)))
        self._logger.debug("Remaining column names: {}".format(self.df.columns.values))

    def __check_health_modes(self, col_name):
        """
        Checking how many timesteps of data are in each mode, by default we use all HVDN modes for processing
        This function only prints out the information
        """
        self._logger.debug("Listing dataset information")
        self._logger.debug(self.df.describe())

        self._logger.critical("Checking health modes within this dataset...")
        health_mode_info = self.df[col_name].value_counts()
        health_mode_available = health_mode_info.index.values
        health_mode_step_count = health_mode_info.values
        self._logger.critical("Dataset contains {} modes, {} steps out of {} steps total".format(health_mode_available,
                                                                                                 health_mode_step_count,
                                                                                                 len(self.df)))

    def __fillnan(self):
        """
        interpolate NAN and inf
        """
        self.df = self.df.interpolate()
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        col_to_fillnan = ['Sample Freq']
        for col in col_to_fillnan:
            self.df[col] = self.df[col].fillna(self.df[col].mean())
        # self._logger.debug("Max value in each column:{}".format(self.df.max().to_string))

    def __normalize_feature_space(self):
        self._logger.critical("Normalizing feature input based on StandardScaler...")
        scaler = StandardScaler()
        #         print(self.df.columns.values)
        self.df[self.valid_column_info['feature']] = scaler.fit_transform(self.df[self.valid_column_info['feature']])
        self._logger.debug("Max value in each column after normalization:{}".format(self.df.max().to_string))

    def __save_read_csv_as_pickle(self):
        self.df.to_pickle(self.cleaned_up_data_name)


class SequenceLabelData:
    """
    Process df into something trainable into: {<sequence>, <label>}
    window_length is how many steps are used to predict the upcoming value

    The data X, Yo are normalized based on the data from the training set

    input:
    state_label_col: the 'state' column names that will be used as labels
    binary_state_label_col: same as state_label_col, but binary
    obs_col: the 'observation' column names that will be used as features or self-supervised labels
    cleaned_up_data: CleanUpData()
    (DEPRECATED)encoding map: how to convert string labels ['H','N','V','D'] into multi-class categories
    path_sq_label_data: dir where previously processed SequenceLabelData are stored
    tvt_exp_id_fname: .csv file to parse tvt_exp_id
    shuffle_tvt: whether to reshuffle, if shuffled, we will overwrite the tvt_exp_id_fname file
    demo_id_list: exp_ids that are used for demo, it won't go into training set
    train_test_val_port: portion to split into train, e.g. - 0.7 means 0.7 percent of experiment id goes to train
    min_number_of_sample: min number of training samples, throw an error if not satisfied, see SV

    private:
    _pkl_names: {} containing all pkl names, keys are _dataset_types
    _dataset_types = ['trian', 'test', 'val', 'demo']
    _train_portion : portion of train set, i.e., 0.7
        (NOTE) by default, we split the rest of the data to test and validation in a 2:1 ratio
    _min_number_of_sample = min_number_of_sample

    public:
    self.X_<dataset_type>: Sequence
    self.Yo_<dataset_type>: Self-supervised labels - observation based
    self.Ys_<dataset_type>: Labels - state based - multi-class
    self.Ys_bin_<dataset_type>: Labels - state based - binary
    self.unique_state_labels: Labels: what are the unique state labels in this data class
    self.tvt_exp_id: exp_ids that goes to train, validation, test sets, demonstration set in form of dictionary
        tvt_exp_id.csv contains all type of conditions

    self.eid_train <list>: experiment ids used for training
    self.eid_test: see above
    self.eid_val: see above
    self.eid_demo: see above + predefined ones
    self.scaler_X
    self.scaler_yo
    """

    def __init__(self, obs_col, cleaned_up_data,
                 min_number_of_sample,
                 state_label_col='multi_class_enc',
                 bin_state_label_col='binary_class_enc',
                 path_sq_label_data="/data/sequence_label_data",
                 sample_per_experiment=800,
                 window_length=100,
                 tvt_exp_id_fname="tvt_exp_id.csv",
                 train_test_val_port=0.7,
                 demo_id_list=[210, 346, 433],
                 shuffle_tvt=False,
                 normalize=True,
                 debug=True):

        self._logger_name = 'Sequence Label Data logger'
        self._dataset_types = ['train', 'test', 'val', 'demo']
        self._tvt_exp_id = {}  # experiment id, keys are the dataset types
        self._pkl_names = {}
        self._debug = debug
        self._set_verbosity(debug)

        self._window_length = window_length

        self._cd_data = cleaned_up_data  # the cleaned up data
        self._obs_col = obs_col  # x columns
        self._state_label_col = state_label_col  # y columns
        self._bin_state_label_col = bin_state_label_col  # y columns in binary format
        self._path = os.getcwd() + path_sq_label_data
        self._spe = sample_per_experiment  # maximum numbers of samples per sequence of data, to reduce memory usage if needed, -1 if use everything

        self._tvt_fname = tvt_exp_id_fname
        self._shuffle_tvt = shuffle_tvt
        self._demo_id = demo_id_list
        self.exp_id_array = np.array(self._cd_data.exp_id_list)  # needs to be numpy arrays to process

        self._normalize = normalize  # whether to normalize

        self._train_portion = train_test_val_port
        self._min_number_of_sample = min_number_of_sample

        self._report_df = pd.DataFrame(index=[0])

    def process(self):
        self._main_processing()  # process everything

    def _main_processing(self):
        self._logger.critical("Chopping dataset to trainable feature and label pairs...")
        self._generate_pkl_name()  # generate the pkl name to save to

        try:  # try read pickle
            self._logger.critical("Found previously processed feature label pairs, reading...")
            self._read_pkl_for_all_dataset_type()
            self._logger.critical("Reading completed.")
        except:
            self._identify_exp_id()  # identify the exp_ids to divide train/validation/test set from tvt_exp_id.csv file
            self._logger.critical("No/failed reading previously processed feature label pairs, (re)processing...")
            self._sequence_pair_conversion()
            self._logger.critical("Processing completed.")

    def _assert_number_of_samples(self, dataset_length, type_of_data):
        if type_of_data != 'demo':  # no need to assert demo, it doesn't have NN significance
            assert dataset_length >= self._min_number_of_sample[type_of_data], "Number of samples does not satisfy " \
                                                                               "minimal number requirements: {} vs {} " \
                                                                               "samples, try increase sample per experiment.".format(
                dataset_length, self._min_number_of_sample[type_of_data])
            pass

    def _generate_pkl_name(self):
        """
        Generates the .pkl file name to identify read/write data file
        """
        self._logger.critical("Generating the pkl file names to read/write...")

        self._array_pkl_name = "spe_" + str(self._spe) + "_window_length_" + str(self._window_length) + ".pkl"

        for type_of_data in self._dataset_types:
            Xpkl_name = self._path + '/' + type_of_data + "_X_" + self._array_pkl_name
            Yopkl_name = self._path + '/' + type_of_data + "_Yo_" + self._array_pkl_name
            Yspkl_name = self._path + '/' + type_of_data + "_Ys_" + self._array_pkl_name
            Yspkl_name_bin = self._path + '/' + type_of_data + "_Ys_bin_" + self._array_pkl_name

            scalerX_pkl_name = self._path + '/' + type_of_data + "_scaler_X_" + self._array_pkl_name
            scalerYo_pkl_name = self._path + '/' + type_of_data + "_scaler_Yo_" + self._array_pkl_name

            self._pkl_names[type_of_data] = [Xpkl_name, Yopkl_name, Yspkl_name, Yspkl_name_bin,
                                             scalerX_pkl_name, scalerYo_pkl_name]

    def _write_pkl(self, X, Yo, Ys, Ys_bin, type_of_data):
        """
        write pickle
        """
        Xpkl_name, Yopkl_name, Yspkl_name, Yspkl_name_bin, scalerX_pkl_name, scalerYo_pkl_name = self._pkl_names[
            type_of_data]
        with open(Xpkl_name, 'wb') as f:
            pkl.dump(X, f)
        with open(Yopkl_name, 'wb') as f:
            pkl.dump(Yo, f)
        with open(Yspkl_name, 'wb') as f:
            pkl.dump(Ys, f)
        with open(Yspkl_name_bin, 'wb') as f:
            pkl.dump(Ys_bin, f)
        if type_of_data == 'train':
            with open(scalerX_pkl_name, 'wb') as f:
                pkl.dump(self.scaler_X, f)
            with open(scalerYo_pkl_name, 'wb') as f:
                pkl.dump(self.scaler_Yo, f)

    def _read_pkl(self, type_of_data):
        """
        read pickle
        """
        self._logger.debug("Loading " + type_of_data + " .pkls...")
        Xpkl_name, Yopkl_name, Yspkl_name, Yspkl_name_bin, scalerX_pkl_name, scalerYo_pkl_name = self._pkl_names[
            type_of_data]
        with open(Xpkl_name, 'rb') as f:
            X = pkl.load(f)
        with open(Yopkl_name, 'rb') as f:
            Yo = pkl.load(f)
        with open(Yspkl_name, 'rb') as f:
            Ys = pkl.load(f)
        with open(Yspkl_name_bin, 'rb') as f:
            Ys_bin = pkl.load(f)
        if type_of_data == 'train':
            with open(scalerX_pkl_name, 'rb') as f:
                self.scaler_X = pkl.load(f)
            with open(scalerYo_pkl_name, 'rb') as f:
                self.scaler_Yo = pkl.load(f)

        self._logger.debug("Loaded " + type_of_data + " .pkls.")
        self._report_converted_data_info(X, Yo, Ys, Ys_bin, type_of_data)
        return X, Yo, Ys, Ys_bin

    def _read_pkl_for_all_dataset_type(self):
        """
        No return, but will assign the X,Yo,Ys values here
        """
        for type_of_data in self._dataset_types:
            if type_of_data == 'train':  # somewhat poor case structure, don't want to install py3.10 on multiple machines
                self.X_train, self.Yo_train, self.Ys_train, self.Ys_bin_train = self._read_pkl(type_of_data)
            elif type_of_data == 'test':
                self.X_test, self.Yo_test, self.Ys_test, self.Ys_bin_test = self._read_pkl(type_of_data)
            elif type_of_data == 'val':
                self.X_val, self.Yo_val, self.Ys_val, self.Ys_bin_val = self._read_pkl(type_of_data)
            elif type_of_data == 'demo':
                self.X_demo, self.Yo_demo, self.Ys_demo, self.Ys_bin_demo = self._read_pkl(type_of_data)

    def _random_sample(self, max_idx, type_of_data):
        """
        Within an exp id, randomly sample
        max_idx is the last sample-able starting index of the experiment
        If max_idx > sample per experiment, we will sample #number of subsequences from the experiment
        Otherwise we will keep sample all subsequences from the experiment
        """
        rng = default_rng()

        # For the demo dataset we will keep the original sequence
        # if spe == -1, means use maximum data
        if self._spe != -1 and type_of_data != 'demo' and max_idx > self._spe:
            start_idx_list = rng.choice(max_idx, size=self._spe, replace=False)
            start_idx_list = sorted(start_idx_list)
        else:
            start_idx_list = range(max_idx)

        return start_idx_list

    def _sequence_pair_conversion(self):
        self._logger.debug("Parsing experiments base on train/val/test/demo.")
        self.X_train, self.Yo_train, self.Ys_train, self.Ys_bin_train = self._sequence_pair_conversion_per_dataset(
            'train', self.eid_train)
        self.X_test, self.Yo_test, self.Ys_test, self.Ys_bin_test = self._sequence_pair_conversion_per_dataset('test',
                                                                                                               self.eid_test)
        self.X_val, self.Yo_val, self.Ys_val, self.Ys_bin_val = self._sequence_pair_conversion_per_dataset('val',
                                                                                                           self.eid_val)
        self.X_demo, self.Yo_demo, self.Ys_demo, self.Ys_bin_demo = self._sequence_pair_conversion_per_dataset('demo',
                                                                                                               self.eid_demo)

    def _sequence_pair_conversion_per_dataset(self, type_of_data, eid):
        """
        :param type_of_data: train, val, test, demo
        :param eid: the self.eid_train, etc ..
        :return: the np array X,Yo,Ys
        Additionally, generate and save the pkls
        """
        X, Yo, Ys, Ys_bin = [], [], [], []
        for exp_id in eid[:]:
            # Group by experiment ID
            exp_id = int(float(exp_id))
            df_in_exp = self._cd_data.df.loc[self._cd_data.df['Exp.ID'] == exp_id]

            self._logger.debug("Processing experiment {}, Experiment length: {}".format(exp_id, len(df_in_exp)))

            # Generate the required 'starting idx'
            start_idx_list = self._random_sample(len(df_in_exp) - self._window_length, type_of_data=type_of_data)
            self._get_data_based_on_starting_idx(df_in_exp, start_idx_list, type_of_data, X, Yo, Ys,
                                                 Ys_bin)  # pass by reference

        # make sure number of samples are satisfying
        self._assert_number_of_samples(len(Ys), type_of_data)

        # convert to np array
        X = np.array(X)
        Yo = np.array(Yo)
        Ys = np.array(Ys).reshape((-1, 1))
        Ys_bin = np.array(Ys_bin).reshape((-1, 1))

        if self._debug:
            self._report_converted_data_info(X, Yo, Ys, Ys_bin, type_of_data)  # report info first before normalization

        # normalize data
        if self._normalize:
            self._logger.critical("Normalizing {} data...".format(type_of_data))
            if type_of_data == 'train':  # needs to record scaler if training set
                X, Yo, self.scaler_X, self.scaler_Yo = self._normalize_numpy_and_return_scaler(X, Yo)
            else:
                X, Yo = self._normalize_numpy_with_scaler(X, Yo)

        # shuffle all data
        X, Yo, Ys, Ys_bin = self._shuffle_all_data(X, Yo, Ys, Ys_bin, type_of_data)

        # save these np arrays into pickles and report information
        self._write_pkl(X, Yo, Ys, Ys_bin, type_of_data)
        self._report_converted_data_info(X, Yo, Ys, Ys_bin, type_of_data)
        return X, Yo, Ys, Ys_bin

    def _shuffle_all_data(self, X, Yo, Ys, Ys_bin, type_of_data):
        if type_of_data != 'demo':
            self._logger.critical("Shuffling processed sl_data...")
            idx = np.random.permutation(len(Ys_bin))
            X = X[idx]
            Yo = Yo[idx]
            Ys = Ys[idx]
            Ys_bin = Ys_bin[idx]
            self._logger.critical("Shuffling completed.")
        return X, Yo, Ys, Ys_bin

    def _get_data_based_on_starting_idx(self, df_in_exp, start_idx_list, type_of_data, X, Yo, Ys,
                                        Ys_bin):  # pass by reference
        """
        This function will ONLY allow data to go into X, Yo, Ys if it's in self._train_mode.
        In this (default) case, type_of_data is unused, but in SequenceLabelExclusive or other subclass, this is used
        """
        for start_idx in start_idx_list:
            observations_past = df_in_exp[self._obs_col].iloc[start_idx:(start_idx + self._window_length)]
            observation_new = df_in_exp[self._obs_col].iloc[start_idx + self._window_length]
            state_new = df_in_exp[self._state_label_col].iloc[start_idx + self._window_length]
            bin_state_new = df_in_exp[self._bin_state_label_col].iloc[start_idx + self._window_length]
            X.append(observations_past)
            Yo.append(observation_new)
            Ys.append(state_new)
            Ys_bin.append(bin_state_new)

    def _normalize_numpy_and_return_scaler(self, X, Yo):
        X, scaler_X = normalize_numpy_and_return_scaler(X)
        self._logger.info("Training data feature means: {}".format(scaler_X.mean_))
        Yo, scaler_Yo = normalize_numpy_and_return_scaler(Yo)
        self._logger.info("Training data observation means: {}".format(scaler_Yo.mean_))
        return X, Yo, scaler_X, scaler_Yo

    def _normalize_numpy_with_scaler(self, X, Yo):
        X = normalize_numpy_with_scaler(X, self.scaler_X)
        Yo = normalize_numpy_with_scaler(Yo, self.scaler_Yo)
        return X, Yo

    def _report_converted_data_info(self, X, Yo, Ys, Ys_bin, type_of_data):
        self._report_meta_data(X, Yo, Ys, Ys_bin, type_of_data)
        self._report_unique_data(Ys, Ys_bin, type_of_data)

    def _report_unique_data(self, Ys, Ys_bin, type_of_data):
        # Report unique labels for multi-class
        unique_state_labels, unique_state_labels_counts = np.unique(Ys, return_counts=True)
        self._logger.critical(
            "Checking unique multi-class labels for {} data, there are {} labels, each has {} counts".format(
                type_of_data, unique_state_labels, unique_state_labels_counts))

        # Record into dataframe
        self._write_loaded_data_into_report_df(multi_class_labels=unique_state_labels,
                                               multi_class_label_counts=unique_state_labels_counts,
                                               type_of_data=type_of_data,
                                               class_type='Multi')


        # Report unique labels for binary-class
        unique_state_labels, unique_state_labels_counts = np.unique(Ys_bin, return_counts=True)
        self._logger.critical(
            "Checking unique binary state labels for {} data, there are {} labels, each has {} counts".format(
                type_of_data, unique_state_labels, unique_state_labels_counts))

        # Record into dataframe
        self._write_loaded_data_into_report_df(multi_class_labels=unique_state_labels,
                                               multi_class_label_counts=unique_state_labels_counts,
                                               type_of_data=type_of_data,
                                               class_type='Binary')

    def _write_loaded_data_into_report_df(self, multi_class_labels, multi_class_label_counts, type_of_data, class_type):
        for i, label in enumerate(multi_class_labels):
            count = multi_class_label_counts[i]
            name = '{} Class {} Count ({})'.format(class_type, label, type_of_data.capitalize())
            self._report_df[name] = count

    def _report_meta_data(self, X, Yo, Ys, Ys_bin, type_of_data):
        self._logger.critical("Meta-Data of " + type_of_data + " set:")
        self._logger.critical("Shape of feature [number of sequence, window length, dimension]:{}".format(X.shape))
        self._logger.critical("Shape of obs label [number of sequence, dimension]:{}".format(Yo.shape))
        self._logger.critical("Shape of state label [number of sequence, dimension]:{}".format(Ys.shape))
        self._logger.critical("Shape of binary state label [number of sequence, dimension]:{}".format(Ys_bin.shape))
        self._logger.debug(
            "Checking feature, observation and state label info of the first sample:\nfeature:{},\nobservation label:{},"
            "\nstate label:{},\nbinary state label:{}".format(X[0, -1, :], Yo[0, :], Ys[0, :], Ys_bin[0, :]))

    def _set_verbosity(self, debug):
        logging.basicConfig()
        self._logger = logging.getLogger(self._logger_name)
        self._debug = debug
        if self._debug:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.CRITICAL)

    def _identify_exp_id(self):
        """
        identify the exp_ids to divide train/validation/test set from tvt_exp_id.csv file
        in the .csv file, the rows are train, validation, test, and demo(subset of test)
        if the shuffle_tvt is set to TRUE, we will reshuffle and overwrite the original .csv file
        RETURN: dict containing train:[]; validation:[]; test: []; demo: []
        """

        def check_and_append_id(exp_ids):
            # when reading .csv there might be columns that are empty "", want to remove them
            id_list = []
            for exp_id in exp_ids:  # remove empty label names ("")
                if exp_id:
                    id_list.append(exp_id)
            return id_list

        def write_exp_id_to_list_and_csv(type_of_data, np_data, csv_w):
            """
            write the generated experiment id into csv
            """
            list_np_data = list(np_data)
            self._tvt_exp_id[type_of_data] = list_np_data  # convert numpy array to list
            csv_w.writerow(list_np_data, )

        cwd = os.getcwd()  # get current working directory
        fname = cwd + "/data/" + self._tvt_fname  # read valid column name from RoSA repo
        self._logger.critical("Parsing exp_id...")
        self._logger.debug("TVT Exp ID file: {}".format(fname))

        number_of_experiments = len(self.exp_id_array)
        self._logger.debug("Number of experiments: {}".format(number_of_experiments))

        if self._shuffle_tvt:  # If we want to shuffle the train/validation/test split
            # Train test split the experiment ids first
            self.eid_train, _eid_test_val = train_test_split(self.exp_id_array,
                                                             train_size=self._train_portion)  # , random_state=42)
            self.eid_test, self.eid_val = train_test_split(_eid_test_val, train_size=0.5)  # , random_state=42)
            self.eid_demo = np.array([])

            # Remove demo experiments from train/val
            for exp_id in self._demo_id:
                self.eid_train = np.delete(self.eid_train, np.argwhere(
                    self.eid_train == exp_id))  # remove exp id used for demo from training set
                self.eid_val = np.delete(self.eid_val, np.argwhere(
                    self.eid_val == exp_id))  # remove exp id used for demo from validation set
                self.eid_test = np.append(self.eid_test, exp_id)
                self.eid_demo = np.append(self.eid_demo, exp_id)

            self.eid_test = np.unique(self.eid_test)  # Make sure no duplicated experiment id is added

            self._logger.debug("Number of experiments in training-{} ;testing-{}; validation-{}".format(
                len(self.eid_train), len(self.eid_test), len(self.eid_val)))

            # This has no use right?
            # eid_dummy = np.concatenate((self.eid_train, self.eid_val, self.eid_test))  # Make sure all exp id are there

            with open(fname, 'w', encoding='utf-8', newline='') as csv_file:  # write the exp id into the csv file
                self._logger.critical("Writing to tvt file {}".format(fname))
                csv_writer = csv.writer(csv_file)
                write_exp_id_to_list_and_csv('train', self.eid_train, csv_writer)
                write_exp_id_to_list_and_csv('val', self.eid_val, csv_writer)
                write_exp_id_to_list_and_csv('test', self.eid_test, csv_writer)
                write_exp_id_to_list_and_csv('demo', self.eid_demo, csv_writer)

        else:  # if we don't need to shuffle, we will read whatever that's already processed
            with open(fname, encoding='utf-8') as csv_file:
                self._logger.critical("Reading from tvt file {}".format(fname))
                csv_reader = csv.reader(csv_file)
                for row, exp_ids in enumerate(csv_reader):
                    if row == 0:
                        self._logger.debug("Parsing training exp id...")
                        self._tvt_exp_id['train'] = check_and_append_id(exp_ids)
                    elif row == 1:
                        self._logger.debug("Parsing validation exp id...")
                        self._tvt_exp_id['val'] = check_and_append_id(exp_ids)
                    elif row == 2:
                        self._logger.debug("Parsing testing exp id...")
                        self._tvt_exp_id['test'] = check_and_append_id(exp_ids)
                    elif row == 3:
                        self._logger.debug("Parsing demo exp id...")
                        self._tvt_exp_id['demo'] = check_and_append_id(exp_ids)
        self.eid_train = self._tvt_exp_id['train']
        self.eid_val = self._tvt_exp_id['val']
        self.eid_test = self._tvt_exp_id['test']
        self.eid_demo = self._tvt_exp_id['demo']
        self._logger.debug("TVT Exp ID info: {}".format(self._tvt_exp_id))

    @property
    def report_df(self):
        df = self._report_df.add_prefix('Loaded Data - ')
        return df

class BalancedSequenceLabelData(SequenceLabelData):
    """
    This will balance the number of samples in each mode
    (NEW) balance_mode: 'multi' vs 'binary'
    """

    def __init__(self,
                 balance_mode,
                 obs_col, cleaned_up_data,
                 min_number_of_sample,
                 state_label_col='multi_class_enc',
                 bin_state_label_col='binary_class_enc',
                 path_sq_label_data="/data/sequence_label_data",
                 sample_per_experiment=800,
                 window_length=100,
                 tvt_exp_id_fname="tvt_exp_id.csv",
                 demo_id_list=[210, 346, 433],
                 train_test_val_port=0.7,
                 shuffle_tvt=False,
                 normalize=True,
                 debug=True,
                 ):
        # Init from super class
        super(BalancedSequenceLabelData, self).__init__(window_length=window_length,
                                                        cleaned_up_data=cleaned_up_data,
                                                        path_sq_label_data=path_sq_label_data,
                                                        min_number_of_sample=min_number_of_sample,
                                                        sample_per_experiment=sample_per_experiment,
                                                        tvt_exp_id_fname=tvt_exp_id_fname,
                                                        demo_id_list=demo_id_list,
                                                        train_test_val_port=train_test_val_port,
                                                        normalize=normalize,
                                                        obs_col=obs_col,
                                                        state_label_col=state_label_col,
                                                        bin_state_label_col=bin_state_label_col,
                                                        shuffle_tvt=shuffle_tvt,
                                                        debug=debug)

        # Overwrite Default initialization
        self._logger_name = 'Balanced Sequence Label Data logger'
        self._set_verbosity(debug)

        # (NEW) balance mode, 'multi' vs 'binary'
        self.balance_mode = balance_mode
        self._balance_number_of_samples = {}  # it will record the maximum number of samples to keep for each type_of_data
        self._idx_to_keep = {}  # it will store the index to keep for each type_of_data

    def process(self):
        self._main_processing()  # inside, it will propagate self._balance_number_of_samples
        self._balance_datasets()

    def _balance_datasets(self):
        self._logger.debug("Original number of samples {}".format(self._balance_number_of_samples))
        for type_of_data in ['train', 'val', 'test']:
            self._identify_idx_to_keep_for_type_of_data(type_of_data)
        self._keep_idx_to_keep()
        self._report_balanced_data()

    def _report_unique_data(self, Ys, Ys_bin, type_of_data):
        """
        This is called in _main_processing()
        """
        _balance_number_of_samples = {}  # local dictionary

        # Report unique labels for multi-class
        unique_state_labels, unique_state_labels_counts = np.unique(Ys, return_counts=True)
        _balance_number_of_samples['multi'] = (unique_state_labels.tolist(), unique_state_labels_counts)
        self._logger.critical(
            "Checking unique multi-class labels for {} data, there are {} labels, each has {} counts".format(
                type_of_data, unique_state_labels, unique_state_labels_counts))

        # Record into dataframe
        self._write_loaded_data_into_report_df(multi_class_labels=unique_state_labels,
                                               multi_class_label_counts=unique_state_labels_counts,
                                               type_of_data=type_of_data,
                                               class_type='Multi')

        # Report unique labels for binary-class
        unique_state_labels, unique_state_labels_counts = np.unique(Ys_bin, return_counts=True)
        _balance_number_of_samples['binary'] = (unique_state_labels.tolist(), unique_state_labels_counts)
        self._logger.critical(
            "Checking unique binary state labels for {} data, there are {} labels, each has {} counts".format(
                type_of_data, unique_state_labels, unique_state_labels_counts))

        # Record into dataframe
        self._write_loaded_data_into_report_df(multi_class_labels=unique_state_labels,
                                               multi_class_label_counts=unique_state_labels_counts,
                                               type_of_data=type_of_data,
                                               class_type='Binary')

        if type_of_data != 'demo':  # We don't need to balance demo set
            self._balance_number_of_samples[type_of_data] = _balance_number_of_samples

    def _report_balanced_data(self):
        self._report_unique_data(self.Ys_train, self.Ys_bin_train, 'train')
        self._report_unique_data(self.Ys_val, self.Ys_bin_val, 'val')
        self._report_unique_data(self.Ys_test, self.Ys_bin_test, 'test')

    def _identify_idx_to_keep_for_type_of_data(self, type_of_data):
        self._logger.critical("Processing {} data with mode: {}...".format(type_of_data, self.balance_mode))
        # Parse type of data
        if type_of_data == 'train':
            Ys = self.Ys_train
        elif type_of_data == 'val':
            Ys = self.Ys_val
        elif type_of_data == 'test':
            Ys = self.Ys_test

        # Parse unique labels and how many samples to get for each label, we will always use 'multi' label as baseline
        unique_labels, unique_number_of_samples = self._balance_number_of_samples[type_of_data]['multi']

        # Shuffle the index to remove dependency of exp_id sequence
        sample_idx = list(range(Ys.shape[0]))
        random.shuffle(sample_idx)

        # Initialize a list of samples to keep
        label_counter = defaultdict(lambda: 0)
        idx_to_keep = []

        # Adjust portion
        label_counter_target = self._adjust_portion_based_on_type_of_data(type_of_data=type_of_data,
                                                                          unique_labels=unique_labels,
                                                                          unique_number_of_samples=unique_number_of_samples)

        # Top off or cut off healthy mode in case we choose 'binary' balance mode
        label_counter_target = self._top_off_or_cut_off_healthy_mode(label_counter_target=label_counter_target,
                                                                     type_of_data=type_of_data)

        # Go through the idx and record the idx of samples to be kept for each sample
        for idx in sample_idx:
            multi_class_label = int(Ys[idx])
            if label_counter[multi_class_label] < label_counter_target[multi_class_label]:
                idx_to_keep.append(idx)
                label_counter[multi_class_label] += 1

        # Store idx to keep
        self._idx_to_keep[type_of_data] = idx_to_keep

    def _keep_idx_to_keep(self):
        idx_to_keep = self._idx_to_keep['train']
        self.X_train = self.X_train[idx_to_keep]
        self.Ys_train = self.Ys_train[idx_to_keep]
        self.Ys_bin_train = self.Ys_bin_train[idx_to_keep]
        self.Yo_train = self.Yo_train[idx_to_keep]

        idx_to_keep = self._idx_to_keep['test']
        self.X_test = self.X_test[idx_to_keep]
        self.Ys_test = self.Ys_test[idx_to_keep]
        self.Ys_bin_test = self.Ys_bin_test[idx_to_keep]
        self.Yo_test = self.Yo_test[idx_to_keep]

        idx_to_keep = self._idx_to_keep['val']
        self.X_val = self.X_val[idx_to_keep]
        self.Ys_val = self.Ys_val[idx_to_keep]
        self.Ys_bin_val = self.Ys_bin_val[idx_to_keep]
        self.Yo_val = self.Yo_val[idx_to_keep]

    def _top_off_or_cut_off_healthy_mode(self, label_counter_target, type_of_data):
        if self.balance_mode == 'binary':
            num_of_sample_faulty_mode = 0
            # We want to count the number of samples that are not healthy
            for multi_class_label, num_of_samples in label_counter_target.items():
                if multi_class_label != 0:
                    num_of_sample_faulty_mode += num_of_samples
            label_counter_target[0] = num_of_sample_faulty_mode
        return label_counter_target

    def _adjust_portion_based_on_type_of_data(self, type_of_data, unique_labels, unique_number_of_samples):
        return self._initialize_default_dict(keys=unique_labels, default_value=min(unique_number_of_samples))

    @staticmethod
    def _initialize_default_dict(keys, default_value):
        # We cannot use the default dict from collection because it will spits out error when enumerating the dict
        dic = {}
        for key in keys:
            dic[key] = default_value
        return dic


class ArbitraryPortionSequenceLabelData(BalancedSequenceLabelData):
    """
    This class defines arbitrary portion of data to keep in the training set
    It only applies for multi-class classification
    (NEW) arg: val_class_portion = {'H': 1, 'NV': 0, 'D': 0.5 (MAYBE remove this)
    (NEW) arg: train_class_number = {'H': 1000, 'D': 0, 'NV': 100, 'N': 100}: this will overwrite the per_class_sample
          1000: we will try to overwrite the label_counter_target
    (NEW) arg: greedy_balance: 'intact' vs 'non-zero', see explanation:
          We will balance the dataset before adjusting the portion, if greedy_balance is 'non-zero', we will balance to
          the minimal number of samples that is non-zero, if 'intact', we will balance to the number of samples whose
          class is not specified in train_val_class_portion.
    """

    def __init__(self,
                 obs_col, cleaned_up_data,
                 min_number_of_sample,
                 multi_class_map,
                 train_class_number={'H': 100, 'N': 100, 'D': 100, 'NV': 100},
                 balance_mode='multi',
                 greedy_balance='intact',
                 state_label_col='multi_class_enc',
                 bin_state_label_col='binary_class_enc',
                 path_sq_label_data="/data/sequence_label_data",
                 sample_per_experiment=800,
                 window_length=100,
                 tvt_exp_id_fname="tvt_exp_id.csv",
                 demo_id_list=[210, 346, 433],
                 train_test_val_port=0.7,
                 shuffle_tvt=False,
                 normalize=True,
                 debug=True):
        super(ArbitraryPortionSequenceLabelData, self).__init__(balance_mode=balance_mode,
                                                                window_length=window_length,
                                                                cleaned_up_data=cleaned_up_data,
                                                                path_sq_label_data=path_sq_label_data,
                                                                min_number_of_sample=min_number_of_sample,
                                                                sample_per_experiment=sample_per_experiment,
                                                                tvt_exp_id_fname=tvt_exp_id_fname,
                                                                demo_id_list=demo_id_list,
                                                                train_test_val_port=train_test_val_port,
                                                                normalize=normalize,
                                                                obs_col=obs_col,
                                                                state_label_col=state_label_col,
                                                                bin_state_label_col=bin_state_label_col,
                                                                shuffle_tvt=shuffle_tvt,
                                                                debug=debug)

        # Overwrite Default initialization
        self._logger_name = 'Arbitrary Portion Sequence Label Data logger'
        self._set_verbosity(debug)

        # (NEW) portion and remap to encoded values
        self._train_class_number = DictMap(multi_class_map, train_class_number).map()
        self._val_class_portion = self._identify_val_class_portion()
        self._greedy_balance = greedy_balance  # see self._identify_as_many_balanced_number_of_samples()
        # End of New

    def _identify_val_class_portion(self):
        """
        New Method, based on self._train_class_number, make sure the _val_class_portion has the same distribution,
        such that the train/val has the same distribution
        """
        healthy_number_of_samples = 0
        val_class_portion = {}

        for multi_class_label, number_of_samples in self._train_class_number.items():
            assert number_of_samples >= 0, f'Number of samples for train_class_number needs to be geq than 0!'
            if multi_class_label == 0:  # Healthy class
                healthy_number_of_samples = number_of_samples
                val_class_portion[multi_class_label] = 1  # We will always anchor around the healthy mode
            else:
                val_class_portion[multi_class_label] = number_of_samples / healthy_number_of_samples
        return val_class_portion

    def _adjust_portion_based_on_type_of_data(self, type_of_data, unique_labels, unique_number_of_samples):
        """
        INHERITED method, NEW content:

        """
        self._logger.critical("Adjusting dataset portion...")

        if type_of_data == 'train':
            label_counter_target = self._identify_potential_train_number_of_samples(unique_labels=unique_labels,
                                                                                    unique_number_of_samples=unique_number_of_samples)
            # label_counter_target = potential_target_number_of_samples
        elif type_of_data == 'val':
            label_counter_target = self._identify_potential_val_number_of_samples(unique_labels=unique_labels,
                                                                                  unique_number_of_samples=unique_number_of_samples)

        else:  # We don't need to balance these for test/demo set
            label_counter_target = self._initialize_default_dict(keys=unique_labels,
                                                                 default_value=min(unique_number_of_samples))

        self._logger.critical("Adjusted number of samples for {} set.".format(type_of_data))
        return label_counter_target

    def _identify_potential_train_number_of_samples(self, unique_labels, unique_number_of_samples):
        """
        New Method: Overwrite the number of potential_target_number_of_samples based on self._train_class_number
        """
        potential_target_number_of_samples = {}

        for class_i, multi_class_label in enumerate(unique_labels):
            number_of_sample_for_class_i = unique_number_of_samples[class_i]
            target_number_of_sample_for_class_i = self._train_class_number[multi_class_label]
            assert target_number_of_sample_for_class_i < number_of_sample_for_class_i, f"You are asking for more data than available! Modify train_class_number!"
            potential_target_number_of_samples[multi_class_label] = target_number_of_sample_for_class_i
        return potential_target_number_of_samples

    def _identify_potential_val_number_of_samples(self, unique_labels, unique_number_of_samples):
        """
        NEW method: called by self._adjust_portion_based_on_type_of_data()
        We want to keep number of samples in the validation set consistent with the ratio in the training set
        """
        label_counter_target = {}
        min_number_of_samples = 0

        for anchor_class, _ in enumerate(unique_labels):  # Use class i as the anchor
            proposed_label_counter_target = {}
            proposal_is_valid = True
            min_number_of_samples_for_this_proposal = np.inf
            anchor_number_of_samples = unique_number_of_samples[anchor_class]  # Use num_of_sample for class i
            if self._val_class_portion[anchor_class] == 0:
                continue
            for class_i, multi_class_label in enumerate(unique_labels):  # Adjust the other class
                number_of_sample_for_class_i = unique_number_of_samples[class_i]
                portion_for_class_i = self._val_class_portion[multi_class_label] / self._val_class_portion[anchor_class]
                target_number_of_sample_for_class_i = int(portion_for_class_i * anchor_number_of_samples)
                if target_number_of_sample_for_class_i > number_of_sample_for_class_i:  # This proposal is invalid
                    proposal_is_valid = False
                    break
                proposed_label_counter_target[multi_class_label] = target_number_of_sample_for_class_i
                min_number_of_samples_for_this_proposal = min(min_number_of_samples_for_this_proposal,
                                                              target_number_of_sample_for_class_i)
            if min_number_of_samples_for_this_proposal >= min_number_of_samples and proposal_is_valid:
                label_counter_target = proposed_label_counter_target
        return label_counter_target

    def _top_off_or_cut_off_healthy_mode(self, label_counter_target, type_of_data):
        """
        We don't top off the train and validation set any more
        """
        if self.balance_mode == 'binary':
            if type_of_data != 'train' and type_of_data != 'val':
                num_of_sample_faulty_mode = 0
                # We want to count the number of samples that are not healthy
                for multi_class_label, num_of_samples in label_counter_target.items():
                    if multi_class_label != 0:
                        num_of_sample_faulty_mode += num_of_samples
                label_counter_target[0] = num_of_sample_faulty_mode
        return label_counter_target


class FewShotClassSequenceLabelData(ArbitraryPortionSequenceLabelData):
    """
    The difference is:
        1. During self.process(), it will call an extra function _record_data_sample_idx(type_of_data)
        2. It will fill in self.data_class_sample_idx = {}
        3. sample_idx will have {'meta-train':  {'0':<np.array>, '1':<np.array>, ...}, ..., 'meta-demo':, {'H, V, N, D}}
            i.e, a dict{dict{<np.array>}}
    """

    def __init__(self,
                 obs_col, cleaned_up_data,
                 min_number_of_sample,
                 multi_class_map,
                 meta_train_class_number={'H': 100, 'N': 100, 'D': 100, 'NV': 100},
                 balance_mode='multi',
                 greedy_balance='intact',
                 state_label_col='multi_class_enc',
                 bin_state_label_col='binary_class_enc',
                 path_sq_label_data="/data/sequence_label_data",
                 sample_per_experiment=800,
                 window_length=100,
                 tvt_exp_id_fname="tvt_exp_id.csv",
                 demo_id_list=[210, 346, 433],
                 train_test_val_port=0.7,
                 shuffle_tvt=False,
                 normalize=True,
                 debug=True):
        super(FewShotClassSequenceLabelData, self).__init__(balance_mode=balance_mode,
                                                            window_length=window_length,
                                                            cleaned_up_data=cleaned_up_data,
                                                            path_sq_label_data=path_sq_label_data,
                                                            min_number_of_sample=min_number_of_sample,
                                                            sample_per_experiment=sample_per_experiment,
                                                            multi_class_map=multi_class_map,
                                                            train_class_number=meta_train_class_number,
                                                            greedy_balance=greedy_balance,
                                                            tvt_exp_id_fname=tvt_exp_id_fname,
                                                            demo_id_list=demo_id_list,
                                                            train_test_val_port=train_test_val_port,
                                                            normalize=normalize,
                                                            obs_col=obs_col,
                                                            state_label_col=state_label_col,
                                                            bin_state_label_col=bin_state_label_col,
                                                            shuffle_tvt=shuffle_tvt,
                                                            debug=debug)

        # Overwrite Default initialization
        self._logger_name = 'Few Shot Sequence Label Data logger'
        self._set_verbosity(debug)

        # (NEW) A new dict{dict} to record the sample idx for train, val, test, demo for each class
        self.meta_class_sample_idx = {}
        self.meta_class_sample_idx_bin = {}  # binary version

    def process(self):
        self._main_processing()  # inside, it will propagate self._balance_number_of_samples
        self._balance_datasets()

        # (NEW)
        self._record_data_sample_idx()
        # Enf of NEW

    def _record_data_sample_idx(self):
        self._logger.critical("Recording meta data/class information for Few-Shot Learning...")
        self._record_data_sample_idx_per_type_of_data('train')
        self._record_data_sample_idx_per_type_of_data('val')
        self._record_data_sample_idx_per_type_of_data('test')

    def _record_data_sample_idx_per_type_of_data(self, type_of_data):
        """
        This function fills in self.meta_class_sample_idx_bin and self.meta_class_sample_idx
        """
        if type_of_data == 'train':
            Ys_bin = self.Ys_bin_train
            Ys = self.Ys_train
        elif type_of_data == 'val':
            Ys_bin = self.Ys_bin_val
            Ys = self.Ys_val
        elif type_of_data == 'test':
            Ys_bin = self.Ys_bin_test
            Ys = self.Ys_test
        elif type_of_data == 'demo':  # We will still record the dictionary for 'demo', but probably not using it
            Ys_bin = self.Ys_bin_demo
            Ys = self.Ys_demo

        # Init dictionaries
        per_type_of_data_dict = defaultdict(lambda: list)
        per_type_of_data_dict_bin = defaultdict(lambda: list)

        # Identify the sample idx based on multi-class labels
        multi_class_labels, count = self._balance_number_of_samples[type_of_data]['multi']
        for multi_class_label in multi_class_labels:
            multi_class_label_idx = np.where(Ys == multi_class_label)[0]
            np.random.shuffle(multi_class_label_idx)  # Probably don't need it
            per_type_of_data_dict[multi_class_label] = multi_class_label_idx

        # Identify the sample idx based on binary class labels
        bin_class_labels, count = self._balance_number_of_samples[type_of_data]['binary']
        for bin_class_label in bin_class_labels:
            bin_class_label_idx = np.where(Ys_bin == bin_class_label)[0]
            np.random.shuffle(bin_class_label_idx)
            per_type_of_data_dict_bin[bin_class_label] = bin_class_label_idx

        # Assign into dictionaries
        meta_type_of_data_name = 'meta-' + type_of_data
        self.meta_class_sample_idx[meta_type_of_data_name] = per_type_of_data_dict
        self.meta_class_sample_idx_bin[meta_type_of_data_name] = per_type_of_data_dict_bin

        # Assertion for debug purpose
        self._logger.debug("Asserting recorded meta-{} data...".format(type_of_data))
        self._logger.debug("Multi-class:")
        count_per_type_of_data = 0
        for multi_class_label, sample_idx in self.meta_class_sample_idx[meta_type_of_data_name].items():
            class_number_of_samples = len(sample_idx)
            self._logger.debug("{} class: {} samples.".format(multi_class_label, class_number_of_samples))
            count_per_type_of_data += class_number_of_samples
        assert count_per_type_of_data == len(Ys), "Some meta data missing!!"

        self._logger.debug("Binary-class:")
        count_per_type_of_data = 0
        for binary_class_label, sample_idx in self.meta_class_sample_idx_bin[meta_type_of_data_name].items():
            class_number_of_samples = len(sample_idx)
            self._logger.debug("{} class: {} samples.".format(binary_class_label, class_number_of_samples))
            count_per_type_of_data += class_number_of_samples
        assert count_per_type_of_data == len(Ys_bin), "Some meta data missing!!"


if __name__ == "__main__":  # Test
    os.chdir("..")

    SV = SharedVariable()  # from utility, load the default values for most operations
    path_rosa = SV.path_rosa  # it will find the path of the RoSA dataset
    latest_column_name = SV.latest_column_name
    class_labels = SV.raw_class_labels

    raw_data = RawData(class_labels=class_labels, path_dataset=path_rosa,
                       latest_column_name=latest_column_name, overwrite_data_file=False,
                       path_loaded_raw_data=SV.path_loaded_raw_data, debug=True)

    cd_data = CleanedUpData(raw_data=raw_data, class_map=SV.default_multi_class_map,
                            binary_class_map=SV.default_binary_class_map,
                            path_processed_data=SV.path_cleaned_up_data,
                            debug=False)  # Load data that is cleaned up

    obs_columns = cd_data.valid_column_info['feature'].copy()

    # Test cases
    test_window_length = 200
    test_spe = SV.default_spe  # Use -1 if you have a crazy amount of memory :), otherwise 200 for testing

    test_sl_data = 1
    test_balance_sl_data = 0
    test_arbitrary_portion_sl_data = 0
    test_few_shot_data = 0

    if test_sl_data:
        sl_data = SequenceLabelData(window_length=test_window_length,
                                    sample_per_experiment=test_spe,
                                    cleaned_up_data=cd_data,
                                    path_sq_label_data=SV.path_sq_label_data,
                                    min_number_of_sample=SV.min_number_of_sample,
                                    obs_col=obs_columns,
                                    shuffle_tvt=False,
                                    debug=True)
        sl_data.process()

    if test_balance_sl_data:
        balanced_sl_data = BalancedSequenceLabelData(balance_mode='binary',
                                                     window_length=test_window_length,
                                                     sample_per_experiment=test_spe,
                                                     cleaned_up_data=cd_data,
                                                     path_sq_label_data=SV.path_sq_label_data,
                                                     min_number_of_sample=SV.min_number_of_sample,
                                                     obs_col=obs_columns,
                                                     shuffle_tvt=False,
                                                     debug=True)
        balanced_sl_data.process()

    if test_arbitrary_portion_sl_data:
        ap_sl_data = ArbitraryPortionSequenceLabelData(balance_mode='multi',
                                                       window_length=test_window_length,
                                                       train_class_number={'H': 100, 'N': 1, 'NV': 1, 'D': 0},
                                                       multi_class_map=SV.default_multi_class_map,
                                                       sample_per_experiment=test_spe,
                                                       cleaned_up_data=cd_data,
                                                       path_sq_label_data=SV.path_sq_label_data,
                                                       min_number_of_sample=SV.min_number_of_sample,
                                                       obs_col=obs_columns,
                                                       shuffle_tvt=False,
                                                       debug=True)
        ap_sl_data.process()

    if test_few_shot_data:
        sl_data = FewShotClassSequenceLabelData(balance_mode='binary',
                                                window_length=test_window_length,
                                                meta_train_class_number={'H': 1000, 'N': 1234, 'NV': 1234, 'D': 1000},
                                                multi_class_map=SV.default_multi_class_map,
                                                sample_per_experiment=test_spe,
                                                cleaned_up_data=cd_data,
                                                path_sq_label_data=SV.path_sq_label_data,
                                                min_number_of_sample=SV.min_number_of_sample,
                                                obs_col=obs_columns,
                                                shuffle_tvt=False,
                                                debug=True)
        sl_data.process()
