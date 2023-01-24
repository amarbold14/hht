from shared.MyData import *
from shared.utility import *
from shared.NN_util import *
import torch


class MyDataLoader:
    """
    Combines most functions to load data from RoSA
    """

    def __init__(self, debug=False, overwrite_data_file=False):
        os.chdir("..")

        self.SV = SharedVariable()
        self.raw_data = RawData(class_labels=self.SV.raw_class_labels,
                                path_dataset=self.SV.path_rosa,
                                latest_column_name=self.SV.latest_column_name,
                                overwrite_data_file=overwrite_data_file,
                                path_loaded_raw_data=self.SV.path_loaded_raw_data,
                                debug=debug)

        self.cd_data = CleanedUpData(raw_data=self.raw_data,
                                     class_map=self.SV.default_multi_class_map,
                                     binary_class_map=self.SV.default_binary_class_map,
                                     path_processed_data=self.SV.path_cleaned_up_data,
                                     debug=debug)

        self.obs_cols = self.cd_data.valid_column_info['feature'].copy()  # column names for features(observations)

        self._report_df = None  # Populates from MyData

    @property
    def report_df(self):
        return self._report_df


class FullShotClassDataLoader(MyDataLoader):
    def __init__(self, data_config, device, logger):
        super(FullShotClassDataLoader, self).__init__()
        self.data_config = data_config
        self.device = device
        self.logger = logger
        self._parse_data_config()

        # Fields not everybody uses
        self.data = None

    def _parse_data_config(self):
        # Parse data
        self.class_type = self.data_config.class_type
        self.window_length = self.data_config.window_length
        self.sample_per_experiment = self.data_config.spe
        self.balance_data = self.data_config.balance_data
        self.one_hot_encoding_mode = 'all'  # by default
        if self.data_config.one_hot_encode_mode:
            self.one_hot_encoding_mode = self.data_config.one_hot_encode_mode

    def _load_full_sequence_label_data(self):
        data = SequenceLabelData(window_length=self.window_length,
                                 sample_per_experiment=self.sample_per_experiment,
                                 cleaned_up_data=self.cd_data,
                                 path_sq_label_data=self.SV.path_sq_label_data,
                                 min_number_of_sample=self.SV.min_number_of_sample,
                                 obs_col=self.obs_cols,
                                 shuffle_tvt=False,
                                 debug=False)
        data.process()  # process data before returning it
        return data

    def _load_balanced_sequence_label_data(self):
        data = BalancedSequenceLabelData(balance_mode=self.class_type,
                                         window_length=self.window_length,
                                         sample_per_experiment=self.sample_per_experiment,
                                         cleaned_up_data=self.cd_data,
                                         path_sq_label_data=self.SV.path_sq_label_data,
                                         min_number_of_sample=self.SV.min_number_of_sample,
                                         obs_col=self.obs_cols,
                                         shuffle_tvt=False,
                                         debug=False)
        data.process()  # process data before returning it
        return data

    def _convert_single_np_array_to_tensor(self, X):
        """
        A generic function to convert numpy array to tensor on a device
        """
        X_tensor = torch.from_numpy(X).float().to(self.device)
        return X_tensor

    def _load_X_tensor_from_sl_data(self):
        X_train = self.data.X_train
        X_val = self.data.X_val
        X_test = self.data.X_test
        X_demo = self.data.X_demo

        X_train_tensor = self._convert_single_np_array_to_tensor(X_train)
        X_val_tensor = self._convert_single_np_array_to_tensor(X_val)
        X_test_tensor = self._convert_single_np_array_to_tensor(X_test)
        X_demo_tensor = self._convert_single_np_array_to_tensor(X_demo)
        return X_train_tensor, X_val_tensor, X_test_tensor, X_demo_tensor

    def _load_Y_from_sl_data(self, class_type):
        if class_type == 'multi':
            Y_train = self.data.Ys_train
            Y_val = self.data.Ys_val
        elif class_type == 'observation':
            Y_train = self.data.Yo_train
            Y_val = self.data.Yo_val
            Y_test = self.data.Yo_test
        elif class_type == 'binary':
            Y_train = self.data.Ys_bin_train
            Y_val = self.data.Ys_bin_val
        Ys_test = self.data.Ys_test
        Yo_test = self.data.Yo_test
        Ys_bin_test = self.data.Ys_bin_test
        Yo_demo = self.data.Yo_demo
        Ys_demo = self.data.Ys_demo
        Ys_bin_demo = self.data.Ys_bin_demo

        return Y_train, Y_val, Ys_test, Yo_test, Ys_bin_test, Yo_demo, Ys_demo, Ys_bin_demo

    def _process_and_convert_Y_to_tensor(self, Y_train, Y_val, Ys_test, Yo_test, Ys_bin_test, Ys_demo, Yo_demo,
                                         Ys_bin_demo, class_type, one_hot_encoding_mode):
        # Tiny bit of extra processing - to invoke one hot encoding
        if class_type == 'multi':
            self.logger.critical("Processing data with one hot encoding...")
            Y_train, oh_enc = one_hot_encoding(Y_train, mode=one_hot_encoding_mode, logger=self.logger)
            Y_val, oh_enc = one_hot_encoding(Y_val, mode=one_hot_encoding_mode, logger=self.logger)
            Ys_test, oh_enc = one_hot_encoding(Ys_test, mode=one_hot_encoding_mode, logger=self.logger)
            Ys_demo, oh_enc = one_hot_encoding(Ys_demo, mode=one_hot_encoding_mode, logger=self.logger)

        Y_train_tensor = self._convert_single_np_array_to_tensor(Y_train)
        Y_val_tensor = self._convert_single_np_array_to_tensor(Y_val)
        Ys_test_tensor = self._convert_single_np_array_to_tensor(Ys_test)
        Yo_test_tensor = self._convert_single_np_array_to_tensor(Yo_test)
        Ys_bin_test_tensor = self._convert_single_np_array_to_tensor(Ys_bin_test)
        Yo_demo_tensor = self._convert_single_np_array_to_tensor(Yo_demo)
        Ys_demo_tensor = self._convert_single_np_array_to_tensor(Ys_demo)
        Ys_bin_demo_tensor = self._convert_single_np_array_to_tensor(Ys_bin_demo)

        return Y_train_tensor, Y_val_tensor, Ys_test_tensor, Yo_test_tensor, Ys_bin_test_tensor, Ys_demo_tensor, Yo_demo_tensor, Ys_bin_demo_tensor

    def _extract_training_data_info(self, X, Y):
        """
        Fill in the number of samples, input dimension, sequence length and output dimension field
        """
        self.n_train_samples = X.shape[0]
        self.input_dimension = X.shape[-1]  # dim = 23
        self.sequence_len = X.shape[1]  # by default - dim = 50 or 100
        self.output_dimension = Y.shape[-1]  # one hot encoding, dim = 6

    def _get_training_data_info(self):
        self.logger.critical("Running on device {}.".format(self.device))
        self.logger.critical("X train dimension - {}".format(
            [self.n_train_samples, self.sequence_len, self.input_dimension]))  # N, L, D_in
        self.logger.critical("Y train dimension - {}".format([self.n_train_samples, self.output_dimension]))  # N, D_out
        return self.input_dimension, self.sequence_len, self.output_dimension

    def load_data(self):
        """
        Process Sequence Label data to Tensors for training
        Args:
            self.data_config
            self.class_type: 'multi', 'binary', 'regression'
            self.window_length: length of sequence
            self.sample_per_experiment: when parsing data, how many sample to get per experimental sequence
            self.balance_data: True by default, will invoke data-balancing
        Returns:
            All necessary data for training
        """
        # Load correct data based on DataLoader Class
        self._load_class_specific_data()

        # Load X data
        X_train_tensor, X_val_tensor, X_test_tensor, X_demo_tensor = self._load_X_tensor_from_sl_data()

        # Load Y data
        Y_train, Y_val, Ys_test, Yo_test, Ys_bin_test, Yo_demo, Ys_demo, Ys_bin_demo = self._load_Y_from_sl_data(
            self.class_type)

        # Extra processing of Y data - one hot encoding and turn to tensor
        Y_train_tensor, Y_val_tensor, Ys_test_tensor, Yo_test_tensor, Ys_bin_test_tensor, Ys_demo_tensor, Yo_demo_tensor, Ys_bin_demo_tensor = self._process_and_convert_Y_to_tensor(
            Y_train=Y_train, Y_val=Y_val, Ys_test=Ys_test, Yo_test=Yo_test, Ys_bin_test=Ys_bin_test, Ys_demo=Ys_demo,
            Ys_bin_demo=Ys_bin_demo, Yo_demo=Yo_demo, class_type=self.class_type,
            one_hot_encoding_mode=self.one_hot_encoding_mode)

        # Get data information
        self._extract_training_data_info(X_train_tensor, Y_train_tensor)
        self.logger.critical("Data loaded into data_loader.")

        # Parse into tuples and return
        return (X_train_tensor, Y_train_tensor), (X_val_tensor, Y_val_tensor), (
            X_test_tensor, Ys_test_tensor, Yo_test_tensor, Ys_bin_test_tensor), (
                   X_demo_tensor, Ys_demo_tensor, Yo_demo_tensor, Ys_bin_demo_tensor)

    def _load_class_specific_data(self):
        if self.balance_data:
            self.data = self._load_balanced_sequence_label_data()
        else:
            self.data = self._load_full_sequence_label_data()
        del self.cd_data, self.raw_data # free up memory

        self._report_df = self.data.report_df


class ArbitraryPortionDataLoader(FullShotClassDataLoader):
    def __init__(self, data_config, device, logger):
        super(ArbitraryPortionDataLoader, self).__init__(data_config=data_config,
                                                         device=device,
                                                         logger=logger)

        self.train_class_number = self.data_config.train_class_number

    def _load_class_specific_data(self):
        self.data = self._load_arbitrary_portion_number_sequence_label_data()
        del self.cd_data, self.raw_data

        self._report_df = self.data.report_df

    def _load_arbitrary_portion_number_sequence_label_data(self):
        data = ArbitraryPortionSequenceLabelData(obs_col=self.obs_cols,
                                                 window_length=self.window_length,
                                                 cleaned_up_data=self.cd_data,
                                                 train_class_number=self.train_class_number,
                                                 multi_class_map=self.SV.default_multi_class_map,
                                                 sample_per_experiment=self.sample_per_experiment,
                                                 balance_mode=self.class_type,
                                                 min_number_of_sample=self.SV.min_number_of_sample,
                                                 debug=False
                                                 )
        data.process()
        return data


class NwayKshotDataLoader(ArbitraryPortionDataLoader):
    """
    Member:
        self.data will be deleted to reduce memory consumption
    Member Function:
        (NEW) load_task_data(): this will be called during training
    """

    def __init__(self, data_config, logger, device):
        super(NwayKshotDataLoader, self).__init__(data_config=data_config,
                                                  logger=logger, device=device)

        # (NEW) members
        _, self.meta_train_ways, self.meta_test_ways, self.N, _, self.k_support, self.k_query = self.data_config.get_public_member()
        self.meta_train_ways, self.meta_test_ways = self._convert_class_label_to_class_idx()

        # Initialize these public variables and populate them
        self.X_train, self.Y_train = None, None
        self.X_val, self.Y_val = None, None
        self.X_test, self.Ys_test, self.Yo_test, self.Ys_bin_test = None, None, None, None
        self.X_demo, self.Ys_demo, self.Yo_demo, self.Ys_bin_demo = None, None, None, None
        self.meta_class_sample_idx = None
        self._populate_data_into_class_members()  # populate them

        del self.data  # free up memory

        # Assert there are enough (>k shots) data per the N classes
        assert self._assert_sufficient_data_in_train_val(), f'Not enough data in train/validation set for number of shots needed!'

    def _populate_data_into_class_members(self):
        """
        Populates:
            self.X_train, Y_train
            self.X_val, Y_val
            self.X_test, Ys_test, Yo_test, Ys_bin_test
            self.X_demo, Ys_demo, Yo_demo, Ys_bin_demo
            self.meta_class_sample_idx
            self.meta_class_sample_idx_bin
        """
        train_data, val_data, test_data, demo_data = self.load_data()
        self.X_train, self.Y_train = train_data
        self.X_val, self.Y_val = val_data
        self.X_test, self.Ys_test, self.Yo_test, self.Ys_bin_test = test_data
        self.X_demo, self.Ys_demo, self.Yo_demo, self.Ys_bin_demo = demo_data

        if self.class_type == 'binary':
            self.meta_class_sample_idx = self.data.meta_class_sample_idx_bin
        elif self.class_type == 'multi':
            self.meta_class_sample_idx = self.data.meta_class_sample_idx

    def _convert_class_label_to_class_idx(self):
        """
        Convert self.meta_train_ways and self.meta_test_ways to index vs strings
        """
        meta_train_ways = util_convert_tuple_class_label_to_class_idx(class_label_tuple=self.meta_train_ways,
                                                                      class_type=self.class_type)
        meta_test_ways = util_convert_tuple_class_label_to_class_idx(class_label_tuple=self.meta_test_ways,
                                                                     class_type=self.class_type)
        return meta_train_ways, meta_test_ways

    def _assert_sufficient_data_in_train_val(self):
        """
        Assert we do have enough data in the train/val set for the support and query sets needed.
        Args:
            self.class_type
            self.meta_class_label_index
            self.SV - for default_binary_class_map
        """
        self.logger.debug("Asserting we have enough data for support and query...")
        types_of_data = ['train', 'val']
        target_number_of_sample_per_class = self.k_support + self.k_query

        for type_of_data in types_of_data:
            meta_type_of_data = 'meta-' + type_of_data
            self.logger.debug("Checking {} data...".format(meta_type_of_data))
            for class_label in self.meta_train_ways:
                number_of_sample_per_class = len(self.meta_class_sample_idx[meta_type_of_data][class_label])
                self.logger.debug(
                    "{} class has {} samples, we need {}.".format(class_label, number_of_sample_per_class,
                                                                  target_number_of_sample_per_class))
                if target_number_of_sample_per_class > number_of_sample_per_class:
                    return False
        return True

    def _load_class_specific_data(self):
        self.data = self._load_few_shot_sequence_label_data()
        del self.cd_data, self.raw_data  # free up memory

        self._report_df = self.data.report_df

    def _load_few_shot_sequence_label_data(self):
        data = FewShotClassSequenceLabelData(obs_col=self.obs_cols,
                                             window_length=self.window_length,
                                             cleaned_up_data=self.cd_data,
                                             meta_train_class_number=self.train_class_number,
                                             multi_class_map=self.SV.default_multi_class_map,
                                             sample_per_experiment=self.sample_per_experiment,
                                             balance_mode=self.class_type,
                                             min_number_of_sample=self.SV.min_number_of_sample,
                                             debug=False)
        data.process()
        return data

    def load_task_data(self, type_of_data, use_all_remain_data_for_test=True):
        """
        Args:
            self.class_type
            self.meta_class_sample_idx
            self.k_support
            self.k_query
            type_of_data: 'train', 'val', 'test'
            use_all_remain_data_for_test: <bool> whether during meta-test we will use all remaining data for testing,
                                                or the number of query will be the same as test and validation.
        Return:
            for 'train' and 'val': X_spt,Y_spt, X_qry, Y_qry
            for 'test' (and 'demo'): X, Y, Yo, Ys and their queries
        Logic:
            based on self.meta_class_sample_idx, randomly sample from the loaded data
        """
        meta_type_of_data = 'meta-' + type_of_data

        if type_of_data == 'train':
            support_idx, query_idx = self._sample_from_meta_X_ways(meta_type_of_data=meta_type_of_data)
            return self.X_train[support_idx], self.Y_train[support_idx], self.X_train[query_idx], self.Y_train[
                query_idx]
        elif type_of_data == 'val':
            support_idx, query_idx = self._sample_from_meta_X_ways(meta_type_of_data=meta_type_of_data)
            return self.X_val[support_idx], self.Y_val[support_idx], self.X_val[query_idx], self.Y_val[query_idx]
        elif type_of_data == 'test':
            support_idx, query_idx = self._sample_from_meta_X_ways(meta_type_of_data=meta_type_of_data,
                                                                   use_all_remain_data_for_test=use_all_remain_data_for_test)
            return self.X_test[support_idx], self.Ys_test[support_idx], self.Yo_test[support_idx], self.Ys_bin_test[
                support_idx], \
                   self.X_test[query_idx], self.Ys_test[query_idx], self.Yo_test[query_idx], self.Ys_bin_test[query_idx]

    def _sample_from_meta_X_ways(self, meta_type_of_data, use_all_remain_data_for_test=True):
        """
        A utility called by self.load_task_data()
        Args:
            meta_type_of_data: 'meta-train', 'meta-test', 'meta-val', 'meta-demo'...
            use_all_remain_data_for_test<boolean>: see self.load_task_data()
        Returns:
            support_idx, query_idx
        """
        # Initialize arrays
        total_k = self.k_support + self.k_query
        support_idx = np.array([])
        query_idx = np.array([])

        # Sample data from X ('train', 'test' or 'val') set
        for class_label in self.meta_train_ways:
            idx = self.meta_class_sample_idx[meta_type_of_data][class_label]
            if meta_type_of_data == 'meta-test' and use_all_remain_data_for_test:
                total_k = len(idx)  # for meta_test we will use k for support and the rest of all test set for query
            sampled_idx_for_class = np.random.choice(idx, size=total_k)
            support_idx = np.concatenate([support_idx, sampled_idx_for_class[:self.k_support]])
            query_idx = np.concatenate([query_idx, sampled_idx_for_class[self.k_support:]])

        return support_idx, query_idx
