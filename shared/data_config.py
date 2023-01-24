from shared.MyDataLoader import *
from shared.utility import *
from model.meta_train_config import *


class DataConfig:
    """
    A class to include data configurations
    class_type: 'multi', 'binary', 'observation'
    """

    def __init__(self, window_length, sample_per_experiment, class_type, batch_size=1, balance_data=True):
        # Used for all data
        self.window_length = window_length
        self.spe = sample_per_experiment
        self.class_type = class_type  # 'multi', 'binary', 'regression'
        self.batch_size = batch_size

        # Used for some data
        self.one_hot_encode_mode = None  # only used for one hot encoded classifications

        # If balance dataset:
        self.balance_data = balance_data

        # Init report_df
        self._report_df = self.init_report_df()

    def one_hot_encode(self, one_hot_encoding_mode):
        self.one_hot_encode_mode = one_hot_encoding_mode
        self._report_df['One-hot encoded'] = self.one_hot_encode_mode

    def __str__(self):
        name = self.concise_str()
        name += "_spe{}".format(self.spe)
        name += "_{}".format(self.class_type)
        name += "_bsize{}".format(self.batch_size)
        return name

    def concise_str(self):
        """
        Will be used for 'run'
        """
        name = ""
        name += "_len{}".format(self.window_length)
        name += "_baln{}".format(self.balance_data)
        if self.one_hot_encode_mode:  # not all training needs one hot encoding
            name += "_onehot{}".format(self.one_hot_encode_mode)
        return name

    def init_report_df(self):
        report_df_init = {'Classes': self.class_type,
                          'Window Length': self.window_length,
                          'Batch Size': self.batch_size,
                          'Balanced': self.balance_data,
                          'Sample Per Experiment': self.spe}
        report_df = pd.DataFrame(report_df_init, index=[0])
        return report_df

    @property
    def report_df(self):
        return self._report_df.add_prefix('Data - ')


class ArbitraryDataConfig(DataConfig):
    """
    balance_data is not used anywhere
    The test set for AP is always balanced
    NEW: train_val_class_portion, train_val_class_number
    """

    def __init__(self, window_length, sample_per_experiment, class_type,
                 train_class_number, batch_size=1, balance_data=True):
        super(ArbitraryDataConfig, self).__init__(window_length=window_length,
                                                  sample_per_experiment=sample_per_experiment,
                                                  class_type=class_type,
                                                  batch_size=batch_size,
                                                  balance_data=balance_data)
        # NEW
        self.train_class_number = train_class_number
        self.augment_report_df()

    def concise_str(self):
        name = ""
        name += "ap_"
        for multi_class_label, number in self.train_class_number.items():
            name += multi_class_label + str(number)
        name += "_len{}".format(self.window_length)
        if self.one_hot_encode_mode:  # not all training needs one hot encoding
            name += "_onehot{}".format(self.one_hot_encode_mode)
        return name

    def augment_report_df(self):
        for multi_class_label, number in self.train_class_number.items():
            name = 'Number of samples for {} class'.format(multi_class_label)
            self._report_df[name] = number


class NwayKshotDataConfig(ArbitraryDataConfig):
    """
    (NEW):
    n_way: we will digest meta_train_class_portion and task_type to determine the n_ways, user DON'T specify this
    k_support, k_query: number of shots, user will specify this
    (Removed):
    batch_size: now it becomes task_number
    meta_train_class_number

    Public member:
        self.K<int> = self.k_support = k_support
        self.k_query<int> = k_query
        self.N<int> = number of classes
        self.meta_train/test_ways<tuple> = ('H','A'), etc
    """

    def __init__(self, window_length, k_support, k_query, sample_per_experiment, class_type,
                 meta_train_class_number, meta_test_ways=('H', 'N', 'NV', 'D'), balance_data=True):
        # Populate other members
        super(NwayKshotDataConfig, self).__init__(window_length=window_length,
                                                  sample_per_experiment=sample_per_experiment,
                                                  class_type=class_type,
                                                  balance_data=balance_data,
                                                  train_class_number=meta_train_class_number)
        # Remove unused members to avoid confusion
        del self.batch_size

        # (NEW) Few Shot data members
        self.meta_train_class_number = meta_train_class_number
        self.meta_test_ways = meta_test_ways  # We will reparse it to tuples in _identify_K_N()
        self.k_support = k_support
        self.k_query = k_query
        self.meta_train_ways, self.meta_test_ways, self.N, self.K = self._identify_K_N()
        self.augment_report_df_with_meta_attributes()

    def _identify_K_N(self):
        K = self.k_support
        if self.class_type == 'multi':
            meta_train_ways = []
            for class_label, number_of_sample in self.meta_train_class_number.items():
                if number_of_sample != 0:
                    meta_train_ways.append(class_label)
            N = len(meta_train_ways)
            meta_test_ways = self.meta_test_ways
        elif self.class_type == 'binary':
            N = 2
            meta_train_ways = ['H', 'A']  # healthy vs faulty
            meta_test_ways = ['H', 'A']
        return tuple(meta_train_ways), tuple(meta_test_ways), N, K

    def __str__(self):
        name = self.concise_str()
        name += "_spe{}".format(self.spe)
        name += "_{}".format(self.class_type)
        return name

    def concise_str(self):
        name = ""
        name += "class_" + str(''.join(self.meta_train_ways))
        name += "_{}way{}shot".format(self.N, self.K)
        for multi_class_label, number in self.train_class_number.items():
            name += multi_class_label + str(number)
        name += "_len{}".format(self.window_length)
        if self.one_hot_encode_mode:  # not all training needs one hot encoding
            name += "_onehot{}".format(self.one_hot_encode_mode)
        return name

    def get_public_member(self):
        return self.meta_train_class_number, self.meta_train_ways, self.meta_test_ways, self.N, self.K, self.k_support, self.k_query

    def init_report_df(self):
        report_df_init = {'Classes': self.class_type,
                          'Window Length': self.window_length,
                          'Sample Per Experiment': self.spe}
        report_df = pd.DataFrame(report_df_init, index=[0])
        return report_df

    def augment_report_df_with_meta_attributes(self):
        self._report_df['N Ways'] = self.N
        self._report_df['K Support'] = self.k_support
        self._report_df['K Query'] = self.k_query

    @property
    def report_df(self):
        return self._report_df.add_prefix('Data - ')


if __name__ == "__main__":  # Testing these modules
    SV, logger, device = initialize_logger_and_device("", debug=True)

    # Test cases
    test_window_length = 200
    test_spe = 200  # Use -1 if you have a crazy amount of memory :), otherwise 200 for testing
    test_batch_size = 64
    class_type = 'binary'
    test_train_class_number = {'H': 100,
                               'N': 100,
                               'NV': 100,
                               'D': 10}

    # Test Train Config
    train_config = LSTMConfig(LSTM_hidden_size=30,
                              n_epochs=10,
                              seed=42,
                              exp_lr_gamma=0.99,
                              early_stop_mode='not-best',
                              early_stop_tolerance=5,
                              criterion_type='BCE',
                              optimizer_type='adam',
                              lr=[0.00001])

    # Test cases
    test_few_shot_data_loader = 0
    if test_few_shot_data_loader:
        data_config = NwayKshotDataConfig(window_length=test_window_length,
                                          sample_per_experiment=test_spe,
                                          class_type=class_type,
                                          k_support=5,
                                          k_query=5,
                                          meta_test_ways=('H', 'N', 'NV'),
                                          meta_train_class_number=test_train_class_number)
        print(data_config.report_df)

    test_full_shot_data_loader = 0
    if test_full_shot_data_loader:
        data_config = DataConfig(window_length=test_window_length, sample_per_experiment=test_spe,
                                 batch_size=test_batch_size,
                                 class_type=class_type)
        print(data_config.report_df)
        data_loader = FullShotClassDataLoader(data_config=data_config, device=device, logger=logger)
        data = data_loader.load_data()

    test_ap_data_loader = 0
    if test_ap_data_loader:
        data_config = ArbitraryDataConfig(window_length=test_window_length, sample_per_experiment=test_spe,
                                          batch_size=test_batch_size,
                                          train_class_number=test_train_class_number,
                                          class_type=class_type)
        data_loader = ArbitraryPortionDataLoader(data_config=data_config, device=device, logger=logger)
        data = data_loader.load_data()
