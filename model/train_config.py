import torch
from torch.optim.lr_scheduler import ExponentialLR
import pandas as pd


class TrainConfig:
    """
    default training config class
    input:
        model: the NN module
        n_epoch
        optimizer_type<str>: adam
        criterion_type<str>: BCE/CE
        exp_lr_gamma<float>: gamma to use for the exponential lr scheduler
        lr<list>, depending on the optimizer, it will contain different lr values
    """

    def __init__(self,
                 n_epochs,
                 optimizer_type,
                 criterion_type,
                 lr,
                 seed,
                 exp_lr_gamma=1.0,
                 early_stop_mode='not-best',
                 early_stop_tolerance=3,
                 report_per_x_batch=50):

        # Optimizer related fields
        self.optimizer = None  # Will be filled in later
        self.optimizer_type = optimizer_type
        self.criterion_type = criterion_type
        self.lr = lr
        self.gamma = exp_lr_gamma
        self.seed = seed

        # Early stopping fields
        self.early_stop_mode = early_stop_mode
        self.early_stop_tolerance = early_stop_tolerance

        # Train process related fields
        self.n_epochs = n_epochs
        self.report_per_x_batch = report_per_x_batch  # Trivial

        # Name for the config
        self.name = self.default_str()

        # df that stores config information
        self._report_df = self.init_report_df()

    def default_str(self):
        name = ""
        name += "_seed{}".format(self.seed)
        name += "_ep{}".format(self.n_epochs)
        name += "_opt{}".format(self.optimizer_type)
        name += "_crit{}".format(self.criterion_type)
        name += "_lr"
        for i in self.lr:
            name += str(i) + "_"
        name += "gam{}".format(self.gamma)
        return name

    def __str__(self):
        return self.name

    def init_report_df(self):
        report_df = pd.DataFrame(index=[0])
        report_df['Epoch'] = self.n_epochs
        report_df['Seed'] = self.seed
        report_df['Optimizer'] = self.optimizer_type
        report_df['Criterion'] = self.criterion_type
        report_df['Learning Rate'] = self.lr
        report_df['Gamma'] = self.gamma
        return report_df

    def get_optimizer(self, model):
        if self.optimizer_type == 'adam':
            assert len(self.lr) == 1, f"Expect different number of arguments for the optimizer."
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr[0])
            return self.optimizer

    def get_scheduler(self):
        scheduler = ExponentialLR(self.optimizer, gamma=self.gamma)
        return scheduler

    def get_criterion(self):
        if self.criterion_type == 'CE':
            return torch.nn.CrossEntropyLoss()
        elif self.criterion_type == 'BCE':
            return torch.nn.BCELoss(reduction='mean')

    @property
    def report_df(self):
        return self._report_df.add_prefix('Train Config - ')


class LSTMConfig(TrainConfig):
    """
    Config for LSTM model
    batch_size_val is no longer used
    (NEW)LSTM_hidden_size: output vector dimension from LSTM
    """

    def __init__(self, n_epochs, optimizer_type, criterion_type, lr,
                 LSTM_hidden_size,
                 seed,
                 exp_lr_gamma=1,
                 early_stop_mode='not-best',
                 early_stop_tolerance=3,
                 report_per_x_batch=50):
        # Initialize with parent class
        super(LSTMConfig, self).__init__(n_epochs=n_epochs,
                                         optimizer_type=optimizer_type,
                                         criterion_type=criterion_type,
                                         lr=lr,
                                         seed=seed,
                                         exp_lr_gamma=exp_lr_gamma,
                                         early_stop_mode=early_stop_mode,
                                         early_stop_tolerance=early_stop_tolerance,
                                         report_per_x_batch=report_per_x_batch
                                         )

        # Child class initialize
        self.LSTM_hidden_size = LSTM_hidden_size

        # Assembly class name further
        self.name = self.new_str()

        # Augment report_df
        self.augment_report_df()

    def new_str(self):
        name = "_hsize_" + str(self.LSTM_hidden_size) + self.name
        return name

    def augment_report_df(self):
        self._report_df['Model'] = 'One Layer LSTM'
        self._report_df['LSTM Hidden Size'] = self.LSTM_hidden_size


class ResNetConfig(TrainConfig):
    def __init__(self, n_epochs, optimizer_type, criterion_type, lr, seed,
                 exp_lr_gamma=1,
                 early_stop_mode='not-best',
                 early_stop_tolerance=3,
                 report_per_x_batch=50):
        super(ResNetConfig, self).__init__(n_epochs=n_epochs,
                                           optimizer_type=optimizer_type,
                                           criterion_type=criterion_type,
                                           lr=lr,
                                           seed=seed,
                                           exp_lr_gamma=exp_lr_gamma,
                                           early_stop_mode=early_stop_mode,
                                           early_stop_tolerance=early_stop_tolerance,
                                           report_per_x_batch=report_per_x_batch
                                           )


class TwoLayerLSTMConfig(LSTMConfig):
    def __init__(self, n_epochs, optimizer_type, criterion_type, lr,
                 LSTM_hidden_size,
                 seed,
                 exp_lr_gamma=1,
                 early_stop_mode='not-best',
                 early_stop_tolerance=3,
                 report_per_x_batch=50):
        # Initialize with parent class
        super(TwoLayerLSTMConfig, self).__init__(n_epochs=n_epochs,
                                                 optimizer_type=optimizer_type,
                                                 criterion_type=criterion_type,
                                                 LSTM_hidden_size=LSTM_hidden_size[0],
                                                 lr=lr,
                                                 seed=seed,
                                                 exp_lr_gamma=exp_lr_gamma,
                                                 early_stop_mode=early_stop_mode,
                                                 early_stop_tolerance=early_stop_tolerance,
                                                 report_per_x_batch=report_per_x_batch
                                                 )
        self.LSTM_hidden_size = LSTM_hidden_size  # <list>
        self.name = self.new_1_str()

        # Augment report_df
        self.augment_report_df()

    def new_1_str(self):  # I'm too lazy to change the name
        name = "_hsize"
        for h in self.LSTM_hidden_size:
            name += str(h) + '_'
        name += self.name
        return name


    def augment_report_df(self):
        self._report_df['Model'] = 'Two Layer LSTM'
        self._report_df['LSTM Hidden Size'] = str(self.LSTM_hidden_size)

