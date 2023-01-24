from model.train_config import *


class MetaTrainConfig:
    """
    Args:
        num_inner_loop_per_epoch: number of training loop per epoch
        num_tasks_per_inner_loop: number of tasks per inner loop
        meta_lr: learning rate of the outer loop
        optimizer_type: the optimizer used for meta training
        meta_exp_lr_gamma: learning rate gamma for the meta loop
    """

    def __init__(self, meta_lr, num_inner_loop_per_epoch=10, num_tasks_per_inner_loop=16, optimizer_type='adam',
                 meta_exp_lr_gamma=1.0):
        self.optimizer = None
        self.num_inner_loop_per_epoch = num_inner_loop_per_epoch
        self.num_tasks_per_inner_loop = num_tasks_per_inner_loop
        self.meta_lr = meta_lr
        self.meta_gamma = meta_exp_lr_gamma
        self.optimizer_type = optimizer_type

        self.name = self._concise_str()

        # df that stores config information
        self._report_df = self.init_report_df()

    def _concise_str(self):
        name = ""
        name += "_ilpe{}".format(self.num_inner_loop_per_epoch)
        name += "_tpil{}".format(self.num_tasks_per_inner_loop)
        name += "_mlr"
        for i in self.meta_lr:
            name += str(i) + "_"
        return name

    def __str__(self):
        return self.name

    def get_meta_optimizer(self, model):
        if self.optimizer_type == 'adam':
            assert len(self.meta_lr) == 1, f"Expect different number of arguments for the optimizer."
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.meta_lr[0])
            self._report_df['Meta Optimizer'] = self.optimizer_type
            return self.optimizer

    def get_scheduler(self):
        scheduler = ExponentialLR(self.optimizer, gamma=self.meta_gamma)
        self._report_df['Meta Gamma'] = str(self.meta_gamma)
        return scheduler

    def init_report_df(self):
        report_df = pd.DataFrame(index=[0])
        report_df['Inner Loop per Epoch'] = self.num_inner_loop_per_epoch
        report_df['Task per Inner Loop'] = self.num_tasks_per_inner_loop
        report_df['Meta Learning Rate'] = str(self.meta_lr)
        return report_df

    @property
    def report_df(self):
        return self._report_df.add_prefix('Meta Train Config - ')


class MAMLTrainConfig(MetaTrainConfig):
    """
    Args:
        (NEW) num_steps_before_query<dict>: in MAML you can update with the support loss many times before sending it to query
    """

    def __init__(self, meta_lr, num_steps_before_query={'train': 1, 'val': 1, 'test': 1},
                 num_inner_loop_per_epoch=10, num_tasks_per_inner_loop={'train': 16, 'val': 50, 'test': 1000},
                 optimizer_type='adam', meta_exp_lr_gamma=1.0):
        self.num_steps_before_query = num_steps_before_query
        super(MAMLTrainConfig, self).__init__(meta_lr=meta_lr, num_inner_loop_per_epoch=num_inner_loop_per_epoch,
                                              num_tasks_per_inner_loop=num_tasks_per_inner_loop,
                                              optimizer_type=optimizer_type, meta_exp_lr_gamma=meta_exp_lr_gamma)

    def _concise_str(self):
        name = "_maml"
        name += "_steps{}_{}_{}".format(self.num_steps_before_query['train'], self.num_steps_before_query['val'],
                                        self.num_steps_before_query['test'])
        name += "_ilpe{}".format(self.num_inner_loop_per_epoch)
        name += "_tpil{}_{}_{}".format(self.num_tasks_per_inner_loop['train'], self.num_tasks_per_inner_loop['val'],
                                       self.num_tasks_per_inner_loop['test'])
        name += "_mlr"
        for i in self.meta_lr:
            name += str(i) + "_"
        name = name[:-1]  # remove the last "_"
        return name

    def init_report_df(self):
        report_df = pd.DataFrame(index=[0])
        report_df['Model'] = 'MAML'
        report_df['Inner Loop per Epoch'] = self.num_inner_loop_per_epoch
        for mode in ['train', 'val', 'test']:
            report_df['Update Steps before Query Meta-{}'.format(mode.capitalize())] = self.num_steps_before_query[mode]
        for mode in ['train', 'val', 'test']:   # Two seperator for loops such that the rows don't mingle with each other
            report_df['Task per Inner Loop Meta-{}'.format(mode.capitalize())] = self.num_tasks_per_inner_loop[mode]
        report_df['Meta Learning Rate'] = str(self.meta_lr)
        return report_df
