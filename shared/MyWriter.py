from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os

class MyWriter(SummaryWriter):
    def __init__(self, train_config, data_config, logger, additional_comment=""):
        self.logger_comment = logger.name
        self.train_comment = str(train_config)
        self.data_comment = str(data_config.concise_str())  # get the short-handed string of data
        self.dir = "./runs/"
        self.full_name = self.dir + self.logger_comment + self.data_comment + self.train_comment + additional_comment

        super(MyWriter, self).__init__(self.full_name)

    def __str__(self):
        """
        Get str representation of the writer, in case we need to use it further
        """
        return self.full_name

    def write_training_meta_data(self, epoch, best_val_acc, early_stopped):
        """
        Save a .txt file as a metadata to illustrate the training effect
        """
        filename = "train_"
        filename += "best_val_acc_" + str(round(best_val_acc, 3))
        filename += "_last_epoch_" + str(epoch)
        filename += "_early_stopped_" + str(early_stopped)
        self.write_metadata(filename)

    def write_inference_time_meta_data(self, prefix, mean_one_sample, std_one_sample, mean_batch_sample,
                                       std_batch_sample, batch_size):
        """
        Save a .txt file as a metadata to illustrate the inference
        """
        filename = prefix
        filename += "inf_time_1_" + str(round(mean_one_sample, 3))
        filename += "_" + str(round(std_one_sample, 3))
        filename += "_batch" + str(batch_size)
        filename += "_" + str(round(mean_batch_sample, 3))
        filename += "_" + str(round(std_batch_sample, 3))
        self.write_metadata(filename)

    def write_metadata(self, filename):
        prelim = self.full_name + "/"
        filename = prelim + filename + ".txt"
        # print(os.getcwd(), 'write meta data')
        # print(os.listdir('.'), 'write meta data dirs')
        # print(filename, 'filename')
        # print(prelim, 'prelim')
        # abs = os.path.join(r'C:/Users/MRL - Workstation/Documents/GitHub/MRL_Few_Shot/', filename)
        # print(abs, 'abs')
        # print(os.path.exists(prelim), 'is dir exist')
        # new = os.path.join(prelim, filename+'.txt')
        # os.makedirs(prelim, exist_ok=True)
        with open(filename, 'w') as f:
            f.write(filename)

    def write_evaluate_report_df(self, filename, df):
        prelim = self.full_name + "/"
        filename = prelim + filename
        df.to_excel(filename)


class MetaWriter(MyWriter):
    def __init__(self, meta_train_config, train_config, data_config, logger):
        self.meta_name = str(meta_train_config)
        super(MetaWriter, self).__init__(train_config=train_config, logger=logger, data_config=data_config,
                                         additional_comment=self.meta_name)
