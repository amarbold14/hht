import higher

from shared.NN_util import *
import time
from torchinfo import summary
import statistics
import os

class MyMetaTrainer:
    """
    Args:
        meta_model<MAML>:
            note: the original 'train_config' is now stored as meta_model.backbone_train_config
        data_loader: now the data_loader has to join because they are called during training
        data_config: e.g., NwayKshotDataConfig
    """

    def __init__(self, meta_model, data_config, data_loader, logger, device, writer):
        # Parse input
        self.data_loader = data_loader
        self.data_config = data_config
        self.logger = logger
        self.device = device
        self.writer = writer

        # Parse meta model - much train_config and meta_train_config are already embodied
        self.meta_model = meta_model
        self.train_config = meta_model.backbone_train_config
        self.num_inner_loop_per_epoch = self.meta_model.num_inner_loop_per_epoch
        self.num_tasks_per_inner_loop = self.meta_model.num_tasks_per_inner_loop
        self.early_stopper = EarlyStopper(tolerance=self.meta_model.backbone_train_config.early_stop_tolerance,
                                          mode=self.meta_model.backbone_train_config.early_stop_mode)

        # Parse data information for meta training
        self.K = self.data_loader.k_support + self.data_loader.k_query
        self.N = self.data_loader.N

        # Get training information
        self.n_epoch = self.meta_model.backbone_train_config.n_epochs
        self.accuracy = Accuracy().to(self.device)

        # Variables in train and data config
        # (Deprecated) batch_size is no longer used
        self.class_type = self.data_config.class_type
        self.model_info = self._evaluate_and_record_meta_model_info()

        # Initialize training log variables
        self._path_best_epoch_meta_val_model = str(self.writer) + "/model_best_epoch_meta_val_acc.pth"
        self._path_latest_model = str(self.writer) + "/model_latest.pth"
        self.best_epoch_meta_train_acc, self.best_epoch_meta_val_acc = 0, 0
        self.total_tasks_done = 0
        self._report_df = pd.DataFrame(index=[0])

        # Initialize evaluation variables
        self.eval_file_prefix = 'eval_'

    def _evaluate_and_record_meta_model_info(self):
        K = (self.K,)  # convert to tuple
        batch_input_dim = K + self.meta_model.backbone_model.input_size
        model_info = summary(self.meta_model.backbone_model, input_size=batch_input_dim, device=self.device,
                             col_names=["input_size", "output_size", "num_params", "mult_adds"], verbose=0)
        return model_info

    def _report_and_record_inner_loop_info(self, epoch, tasks_done, meta_train_loss, meta_train_acc, meta_val_loss,
                                           meta_val_acc):
        # Log information with debug
        self.logger.debug("Epoch {}, Inner Loop {} " \
                          "Meta-Train Loss {:.3f}, Acc {:.3f}; " \
                          "Meta-Val Loss {:.3f}, Acc {:.3f}.".format(epoch, tasks_done,
                                                                     meta_train_loss, meta_train_acc,
                                                                     meta_val_loss, meta_val_acc))
        # Write information
        self.writer.add_scalar("meta_train_loss/inner_loop", meta_train_loss, self.total_tasks_done)
        self.writer.add_scalar("meta_val_loss/inner_loop", meta_val_loss, self.total_tasks_done)
        self.writer.add_scalar("meta_train_acc/inner_loop", meta_train_acc, self.total_tasks_done)
        self.writer.add_scalar("meta_val_acc/inner_loop", meta_val_acc, self.total_tasks_done)

    def _report_and_record_epoch_info(self, epoch, epoch_start_time, epoch_meta_train_loss, epoch_meta_train_acc,
                                      epoch_meta_val_loss, epoch_meta_val_acc):
        # Get Epoch time
        epoch_period = time.time() - epoch_start_time

        # Calculate per epoch metrics - the query loss are the meta-train loss
        avg_epoch_meta_train_loss = statistics.mean(epoch_meta_train_loss)
        avg_epoch_meta_train_acc = statistics.mean(epoch_meta_train_acc)
        avg_epoch_meta_val_loss = statistics.mean(epoch_meta_val_loss)
        avg_epoch_meta_val_acc = statistics.mean(epoch_meta_val_acc)
        std_epoch_meta_val_acc = statistics.stdev(epoch_meta_val_acc)

        # Record per epoch metrics
        self.writer.add_scalar("meta_train_loss/epoch", avg_epoch_meta_train_loss, epoch)
        self.writer.add_scalar("meta_train_acc/epoch", avg_epoch_meta_train_acc, epoch)
        self.writer.add_scalar("meta_val_loss/epoch", avg_epoch_meta_val_loss, epoch)
        self.writer.add_scalar("meta_val_acc_mean/epoch", avg_epoch_meta_val_acc, epoch)
        self.writer.add_scalar("meta_val_acc_std/epoch", std_epoch_meta_val_acc, epoch)

        # Record and save best models
        if avg_epoch_meta_val_acc > self.best_epoch_meta_val_acc:
            self.best_epoch_val_acc = avg_epoch_meta_val_acc
            torch.save(self.meta_model.backbone_model.state_dict(), self._path_best_epoch_meta_val_model)
        if avg_epoch_meta_train_acc > self.best_epoch_meta_train_acc:
            self.best_epoch_train_acc = avg_epoch_meta_train_acc

        # Record best model in Writer
        self.writer.add_scalar("best_epoch_meta_val_acc/epoch", self.best_epoch_meta_val_acc, epoch)
        self.writer.add_scalar("total_time_epoch/epoch", epoch_period, epoch)

        # Log things
        self.logger.critical(
            "Epoch {}, Meta-Train Loss {:.3f}, Acc {:.3f}; " \
            "Meta-Val Loss {:.3f}, Acc mean {:.3f}, std {:.3f}; " \
            "Epoch Time {:.3f} sec".format(
                epoch, avg_epoch_meta_train_loss, avg_epoch_meta_train_acc, avg_epoch_meta_val_loss,
                avg_epoch_meta_val_acc, std_epoch_meta_val_acc,
                epoch_period))

        return avg_epoch_meta_train_loss, avg_epoch_meta_train_acc, avg_epoch_meta_val_loss, avg_epoch_meta_val_acc

    def load_which_model(self, which_model):
        model = self.meta_model.backbone_model  # load architecture
        torch.save(model.state_dict(), self._path_latest_model)

        # overwrite local 'model'
        if which_model == 'best-epoch':
            model.load_state_dict(torch.load(self._path_best_epoch_meta_val_model))
            return model, self._path_best_epoch_meta_val_model
        elif which_model == 'latest':
            return model, self._path_latest_model

    def _training_termination_sequence(self, epoch):
        if self.early_stopper.is_early_stop:
            self.logger.critical("Training early stopped at Epoch {}.".format(epoch))
        else:
            self.logger.critical("Training Completed after {} epochs.".format(epoch))

        self.writer.write_training_meta_data(epoch=epoch, best_val_acc=self.best_epoch_val_acc,
                                             early_stopped=self.early_stopper.is_early_stop)

        self._report_df['Last Epoch Number'] = epoch

    @property
    def report_df(self):
        df = pd.concat([self.early_stopper.report_df, self._report_df], axis=1, join='inner').add_prefix('Meta Trainer - ')
        return df



class MAMLTrainer(MyMetaTrainer):
    def __init__(self, meta_model, data_loader, data_config, logger, device, writer):
        super(MAMLTrainer, self).__init__(meta_model=meta_model, data_config=data_config, data_loader=data_loader,
                                          logger=logger, device=device, writer=writer)

    def train(self):
        # Initialization
        self.logger.critical("Main MAML training loop starts...")
        self.total_tasks_done = 0

        # Main train loop
        # with torch.backends.cudnn.flags(enabled=False):  # for pytorch RNN
        for epoch in range(self.n_epoch):
            epoch_start_time = time.time()
            epoch_meta_train_loss, epoch_meta_train_acc, epoch_meta_val_loss, epoch_meta_val_acc = [], [], [], []

            for inner_loop in range(self.num_inner_loop_per_epoch):
                # Train
                with torch.backends.cudnn.flags(enabled=False):  # for pytorch RNN
                    meta_train_loss, meta_train_acc = self.meta_model.train()
                    epoch_meta_train_loss.append(meta_train_loss)
                    epoch_meta_train_acc.append(meta_train_acc)

                # Validation
                _, _, _, meta_val_loss, meta_val_acc, _ = self.meta_model.validate()
                epoch_meta_val_loss.append(meta_val_loss)
                epoch_meta_val_acc.append(meta_val_acc)

                # Record inner loop information
                self.total_tasks_done += self.num_tasks_per_inner_loop['train']
                self._report_and_record_inner_loop_info(epoch=epoch, tasks_done=inner_loop,
                                                        meta_train_loss=meta_train_loss,
                                                        meta_train_acc=meta_train_acc, meta_val_loss=meta_val_loss,
                                                        meta_val_acc=meta_val_acc)

            # Record epoch information
            _, _, _, avg_epoch_meta_val_acc = self._report_and_record_epoch_info(epoch=epoch,
                                                                                 epoch_start_time=epoch_start_time,
                                                                                 epoch_meta_train_loss=epoch_meta_train_loss,
                                                                                 epoch_meta_train_acc=epoch_meta_train_acc,
                                                                                 epoch_meta_val_loss=epoch_meta_val_loss,
                                                                                 epoch_meta_val_acc=epoch_meta_val_acc)

            # Update early stopper
            self.early_stopper.step(early_stopping_variable=avg_epoch_meta_val_acc)
            self.logger.critical(
                "Early stopper mode:{}, tolerance:{}, counter:{}, value: {:.3f}.".format(self.early_stopper.mode,
                                                                                         self.early_stopper.tolerance,
                                                                                         self.early_stopper.tolerance_counter,
                                                                                         self.early_stopper.early_stopping_variable))
            if self.early_stopper.is_early_stop:
                break

            # Step meta scheduler
            self.meta_model.meta_scheduler.step()

        # End
        self._training_termination_sequence(epoch)
        return self.meta_model.backbone_model
