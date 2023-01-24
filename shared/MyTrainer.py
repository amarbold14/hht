import pandas as pd
import torch.nn.functional

from shared.NN_util import *
import time
from torchinfo import summary
import sys
import hht

class MyTrainer:
    def __init__(self, model, train_config, data_loader, logger, device, writer):
        # These are helpers related to training
        self.train_config = train_config
        self.data_loader = data_loader
        self.data_config = data_loader.data_config
        self.logger = logger
        self.device = device
        self.writer = writer
        self.early_stopper = EarlyStopper(tolerance=self.train_config.early_stop_tolerance,
                                          mode=self.train_config.early_stop_mode)
        self.report_per_x_batch = self.train_config.report_per_x_batch

        # Get optimizer and criterion
        self.model = model
        self.criterion = self.train_config.get_criterion()
        self.optimizer = self.train_config.get_optimizer(self.model)
        self.scheduler = self.train_config.get_scheduler()
        self.n_epoch = self.train_config.n_epochs
        self.accuracy = Accuracy(task='binary').to(self.device)

        # Variables in train and data config
        self.class_type = self.data_config.class_type
        self.batch_size = self.data_config.batch_size
        self.model_info = self._evaluate_and_report_model_info()

        # Initialize training log variables that will dynamically update
        self._best_epoch_val_model_path = str(self.writer) + "/model_best_epoch_val_acc.pth"
        self._best_batch_val_model_path = str(self.writer) + "/model_best_batch_val_acc.pth"
        self._latest_model_path = str(self.writer) + "/model_latest.pth"
        self.best_epoch_val_acc, self.best_batch_val_acc, self.best_epoch_train_acc, self.total_batch_number = 0, 0, 0, 0
        self._report_df = pd.DataFrame(index=[0])

        # Initialize evaluation variables
        self.eval_file_prefix = 'eval_'

    def _evaluate_and_report_model_info(self):
        batch_size = (self.batch_size,)  # convert to tuple
        batch_input_dim = batch_size + self.model.input_size
        model_info = summary(self.model, input_size=batch_input_dim, device=self.device,
                             col_names=["input_size", "output_size", "num_params", "mult_adds"], verbose=0)
        return model_info

    def _report_and_record_batch_info(self, batch_number, epoch, loss, acc, loss_val, acc_val):
        # Report progress every x number batches
        if batch_number % self.report_per_x_batch == 0:
            self.logger.critical(
                "Epoch {}, Batch {}, Train Loss {:.3f}, Train Acc {:.3f}, Val Loss {:.3f}, Val Acc {:.3f}".format(
                    epoch, batch_number, loss, acc, loss_val, acc_val))

        # Log information
        self.writer.add_scalar("train_loss/batch", loss, self.total_batch_number)
        self.writer.add_scalar("val_loss/batch", loss_val, self.total_batch_number)
        self.writer.add_scalar("train_acc/batch", acc, self.total_batch_number)
        self.writer.add_scalar("val_acc/batch", acc_val, self.total_batch_number)

        # Record and save best models
        if acc_val > self.best_batch_val_acc:
            self.best_batch_val_acc = acc_val
            torch.save(self.model.state_dict(), self._best_batch_val_model_path)

    def _report_and_record_epoch_info(self, batch_number, epoch, epoch_start_time, epoch_train_loss,
                                      epoch_val_loss, epoch_train_acc, epoch_val_acc):
        # Get Epoch time
        epoch_period = time.time() - epoch_start_time

        # Calculate per epoch metrics
        avg_epoch_train_loss = epoch_train_loss / batch_number
        avg_epoch_val_loss = epoch_val_loss / batch_number
        avg_epoch_train_acc = epoch_train_acc / batch_number
        avg_epoch_val_acc = epoch_val_acc / batch_number

        # Record per epoch metrics
        self.writer.add_scalar("train_loss/epoch", avg_epoch_train_loss, epoch)
        self.writer.add_scalar("val_loss/epoch", avg_epoch_val_loss, epoch)
        self.writer.add_scalar("train_acc/epoch", avg_epoch_train_acc, epoch)
        self.writer.add_scalar("val_acc/epoch", avg_epoch_val_acc, epoch)

        # Record and save best models
        if avg_epoch_val_acc > self.best_epoch_val_acc:
            self.best_epoch_val_acc = avg_epoch_val_acc
            torch.save(self.model.state_dict(), self._best_epoch_val_model_path)
        if avg_epoch_train_acc > self.best_epoch_train_acc:
            self.best_epoch_train_acc = avg_epoch_train_acc

        # Record best model in Writer
        self.writer.add_scalar("best_epoch_val_acc/epoch", self.best_epoch_val_acc, epoch)
        self.writer.add_scalar("total_time_epoch/epoch", epoch_period, epoch)

        # Log epoch information
        self.logger.critical(
            "Epoch {}, Train Loss {:.3f}, Train Acc {:.3f}, Val Loss {:.3f}, Val Acc {:.3f}, Epoch Time {:.3f} sec".format(
                epoch, avg_epoch_train_loss, avg_epoch_train_acc, avg_epoch_val_loss, avg_epoch_val_acc, epoch_period))

        return avg_epoch_train_loss, avg_epoch_val_loss, avg_epoch_train_acc, avg_epoch_val_acc

    def load_which_model(self, which_model):
        model = self.model  # load architecture
        torch.save(self.model.state_dict(), self._latest_model_path)

        # overwrite local 'model'
        if which_model == 'best-epoch':
            model.load_state_dict(torch.load(self._best_epoch_val_model_path))
            return model, self._best_epoch_val_model_path
        elif which_model == 'best-batch':
            model.load_state_dict(torch.load(self._best_batch_val_model_path))
            return model, self._best_batch_val_model_path
        elif which_model == 'latest':
            return model, self._latest_model_path

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
        df = pd.concat([self.early_stopper.report_df, self._report_df], axis=1, join='inner').add_prefix('Trainer - ')
        return df


class ModelTrainer(MyTrainer):
    """
    Trainer for full shot classification
    """

    def __init__(self, model, train_config, data_loader, logger, device, writer):
        super(ModelTrainer, self).__init__(model=model, train_config=train_config,
                                           data_loader=data_loader, logger=logger,
                                           device=device, writer=writer)

    def train(self, train_data, val_data):
        X_train_tensor, Y_train_tensor = train_data
        X_val_tensor, Y_val_tensor = val_data

        # Add Graph to writer
        # self.writer.add_graph(model=self.model, input_to_model=X_val_tensor[0, :, :])

        # Start training
        for epoch in range(self.n_epoch):
            self.logger.critical("Epoch {}, lr {}".format(epoch, self.scheduler.get_last_lr()))
            epoch_start_time = time.time()
            epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc = 0, 0, 0, 0

            # Shuffle the samples for training
            permutation = torch.randperm(X_train_tensor.shape[0])

            # Start batch iteration
            for starting_idx in range(0, X_train_tensor.shape[0], self.batch_size):
                # Declare training
                self.model.train()
                self.optimizer.zero_grad()

                # Obtain batched samples
                batch_number = starting_idx // self.batch_size + 1
                sample_idx = permutation[starting_idx: starting_idx + self.batch_size]
                batch_x, batch_y = X_train_tensor[sample_idx], Y_train_tensor[sample_idx]

                # Process the batch
                for i in range(batch_x.shape[0]):
                    batch_processed = torch.FloatTensor(batch_x.shape[0], 3, 224, 224).to('cuda')
                    cur_point = batch_x[i, :, 10:13].numpy(force=True)
                    cur_processed = hht.get_image(cur_point)
                    batch_processed[i, :, :, :] = cur_processed

                self.total_batch_number += 1

                # Forward pass
                outputs = self.model.forward(batch_processed)
                outputs = torch.nn.functional.sigmoid(outputs)

                loss = self.criterion(outputs, batch_y)
                acc, _, _ = nn_util_network_output_to_class_prediction_acc(outputs=outputs, groundtruths=batch_y,
                                                                           class_type=self.class_type,
                                                                           accuracy=self.accuracy)
                loss.backward()
                self.optimizer.step()

                # Accumulate epoch loss for training
                epoch_train_loss += loss.item()
                epoch_train_acc += acc.item()

                # Forward pass with validation set
                self.model.eval()
                with torch.no_grad():  # Use the whole validation set
                    num_of_points = X_val_tensor.shape[0]
                    # num_of_points = 2
                    for i in range(num_of_points):
                        X_val_processed = torch.FloatTensor(num_of_points, 3, 224, 224).to('cuda')
                        cur_point = X_val_tensor[i, :, 10:13].numpy(force=True)
                        cur_processed = hht.get_image(cur_point)
                        X_val_processed[i, :, :, :] = cur_processed

                    outputs_val = self.model.forward(X_val_processed)
                    outputs_val = torch.nn.functional.sigmoid(outputs_val)

                    # Compute loss and accuracy
                    loss_val = self.criterion(outputs_val, Y_val_tensor)
                    acc_val, _, _ = nn_util_network_output_to_class_prediction_acc(outputs=outputs_val,
                                                                                   groundtruths=Y_val_tensor,
                                                                                   accuracy=self.accuracy,
                                                                                   class_type=self.class_type)
                    # Accumulate loss and accuracy for Writer
                    epoch_val_loss += loss_val.item()
                    epoch_val_acc += acc_val.item()

                # Report batch information
                self._report_and_record_batch_info(batch_number=batch_number, epoch=epoch,
                                                   loss=loss, acc=acc, loss_val=0, acc_val=0)
            # Record epoch information
            _, _, _, avg_epoch_val_acc = self._report_and_record_epoch_info(batch_number=batch_number, epoch=epoch,
                                                                            epoch_start_time=epoch_start_time,
                                                                            epoch_train_loss=epoch_train_loss,
                                                                            epoch_val_loss=epoch_val_loss,
                                                                            epoch_train_acc=epoch_train_acc,
                                                                            epoch_val_acc=epoch_val_acc)

            # Update early stopper
            self.early_stopper.step(early_stopping_variable=avg_epoch_val_acc)
            self.logger.critical(
                "Early stopper mode:{}, tolerance:{}, counter:{}, value: {:.3f}.".format(self.early_stopper.mode,
                                                                                         self.early_stopper.tolerance,
                                                                                         self.early_stopper.tolerance_counter,
                                                                                         self.early_stopper.early_stopping_variable))
            if self.early_stopper.is_early_stop:
                break

            # Update scheduler
            self.scheduler.step()

        self._training_termination_sequence(epoch)
        return self.model

