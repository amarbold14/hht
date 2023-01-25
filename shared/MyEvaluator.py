from shared.NN_util import *
import pandas as pd
import hht


# Utility function to plot demo predictions
def plot_demo_predictions(ts, x1, x2, x3, mode, x_label, y_label):
    # Plot figures
    fig = plt.figure(figsize=(9, 16))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    ax1.plot(ts, x1, alpha=0.6)
    ax2.plot(ts, x2, alpha=0.6)
    ax3.plot(ts, x3, alpha=0.6)

    ax1.set_ylabel(y_label)
    ax1.set_title(mode + ' class: ground-truth')
    ax2.set_ylabel(y_label)
    ax2.set_title(mode + ' class: predicted value')
    ax3.set_ylabel('difference')
    ax3.set_title(mode + ' class: predicted and ground-truth difference')
    ax3.set_xlabel(x_label)

    fig_name = mode + '_demo'
    return fig, fig_name


class Evaluator:
    """
    Evaluates the trainer model, given Trainer
    Args:
        Trainer: ModelTrainer, MAMLTrainer, etc
    Member:
        report_df: An accessible dataframe to echo and export to .csv
    """

    def __init__(self, Trainer):
        self.trainer = Trainer
        self.writer = self.trainer.writer

        # Initialize report_df
        self.class_type = self.trainer.class_type  # 'binary' or 'multi'
        self._logger_report_df = pd.DataFrame({"Logger Name": self.trainer.logger.name}, index=[0])
        self._data_report_df = self.trainer.data_config.report_df
        self._data_loader_report_df = self.trainer.data_loader.report_df
        self._train_report_df = self.trainer.train_config.report_df
        self._trainer_report_df = self.trainer.report_df
        self._report_df = pd.DataFrame(index=[0])

    @property
    def report_df(self):
        df = pd.concat(
            [self._logger_report_df, self._data_report_df, self._data_loader_report_df, self._train_report_df,
             self._trainer_report_df, self._report_df.add_prefix("Eval - ")], axis=1, join='inner')
        return df

    def write_report_df(self):
        self.writer.write_evaluate_report_df(filename='result.xlsx', df=self.report_df.T)


class ModelEvaluator(Evaluator):
    """
    Evaluator for ModelTrainer class
    """

    def __init__(self, Trainer):
        super().__init__(Trainer=Trainer)

    def evaluate(self, test_data, rep=1000, which_model='best-batch', print_model_architecture=False):
        """
        which: 'best-batch', 'best-epoch', 'latest'
        """
        # Parse input arguments
        self._report_df['Which Model?'] = which_model
        X_test, Ys_test, Yo_test, Ys_bin_test = test_data

        # Remove cache
        torch.cuda.empty_cache()

        # Load model
        model, path = self.trainer.load_which_model(which_model=which_model)
        model.eval()

        # Forward pass on the test set
        with torch.no_grad():
            num_of_points = X_test.shape[0]
            print('X test shape: ', X_test.shape)
            print('X', X_test[:,:,10:13].shape)
            print(Ys_test.shape)
            for i in range(num_of_points):
                X_test_processed = torch.FloatTensor(num_of_points, 3, 224, 224).to('cuda')
                cur_point = X_test[i, :, 10:13].numpy(force=True)
                cur_processed = hht.get_image(cur_point)
                X_test_processed[i, :, :, :] = cur_processed

            outputs = model(X_test_processed)

        # Evaluate confusion matrix information
        test_acc = self.evaluate_and_write_confusion_matrix(outputs=outputs, Ys=Ys_test, Ys_bin=Ys_bin_test,
                                                            which_model=which_model)

        # Evaluate inference time performance
        self.evaluate_and_write_inference_time(model=model, batch_size=self.trainer.batch_size, rep=rep)

        # Report accuracy and which model
        self.report_accuracy(test_acc=test_acc)

        # Write result into Excel sheet
        self.write_report_df()

        # Print model architecture optionally
        if print_model_architecture:
            self.trainer.logger.critical(str(self.trainer.model_info))

    def evaluate_demo(self, demo_data, which_model='best-batch'):
        # Parse demo data
        X_demo, Ys_demo, Yo_demo, Ys_bin_demo = demo_data

        # Load model, just as self.evaluate
        model, path = self.trainer.load_which_model(which_model=which_model)
        model.eval()

        # Infer on the demo set
        self.infer_demo(model=model, X_demo=X_demo, Ys_demo=Ys_demo, Ys_bin_demo=Ys_bin_demo)

    def infer_demo(self, model, X_demo, Ys_demo, Ys_bin_demo):
        # Generate output for demo set
        with torch.no_grad():
            outputs = model(X_demo)

        prediction, ground_truth = nn_util_network_output_to_class_prediction(outputs=outputs,
                                                                              groundtruths=Ys_demo,
                                                                              class_type=self.class_type)

        # Convert to numpy
        prediction = prediction.cpu().detach().numpy().flatten()
        prediction_bin = np.array(prediction != 0).astype(int).flatten()  # unhealthy
        ground_truth = ground_truth.cpu().detach().numpy().flatten()
        ground_truth_bin = Ys_bin_demo.cpu().detach().numpy().flatten()
        difference_array_bin = np.subtract(ground_truth_bin, prediction_bin)
        difference_array = (ground_truth != prediction).astype(int).flatten()

        # Get x(time) axis
        timestep = np.arange(0, len(prediction_bin))

        # Plot the predictions
        x_label = 'time step'
        y_label = 'healthy or faulty'

        # Plot
        fig_bin, fig_name_bin = plot_demo_predictions(ts=timestep, x1=ground_truth_bin, x2=prediction_bin,
                                                      x3=difference_array_bin, mode='binary', x_label=x_label,
                                                      y_label=y_label)
        fig, fig_name = plot_demo_predictions(ts=timestep, x1=ground_truth, x2=prediction, x3=difference_array,
                                              mode='multi', x_label=x_label, y_label=y_label)

        # Save and write figures
        fig.savefig(str(self.trainer.writer) + '/' + self.trainer.eval_file_prefix + fig_name + '.png',
                    bbox_inches='tight')
        fig_bin.savefig(str(self.trainer.writer) + '/' + self.trainer.eval_file_prefix + fig_name_bin + '.png',
                        bbox_inches='tight')
        self.trainer.writer.add_figure(fig_name, figure=fig)
        self.trainer.writer.add_figure(fig_name_bin, figure=fig_bin)

    def evaluate_and_write_confusion_matrix(self, outputs, Ys, Ys_bin, which_model):
        """
        Args:
            Ys: multi class labels <groundtruth>
            Ys_bin: binary class labels <groundtruth>
            outputs: multi/binary class labels <prediction>
            which_model: pass from upstream, for writer and logger
        """
        # Evaluate confusion matrix information
        cnf, cnf_per_class, asym_cnf, asym_cnf_ratio, f, f_asym, test_acc = nn_evaluate_confusion_matrix(
            outputs=outputs,
            Ys_test=Ys,
            Ys_bin_test=Ys_bin,
            class_type=self.class_type,
            accuracy=self.trainer.accuracy)

        # Write confusion matrix information
        self.trainer.logger.critical("Asymmetric confusion matrix:\n {}".format(asym_cnf))
        self.trainer.logger.critical("Asymmetric confusion matrix ratio:\n {}".format(asym_cnf_ratio))
        self.trainer.logger.critical("Confusion matrix:\n {}".format(cnf))
        self.trainer.logger.critical("Evaluated using '{}' model.".format(which_model))
        f.savefig(str(self.trainer.writer) + "/" + self.trainer.eval_file_prefix + 'confusion_matrix.png',
                  bbox_inches='tight')
        f_asym.savefig(str(self.trainer.writer) + "/" + self.trainer.eval_file_prefix + 'asym_confusion_matrix.png',
                       bbox_inches='tight')

        # Write figure into summary writer
        self.trainer.writer.add_figure("Confusion Matrix", figure=f)
        self.trainer.writer.add_figure("Asymmetric Confusion Matrix", figure=f_asym)

        # Populate confusion matrix information to
        self.populate_cnf_to_report_df(cnf=cnf, asym_cnf=asym_cnf, cnf_per_class=cnf_per_class)

        return test_acc

    def populate_cnf_to_report_df(self, cnf, cnf_per_class, asym_cnf):
        """
        Parse confusion matrix into member report_df
        """
        self._report_df['Total Validation Count'] = np.sum(cnf)
        if self.class_type == 'binary':
            self._report_df['CNF True Positive/Faulty (TP)'] = cnf[1][1]
            self._report_df['CNF True Negative/Healthy Rate (TN)'] = cnf[0][0]
            self._report_df['CNF False Positive/Faulty (FP)'] = cnf[0][1]
            self._report_df['CNF False Negative/Healthy (FN)'] = cnf[1][0]
            self._report_df['TPR'] = cnf_per_class[0][1][1]
            self._report_df['TNR'] = cnf_per_class[0][0][0]
            self._report_df['FPR'] = cnf_per_class[0][0][1]
            self._report_df['FNR'] = cnf_per_class[0][1][0]

        elif self.class_type == 'multi':
            for groundtruth_class, all_predictions_in_groundtruth_class in enumerate(cnf):
                for predicted_class, cnf_value in enumerate(all_predictions_in_groundtruth_class):
                    name = 'CNF(m)_{},{}'.format(groundtruth_class, predicted_class)
                    self._report_df[name] = cnf_value

        for groundtruth_class, all_predictions_in_groundtruth_class in enumerate(asym_cnf):
            for predicted_class, cnf_value in enumerate(all_predictions_in_groundtruth_class):
                name = 'CNF(b)_{},{}'.format(groundtruth_class, predicted_class)
                self._report_df[name] = cnf_value

    def report_accuracy(self, test_acc):
        # Report best accuracy
        self.trainer.logger.critical(
            "Accuracy: train {:.3f}; best batch validation {:.3f}; test {:.3f}.".format(
                self.trainer.best_epoch_train_acc,
                self.trainer.best_epoch_val_acc,
                test_acc))

        self._report_df['Accuracy Train (best epoch)'] = self.trainer.best_epoch_train_acc
        self._report_df['Accuracy Val (best epoch)'] = self.trainer.best_epoch_val_acc
        self._report_df['Accuracy Test'] = test_acc.item()

    def evaluate_and_write_inference_time(self, model, batch_size, rep=1000):
        mean_one, std_one, mean_batch, std_batch = nn_evaluate_model_inference_time(model=model,
                                                                                    repetitions=rep,
                                                                                    device=self.trainer.device,
                                                                                    batch_size=batch_size,
                                                                                    logger=self.trainer.logger)

        ## Record inference time results
        # To writer
        self.trainer.writer.write_inference_time_meta_data(prefix=self.trainer.eval_file_prefix,
                                                           mean_one_sample=mean_one, std_one_sample=std_one,
                                                           mean_batch_sample=mean_batch,
                                                           std_batch_sample=std_batch,
                                                           batch_size=batch_size)
        # To report_df
        self._report_df['Inference Time Number of Repetitions'] = rep
        self._report_df['Inference Time Mean <1 Sample> [ms]'] = mean_one
        self._report_df['Inference Time Std <1 Sample> [ms]'] = std_one
        self._report_df['Inference Time Mean <{} Samples> [ms]'.format(batch_size)] = mean_batch
        self._report_df['Inference Time Std <{} Samples> [ms]'.format(batch_size)] = std_batch


class MAMLEvaluator(ModelEvaluator):
    """
    Args:
        Trainer<MetaTrainer>
        data_loader: Meta/MAML data_loader to load test tasks and demo tasks
    NEW Members:
        self.best_test_model, self.worst_test_model: we will save them for the demo set
        self.num_of_tasks_for_test_data: directly get from meta_model
        self.num_of_tasks_for_demo_data: set to be the same as above
        self._meta_train_report_df: report_df from MetaTrainConfig
    """

    def __init__(self, Trainer, data_loader):
        super().__init__(Trainer=Trainer)

        # (NEW)
        self.data_loader = data_loader
        self.num_of_tasks_for_test_data = self.trainer.meta_model.num_tasks_per_inner_loop['test']
        self.num_of_tasks_for_demo_data = self.num_of_tasks_for_test_data
        self.best_test_model, self.worst_test_model = None, None
        self._meta_train_report_df = self.trainer.meta_model.meta_train_config.report_df

    @property
    def report_df(self):
        df = pd.concat([self._logger_report_df, self._data_report_df, self._data_loader_report_df,
                        self._meta_train_report_df, self._train_report_df,
                        self._trainer_report_df, self._report_df.add_prefix("Eval - ")], axis=1, join='inner')
        return df

    def evaluate(self, rep=1000, which_model='best-epoch', print_worst_model=False, print_model_architecture=False):
        """
        Evaluates on the test set
        Args:
            which_model: 'best-epoch', 'latest'
            rep: to evaluate inference time for one sample we take the average of [rep] samples
            print_worst_model: whether to echo the worst model or not
            print_model_architecture: whether to echo the model architecture in the end
        """
        # Remove cache and release memory
        del self.trainer.data_loader.X_train, self.trainer.data_loader.Y_train
        del self.trainer.data_loader.X_val, self.trainer.data_loader.Y_val
        torch.cuda.empty_cache()

        # Load model
        meta_model, path = self.trainer.load_which_model(which_model=which_model)
        self.trainer.meta_model.backbone_model = meta_model  # Replace the Trainer model to be the one we loaded

        # This pass uses all data except for the support as query -> std(accuracy) is significantly smaller
        # Forward pass on the test set, echo and record results
        test_result = self.trainer.meta_model.test(use_all_remain_data_for_test=True)
        support_loss, support_acc, support_acc_std, query_loss, query_acc, query_acc_std, self.best_test_model, self.worst_test_model, num_tasks = test_result
        self.trainer.logger.critical(
            "Evaluated on {} test set tasks - avg support loss: {:.3f}, avg support acc: {:.3f}, std support acc: {:.3f}, "
            "avg query loss: {:.3f}, avg query acc: {:.3f}, std query acc {:.3f}.".format(num_tasks, support_loss,
                                                                                          support_acc, support_acc_std,
                                                                                          query_loss, query_acc,
                                                                                          query_acc_std))
        self._report_df['Number of Test Tasks'] = num_tasks
        self._report_df['Avg Test Support Loss'] = support_loss
        self._report_df['Avg Test Query Loss'] = query_loss
        self._report_df['Avg Test Support Acc'] = support_acc
        self._report_df['Std Test Support Acc'] = support_acc_std
        self._report_df['Avg Test Query Acc'] = query_acc
        self._report_df['Std Test Query Acc'] = query_acc_std

        # This pass uses the same amount of support and query for test as for training std(accuracy) is significantly greater
        # Forward pass on the test set, echo and record results
        test_echo_name = "Small Query"
        test_result = self.trainer.meta_model.test(use_all_remain_data_for_test=False)
        support_loss, support_acc, support_acc_std, query_loss, query_acc, query_acc_std, best_model_small_query, _, num_tasks = test_result
        self.trainer.logger.critical(
            "Evaluated on {} test set tasks for {} - avg support loss: {:.3f}, avg support acc: {:.3f}, std support acc: {:.3f}, "
            "avg query loss: {:.3f}, avg query acc: {:.3f}, std query acc {:.3f}.".format(num_tasks,
                                                                                          test_echo_name,
                                                                                          support_loss,
                                                                                          support_acc,
                                                                                          support_acc_std,
                                                                                          query_loss,
                                                                                          query_acc,
                                                                                          query_acc_std))
        self._report_df['Number of Test Tasks {}'.format(test_echo_name)] = num_tasks
        self._report_df['Avg Test Support Loss {}'.format(test_echo_name)] = support_loss
        self._report_df['Avg Test Query Loss {}'.format(test_echo_name)] = query_loss
        self._report_df['Avg Test Support Acc {}'.format(test_echo_name)] = support_acc
        self._report_df['Std Test Support Acc {}'.format(test_echo_name)] = support_acc_std
        self._report_df['Avg Test Query Acc {}'.format(test_echo_name)] = query_acc
        self._report_df['Std Test Query Acc {}'.format(test_echo_name)] = query_acc_std

        # Evaluate best model by default
        self.trainer.logger.critical("Evaluating BEST test model...")
        best_test_acc = self._evaluate_best_or_worst_model(best_worst_model=self.best_test_model,
                                                           which_model=which_model,
                                                           rep=rep)

        # Evaluate worst model (DEPRECATED)
        if print_worst_model:
            self.trainer.logger.critical("Evaluating WORST test model...")
            worst_test_acc = self._evaluate_best_or_worst_model(best_worst_model=self.worst_test_model,
                                                                which_model=which_model, rep=rep)
            # Report accuracies
            self.trainer.logger.critical(
                "Accuracy: best train {:.3f}; best batch validation {:.3f}; best test {:.3f}, worst test {:.3f}.".format(
                    self.trainer.best_epoch_train_acc, self.trainer.best_epoch_val_acc, best_test_acc, worst_test_acc))
        else:  # Default path
            # Final echo and record best accuracies
            self.trainer.logger.critical(
                "Accuracy: best train {:.3f}; best batch validation {:.3f}; best test support {:.3f}, query {:.3f}.".format(
                    self.trainer.best_epoch_train_acc, self.trainer.best_epoch_val_acc,
                    self.best_test_model['acc_support'],
                    self.best_test_model['acc_query']))

            self._report_df['Best Train - Query Accuracy'] = self.trainer.best_epoch_train_acc
            self._report_df['Best Val - Query Accuracy'] = self.trainer.best_epoch_val_acc
            self._report_df['Best Test - Support Accuracy'] = self.best_test_model['acc_support']
            self._report_df['Best Test - Query Accuracy'] = self.best_test_model['acc_query']
            self._report_df['Best Test {} - Support Accuracy'.format(test_echo_name)] = best_model_small_query['acc_support']
            self._report_df['Best Test {} - Query Accuracy'.format(test_echo_name)] = best_model_small_query['acc_query']

        # Write result into Excel sheet
        self.write_report_df()

        # Print model architecture optionally
        if print_model_architecture:
            self.trainer.logger.critical(str(self.trainer.model_info))

    def _evaluate_best_or_worst_model(self, best_worst_model, which_model, rep):
        """
        Evaluate the confusion matrix and inference time
        Args:
            best_worst_model: best_model or worst_model
            which_model: pass from upstream
        """
        # Forward pass the model
        model = best_worst_model['model']
        X = best_worst_model['x_query']
        Ys = best_worst_model['y_query']
        Ys_bin = best_worst_model['yb_query']

        model.eval()
        outputs = model(X)

        # Evaluate confusion matrix information
        test_acc = self.evaluate_and_write_confusion_matrix(outputs=outputs, Ys=Ys, Ys_bin=Ys_bin,
                                                            which_model=which_model)

        # Evaluate inference time performance
        self.evaluate_and_write_inference_time(model=model, batch_size=64, rep=rep)

        return test_acc

    def evaluate_demo(self, which_model='best_test'):
        """
        Args:
            which_model: 'best_test' or 'worst_test'
        """
        X_demo = self.trainer.data_loader.X_demo
        Ys_demo = self.trainer.data_loader.Ys_demo
        Yo_demo = self.trainer.data_loader.Yo_demo
        Ys_bin_demo = self.trainer.data_loader.Ys_bin_demo

        # Load model
        if which_model == 'best_test':
            model = self.best_test_model['model']
        elif which_model == 'worst_test':
            model = self.worst_test_model['model']

        model.eval()

        # Infer on demo set
        self.infer_demo(model=model, X_demo=X_demo, Ys_demo=Ys_demo, Ys_bin_demo=Ys_bin_demo)
