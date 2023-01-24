from sklearn.preprocessing import OneHotEncoder
from shared.utility import SharedVariable, report_converted_data_info
import numpy as np
import torch
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
import pandas as pd

def one_hot_encoding(Y, logger, mode='seen'):
    """
    :param Y: Process multi-classification labels to one hot vectors
    :param mode: 'seen', 'all':
        'seen' means encoding only seen labels in dataset
        'all' means using encoding all possible labels from shared variable
    :return: Y_oh, enc
    """
    logger.critical("One hot embedding the data with {} mode...".format(mode))
    enc = OneHotEncoder(handle_unknown='error')

    if mode == 'seen':
        Y_oh = enc.fit_transform(Y).toarray()
    elif mode == 'all':
        SV = SharedVariable()
        multi_class_map = SV.default_multi_class_map
        encoded_classes = np.fromiter(multi_class_map.values(), dtype=int).reshape(-1, 1)
        enc.fit(encoded_classes)
        Y_oh = enc.transform(Y).toarray()
    return Y_oh, enc


def manual_scheduler(epoch, lr):
    """
    how to use: lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    """
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.005)


def nn_util_extract_and_report_training_data(logger, device, X, Y):
    logger.critical("Running on device {}.".format(device))
    logger.critical("X train dimension - {}".format(X.shape))  # N, L, D_in
    logger.critical("Y train dimension - {}".format(Y.shape))  # N, D_out
    input_dimension = X.shape[-1]  # dim = 23
    sequence_len = X.shape[1]  # by default - dim = 50 or 100
    output_dimension = Y.shape[-1]  # one hot encoding, dim = 6
    return input_dimension, sequence_len, output_dimension


def nn_util_network_output_to_class_prediction_acc(outputs, groundtruths, class_type, accuracy=Accuracy(task='binary')):
    """
    :param accuracy: an Accuracy()
    :param outputs: dim: [batch_num, num_classes]
    :param groundtruths: dim: same as above
    :param class_type: 'multi', 'binary' or 'observation'
    :return: accuracy of prediction<int>
    """
    batch_prediction, batch_ground_truth = nn_util_network_output_to_class_prediction(outputs, groundtruths, class_type)

    # get accuracy
    if class_type == 'multi':
        acc = accuracy(batch_ground_truth, batch_prediction)
    elif class_type == 'binary':
        acc = torch.eq(batch_prediction, groundtruths).sum() / len(batch_prediction)
    return acc, batch_prediction, batch_ground_truth


def nn_util_network_output_to_class_prediction(outputs, groundtruths, class_type):
    """
    Turn outputs into class predictions and then into tensors
    """
    if class_type == 'multi':
        batch_prediction = torch.argmax(outputs, dim=1)
        batch_ground_truth = torch.argmax(groundtruths, dim=1)
    elif class_type == 'binary':
        batch_prediction = (outputs > 0.5).float()
        batch_ground_truth = groundtruths
    return batch_prediction, batch_ground_truth


def nn_util_asymmetric_confusion_matrix(ground_truth, prediction, mode):
    """
    ground_truth: [0,1,2,3, .., m] classes
    prediction: [0,1] classes
    mode: 'binary' or 'multi'

    return: mx2 numpy array, row is ground_truth, col is binary prediction, this can help us analyze how difficult...
            ...each mode is to make a healthy/faulty prediction
    """

    def nn_util_cnf_to_ratio(cnf):
        return normalize(cnf, axis=1, norm='l1')

    unique_classes, _ = np.unique(ground_truth, return_counts=True)
    num_of_classes = len(unique_classes)

    asym_cnf = np.zeros(shape=(num_of_classes, 2), dtype=int)  # mx2

    # If mode is binary, we just fill in the matrix
    if mode == 'binary':
        for idx, pred in enumerate(prediction):
            row = int(ground_truth[idx])
            col = int(pred)
            asym_cnf[row][col] += 1
    # If mode is multi, we collapse prediction of 1,2,3 to 0
    elif mode == 'multi':
        for idx, pred in enumerate(prediction):
            row = int(ground_truth[idx])
            col = int(pred)
            if col != 0:
                col = 1  # count as a faulty alarm
            asym_cnf[row][col] += 1

    # Compute asym_conf as ratio
    asym_cnf_ratio = nn_util_cnf_to_ratio(asym_cnf)

    return asym_cnf, asym_cnf_ratio


def nn_util_read_confusion_matrix(cnf):
    """
    Args: cnf <np.array>
    Returns:
        cnf_per_class <np.array> Parse the confusion matrix to various classes
    """
    # Get fpr, fnr, tpr, tnr
    number_of_classes = len(cnf)
    cnf_per_class = {}

    for ith_class in range(number_of_classes):
        tp = cnf[ith_class][ith_class]
        fp = sum(cnf[ith_class]) - tp
        fn = sum([cnf[ith_class][j] for j in range(number_of_classes)]) - tp
        tn = np.sum(cnf) - tp - fn - fp

        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        fnr = 1 - tpr
        fpr = 1 - tnr

        cnf_ith_class = np.array([[tpr, fpr], [fnr, tnr]])
        cnf_per_class[ith_class] = cnf_ith_class

    return cnf_per_class


def nn_evaluate_model_inference_time(model, batch_size, device, logger, repetitions=1000):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # A small util function in a function
    def evaluate_base_on_batch_size(bs):
        timings = np.zeros((repetitions, 1))
        dummy_input = torch.randn(bs, model.L, model.H_in, dtype=torch.float).to(device)
        torch.cuda.synchronize()

        # GPU warm up
        for _ in range(10):
            _ = model(dummy_input)

        # Measure inference performance for sequential data
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                # Wait for GPU sync
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

        timings = timings / bs
        mean_inference_time = np.mean(timings)
        std_inference_time = np.std(timings)

        return mean_inference_time, std_inference_time

    # Start Evaluation
    mean_one_sample, std_one_sample = evaluate_base_on_batch_size(1)
    mean_batch_sample, std_batch_sample = evaluate_base_on_batch_size(batch_size)

    # Record Data
    logger.critical('Inference timing performance over {} dummy samples: \n' \
                    'One sample: Mean inference time {:.3f} ms, std: {:.3f} ms.\n' \
                    'Batched <{}> samples: Mean inference time {:.3f} ms, std: {:.3f} ms.'.format(
        repetitions, mean_one_sample, std_one_sample, batch_size, mean_batch_sample, std_batch_sample))

    return mean_one_sample, std_one_sample, mean_batch_sample, std_batch_sample


def nn_evaluate_confusion_matrix(outputs, Ys_test, Ys_bin_test, class_type, accuracy=Accuracy(task='binary')):
    """
    Returns:
        cnf <np.array>: confusion matrix
        cnf_per_class <np.array> Parse the confusion matrix to various classes
        asym_cnf <np.array>
        asym_cnf_ratio <np.array>
        f, f_asym <matplotlib plots>
    """
    # Convert to class prediction from NN_util
    acc, prediction, ground_truth = nn_util_network_output_to_class_prediction_acc(outputs=outputs,
                                                                                   groundtruths=Ys_test,
                                                                                   class_type=class_type,
                                                                                   accuracy=accuracy)

    # Convert to numpy
    ground_truth_np = ground_truth.cpu().detach().numpy()
    ground_truth_bin_np = Ys_bin_test.cpu().detach().numpy()
    prediction_np = prediction.cpu().detach().numpy()

    # Get confusion matrix
    if class_type == 'multi':
        cnf = confusion_matrix(ground_truth_np, prediction_np)
    elif class_type == 'binary':
        cnf = confusion_matrix(ground_truth_bin_np, prediction_np)

    # Get asymmetric confusion matrix
    asym_cnf, asym_cnf_ratio = nn_util_asymmetric_confusion_matrix(ground_truth_np, prediction_np, mode=class_type)

    # Get fpr, fnr, tpr, tnr
    cnf_per_class = nn_util_read_confusion_matrix(cnf)

    # Plot confusion matrix
    f, ax = plt.subplots()
    sn.set(font_scale=1)
    ax = sn.heatmap(cnf, annot=True, annot_kws={'size': 8}, ax=ax, fmt='d', cmap="crest")
    ax.set(ylabel="Actual Classes", xlabel='Predicted Classes')

    f_asym, ax = plt.subplots()
    sn.set(font_scale=1)
    ax = sn.heatmap(asym_cnf, annot=True, annot_kws={'size': 8}, ax=ax, fmt='d', cmap="crest")
    ax.set(ylabel="Actual Classes", xlabel='Predicted Classes')

    return cnf, cnf_per_class, asym_cnf, asym_cnf_ratio, f, f_asym, acc


class EarlyStopper:
    def __init__(self, mode='not-best', tolerance=5):
        """
        mode:
            non-increasing: does not improve accuracy from past <tolerance> number of epochs
            not-best: does not improve the best accuracy for <tolerance> number of epochs
        """
        self._mode = mode
        self._tolerance = tolerance

        self.is_early_stop = False  # Callback boolean
        self.tolerance_counter = 0
        self._stored_early_stopping_variable = 0

        self._report_df = self.init_report_df()

    def init_report_df(self):
        report_df = pd.DataFrame(index=[0])
        report_df['Early Stopper Mode'] = self._mode
        report_df['Early Stopper Tolerance'] = self._tolerance
        return report_df

    def step(self, early_stopping_variable):
        """
        Will take new val_acc, and early stop depending on mode
        """
        if self._mode == 'non-increasing':
            if early_stopping_variable >= self._stored_early_stopping_variable:
                self.tolerance_counter = 0
            else:
                self.tolerance_counter += 1
            self._stored_early_stopping_variable = early_stopping_variable
        elif self._mode == 'not-best':
            if early_stopping_variable >= self._stored_early_stopping_variable:
                self.tolerance_counter = 0
                self._stored_early_stopping_variable = early_stopping_variable
            else:
                self.tolerance_counter += 1

        # Determine if early terminates
        if self.tolerance_counter > self._tolerance:
            self.is_early_stop = True

    @property
    def early_stopping_variable(self):
        return self._stored_early_stopping_variable

    @property
    def mode(self):
        return self._mode

    @property
    def tolerance(self):
        return self._tolerance

    @property
    def report_df(self):
        self._report_df['Early Stopped?'] = self.is_early_stop
        return self._report_df
