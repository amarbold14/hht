import statistics

import torch
from model.torch_lstm import *
from shared.utility import *
from shared.data_config import *
from model.train_config import *
from shared.MyWriter import *
from shared.MyMetaTrainer import *
from copy import deepcopy


class MAML(nn.Module):
    """
    Args:
        backbone_model: e.g. OneLayerLSTM, this is initialized already
        backbone_train_config: e.g., LSTMConfig, unfortunately, the inner loop optimizers, criterion and scheduler are all inside
        meta_train_config<MetaTrainConfig>
        data_loader<NwayKshotDataLoader>
    """

    def __init__(self, backbone_model, backbone_train_config, meta_train_config, data_loader, logger, device):
        super(MAML, self).__init__()
        self.backbone_model = backbone_model
        self.meta_train_config = meta_train_config
        self.backbone_train_config = backbone_train_config
        self.data_loader = data_loader

        # Parse inner loop train config
        self.inner_opt = self.backbone_train_config.get_optimizer(model=self.backbone_model)
        self.inner_scheduler = self.backbone_train_config.get_scheduler()  # Not used
        self.criterion = self.backbone_train_config.get_criterion()
        self.class_type = self.data_loader.class_type

        # Parse meta loop train config
        self.meta_opt = self.meta_train_config.get_meta_optimizer(model=self.backbone_model)
        self.meta_scheduler = self.meta_train_config.get_scheduler()
        self.num_inner_loop_per_epoch = self.meta_train_config.num_inner_loop_per_epoch
        self.num_tasks_per_inner_loop = self.meta_train_config.num_tasks_per_inner_loop
        self.num_steps_before_query = self.meta_train_config.num_steps_before_query

        # Parse logger
        self._logger = logger
        self.accuracy = Accuracy().to(device)

    def train(self):
        """
        Maml train:
            follow https://www.youtube.com/watch?v=9XqP7zhYbMQ&ab_channel=anucvml
            follow fsl_ts
        Returns:
            query_loss: accumulated query loss
        """
        # Initialize parameters to be passed upstream
        losses_query, accs_query = [], []
        self.meta_opt.zero_grad()

        # we can have multiple tasks per every inner loop
        for task_i in range(self.num_tasks_per_inner_loop['train']):
            with higher.innerloop_ctx(self.backbone_model, self.inner_opt, copy_initial_weights=False) as (
                    fnet, diffopt):
                x_support, y_support, x_query, y_query = self.data_loader.load_task_data(type_of_data='train')

                # in MAML we can update a few times before 'test' on query
                for _ in range(self.num_steps_before_query['train']):
                    outputs = fnet(x_support)
                    loss_support = self.criterion(outputs, y_support)
                    diffopt.step(loss_support)
                    acc_support, _, _ = nn_util_network_output_to_class_prediction_acc(outputs=outputs,
                                                                                       groundtruths=y_support,
                                                                                       class_type=self.class_type,
                                                                                       accuracy=self.accuracy)

                # Compute query loss
                outputs_query = fnet(x_query)
                loss_query = self.criterion(outputs_query, y_query)
                losses_query.append(loss_query.detach())
                # Update MAML with query loss, accumulates to self.meta_opt
                loss_query.backward()

                # Assess query set accuracy
                acc_query, _, _ = nn_util_network_output_to_class_prediction_acc(outputs=outputs_query,
                                                                                 groundtruths=y_query,
                                                                                 class_type=self.class_type,
                                                                                 accuracy=self.accuracy)
                accs_query.append(acc_query)

        # Step the accumulated query loss
        self.meta_opt.step()

        # Report a query loss to trainer
        avg_query_loss = sum(losses_query) / len(losses_query)
        avg_query_acc = sum(accs_query) / len(accs_query)
        return avg_query_loss.item(), avg_query_acc.item()

    def _fine_tune(self, type_of_data, x_support, y_support, x_query, y_query):
        """
        Args:
            type_of_data: 'val', 'test'
        """

        # Create a copy of the current backbone model
        eval_model = deepcopy(self.backbone_model)
        eval_inner_opt = self.backbone_train_config.get_optimizer(model=eval_model)

        # Follows the same process as self.train()
        self.meta_opt.zero_grad()

        # Train on support
        for _ in range(self.num_steps_before_query[type_of_data]):
            outputs = eval_model(x_support)
            loss_support = self.criterion(outputs, y_support)

            # Backprop
            eval_inner_opt.zero_grad()
            loss_support.backward()
            eval_inner_opt.step()

            acc_support, _, _ = nn_util_network_output_to_class_prediction_acc(outputs=outputs,
                                                                               groundtruths=y_support,
                                                                               class_type=self.class_type,
                                                                               accuracy=self.accuracy)

        # Compute query loss
        with torch.no_grad():
            outputs_query = eval_model(x_query)
            loss_query = self.criterion(outputs_query, y_query)

            # Assess query set accuracy
            acc_query, _, _ = nn_util_network_output_to_class_prediction_acc(outputs=outputs_query,
                                                                             groundtruths=y_query,
                                                                             class_type=self.class_type,
                                                                             accuracy=self.accuracy)

        loss_support_detached = loss_support.detach().item()
        acc_support_detached = acc_support.item()
        loss_query_detached = loss_query.detach().item()
        acc_query_detached = acc_query.item()

        return loss_support_detached, acc_support_detached, loss_query_detached, acc_query_detached, eval_model
        #return loss_support.detach().item(), acc_support.item(), loss_query.detach().item(), acc_query.item(), eval_model

    @staticmethod
    def _compute_mean_std_of_fine_tune(losses_support, accs_support, losses_query, accs_query):
        avg_support_loss = statistics.mean(losses_support)
        avg_query_loss = statistics.mean(losses_query)

        # Accuracies
        avg_support_acc = statistics.mean(accs_support)
        std_support_acc = statistics.stdev(accs_support)
        avg_query_acc = statistics.mean(accs_query)
        std_query_acc = statistics.stdev(accs_query)
        return avg_support_loss, avg_support_acc, std_support_acc, avg_query_loss, avg_query_acc, std_query_acc

    def validate(self):
        """
        Validation
        """
        # Initialize parameters to be passed upstream
        type_of_data = 'val'
        losses_support, losses_query, accs_support, accs_query = [], [], [], []

        for task_i in range(self.num_tasks_per_inner_loop[type_of_data]):
            # data_loader loads val or test differently
            x_support, y_support, x_query, y_query = self.data_loader.load_task_data(type_of_data=type_of_data)

            # Evaluate
            loss_support, acc_support, loss_query, acc_query, _ = self._fine_tune(type_of_data=type_of_data,
                                                                                  x_support=x_support,
                                                                                  x_query=x_query,
                                                                                  y_support=y_support, y_query=y_query)
            losses_support.append(loss_support)
            losses_query.append(loss_query)
            accs_query.append(acc_query)
            accs_support.append(acc_support)

        # Report a query loss to trainer
        return self._compute_mean_std_of_fine_tune(losses_support=losses_support,
                                                   accs_support=accs_support,
                                                   losses_query=losses_query,
                                                   accs_query=accs_query)

    def test(self, use_all_remain_data_for_test=True):
        """
        Args:
            use_all_remain_data_for_test: <bool> whether during meta-test we will use all remaining data for testing,
                                                or the number of query will be the same as test and validation.
        Returns:
            best_model, worst_model: <dict> with fields 'model', 'acc_query', 'x_query', 'y_query', 'acc_support'
        """
        type_of_data = 'test'
        number_of_tasks = self.num_tasks_per_inner_loop[type_of_data]

        # Reduce the number of tasks needed if we are using all data for testing
        if use_all_remain_data_for_test:
            number_of_tasks = number_of_tasks//2

        # These are extra things to return upstream
        best_acc_query, worst_acc_query = 0, 1
        best_acc_support, worst_acc_support = 0, 1
        best_model, worst_model = {}, {}
        losses_support, losses_query, accs_support, accs_query = [], [], [], []

        for task_i in range(number_of_tasks):
            self._logger.debug("Evaluating test task {}".format(task_i))

            # data_loader loads val or test differently
            x_support, ys_support, yo_support, yb_support, x_query, ys_query, yo_query, yb_query = self.data_loader.load_task_data(
                type_of_data=type_of_data, use_all_remain_data_for_test=use_all_remain_data_for_test)

            # Parse labels based on the type of task
            if self.class_type == 'binary':
                y_support = yb_support
                y_query = yb_query
            elif self.class_type == 'multi':
                y_support = ys_support
                y_query = ys_query

            del yb_support, ys_support, ys_query, yo_support, yo_query  # yb_query is kept as the ground-truth label

            # Evaluate
            loss_support, acc_support, loss_query, acc_query, model = self._fine_tune(type_of_data=type_of_data,
                                                                                      x_support=x_support,
                                                                                      x_query=x_query,
                                                                                      y_support=y_support,
                                                                                      y_query=y_query)
            losses_support.append(loss_support)
            losses_query.append(loss_query)
            accs_query.append(acc_query)
            accs_support.append(acc_support)

            # Record best and worst model
            if acc_query >= best_acc_query:
                best_model['model'] = model
                best_model['x_query'] = x_query
                best_model['y_query'] = y_query
                best_model['yb_query'] = yb_query
                best_acc_query = acc_query
                best_acc_support = acc_support
            if worst_acc_query >= acc_query:
                worst_model['model'] = model
                worst_model['x_query'] = x_query
                worst_model['y_query'] = y_query
                worst_model['yb_query'] = yb_query
                worst_acc_query = acc_query
                worst_acc_support = acc_support

        best_model['acc_query'] = best_acc_query
        best_model['acc_support'] = best_acc_support
        worst_model['acc_query'] = worst_acc_query
        worst_model['acc_support'] = worst_acc_support

        avg_support_loss, avg_support_acc, std_support_acc, avg_query_loss, avg_query_acc, std_query_acc = self._compute_mean_std_of_fine_tune(
            losses_support=losses_support,
            accs_support=accs_support,
            losses_query=losses_query,
            accs_query=accs_query)

        # Report a query loss to trainer
        return avg_support_loss, avg_support_acc, std_support_acc, avg_query_loss, avg_query_acc, std_query_acc, best_model, worst_model, number_of_tasks


if __name__ == "__main__":  # Test and usage example of MAML
    # Test cases
    test_window_length = 200
    test_spe = 200  # Use -1 if you have a crazy amount of memory :), otherwise 200 for testing
    test_batch_size = 64
    class_type = 'binary'
    test_train_class_number = {'H': 100,
                               'N': 100,
                               'NV': 100,
                               'D': 100}

    # Initialize model configuration
    train_config = LSTMConfig(LSTM_hidden_size=30,
                              n_epochs=100,
                              seed=42,
                              exp_lr_gamma=0.99,
                              early_stop_mode='not-best',
                              early_stop_tolerance=5,
                              criterion_type='BCE',
                              optimizer_type='adam',
                              lr=[0.00001])

    # Initialize logger, random and device
    SV, logger, device = util_init_SV_logger_device_seed('MAML-TEST', seed=train_config.seed)

    # Initialize data configuration
    data_config = NwayKshotDataConfig(window_length=test_window_length,
                                      sample_per_experiment=test_spe,
                                      class_type=class_type,
                                      k_support=5,
                                      k_query=5,
                                      meta_test_ways=('H', 'N', 'NV'),
                                      meta_train_class_number=test_train_class_number)

    # Wrap all these into a Config class
    meta_train_config = MAMLTrainConfig(meta_lr=[0.001])

    # Initialize DataLoader and Writer
    data_loader = NwayKshotDataLoader(data_config=data_config, logger=logger, device=device)

    writer = MetaWriter(meta_train_config=meta_train_config, train_config=train_config, data_config=data_config,
                        logger=logger)

    # Declare backbone model
    model = BinaryOneLayerLSTM(data_loader=data_loader, train_config=train_config).to(device)
    maml_model = MAML(backbone_model=model, backbone_train_config=train_config, meta_train_config=meta_train_config,
                      data_loader=data_loader)

    # Initialize Trainer
    Trainer = MAMLTrainer(meta_model=maml_model, data_config=data_config, logger=logger, device=device,
                          writer=writer, data_loader=data_loader)

    # Train
    Trainer.train()
