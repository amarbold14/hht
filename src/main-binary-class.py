# Import torch-related dependencies

from model.resnet12 import *
from shared.MyTrainer import *
from shared.MyWriter import MyWriter
from shared.data_config import *
from shared.MyEvaluator import *
from shared.utility import SharedVariable
import torchvision

cwd = os.getcwd()

def main_binary_resnet(train_config, data_config):
    # Initialize logger, random and device
    _, logger, device = util_init_SV_logger_device_seed('Resnet-Supervised-Binary', seed=train_config.seed)

    # Initialize DataLoader and Writer
    data_loader = ArbitraryPortionDataLoader(data_config=data_config, device=device, logger=logger)
    writer = MyWriter(train_config=train_config, data_config=data_config, logger=logger)

    # Load data and process a few extra steps
    train_data, val_data, test_data, demo_data = data_loader.load_data()


    # Declare model
    model = torchvision.models.resnet50(num_classes=1).to(device)

    input_dim = data_loader.input_dimension
    sequence_len = data_loader.sequence_len
    output_dim = data_loader.output_dimension

    model.input_size = (3, 224, 224)
    model.H_in = 224
    model.L = 224
    model.H_out = output_dim

    # Train model
    Trainer = ModelTrainer(model=model, train_config=train_config, data_loader=data_loader,
                           device=device, logger=logger, writer=writer)
    Trainer.train(train_data=train_data, val_data=val_data)

    # Evaluate model
    Evaluator = ModelEvaluator(Trainer=Trainer)
    Evaluator.evaluate(test_data=test_data, print_model_architecture=False)
    # Evaluator.evaluate_demo(demo_data=demo_data)

    # Release
    writer.close()
    os.chdir(cwd)


if __name__ == '__main__':
    # Iterators
    parameters = {'train_class_number_list': SharedVariable().default_train_class_number_list_bin,
                  'batch_size_list': [64],
                  'n_epoch_list': [100],
                  'early_stop_tolerance_list': [5],
                  'seed_list': [42]}

    # Batch run experiments
    for seed in parameters['seed_list']:
        for train_class_number in parameters['train_class_number_list']:
            for batch_size in parameters['batch_size_list']:
                for n_epoch in parameters['n_epoch_list']:
                    for early_stop_tolerance in parameters['early_stop_tolerance_list']:

                        # Initialize model configuration
                        train_config = ResNetConfig(
                                                  n_epochs=n_epoch,
                                                  seed=seed,
                                                  exp_lr_gamma=0.99,
                                                  early_stop_mode='not-best',
                                                  early_stop_tolerance=early_stop_tolerance,
                                                  criterion_type='BCE',
                                                  optimizer_type='adam',
                                                  lr=[0.00001])

                        # Initialize data configuration
                        data_config = ArbitraryDataConfig(window_length=2000,
                                                          sample_per_experiment=SharedVariable().default_spe,
                                                          class_type='binary',
                                                          batch_size=16,
                                                          train_class_number=train_class_number)
                        # Run
                        main_binary_resnet(train_config=train_config, data_config=data_config)