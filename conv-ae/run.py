import argparse
import yaml
import torch
import os
import pprint
import data
import modelae
import train
import evaluate

def read_from_config(config, item, default_value):
    try:
        value = config[item]
        if isinstance(default_value, float): return float(value)
        elif isinstance(default_value, int): return int(value)
        elif isinstance(default_value, bool): return bool(value)
        else:
            return value
    except:
        return default_value



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate an autoencoder model on MNIST dataset.')

    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('configfile', help='YAML file storing configuration information')
    args = parser.parse_args()

    try:
        with open(args.configfile, "r") as stream:
            try:
                config = yaml.safe_load(stream)
                pp = pprint.PrettyPrinter(indent=2)
                pp.pprint(config)
            except yaml.YAMLError as exc:
                print(exc)
    except:
        print(f'Cannot find configfile: {args.configfile}')

    datafolder = read_from_config(config, 'datafolder', '.')
    dataset = data.set_mnist_data(datafolder=datafolder, digits=[5])
    training_dataset, validation_dataset, test_dataset = data.split_training_validation_test(dataset, 0.5, 0.2, 0.3)

    ae = modelae.AutoEncoder() # This model lives outside the config file, and cannot be changed.

    device = read_from_config(config, 'device', -1)
    batch_size = read_from_config(config, 'batch-size', 16)
    pretrained_model_file = read_from_config(config, 'pretrained-model-file', '')

    if args.train:
        training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size, shuffle=False)
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size, shuffle=False)

        device = read_from_config(config, 'device', -1)
        training = train.Training(device)
        training.set_loss()
        training.set_model(ae)
        learning_rate = read_from_config(config, 'learning-rate', 1e-2)
        weight_decay = read_from_config(config, 'weight-decay', 1e5)
        training.set_optimizer(learning_rate=learning_rate, weight_decay=weight_decay)
        training.set_training_dataloader(training_dataloader)
        training.set_validation_dataloader(validation_dataloader)

        if pretrained_model_file and os.path.isfile(pretrained_model_file):
            training.load_checkpoint(pretrained_model_file)

        num_epoch = read_from_config(config, 'num-epoch', 1)
        show_loss = read_from_config(config, 'show-loss', True)
        checkpt_every = read_from_config(config, 'checkpt_every', -11)
        training.train(num_epoch=num_epoch, show_loss=show_loss, checkpt_every=checkpt_every) 
    else:
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

        inference = evaluate.Evaluate(device)
        inference.set_loss()
        inference.set_model(ae)

        pretrained_model_loaded = False
        if pretrained_model_file and os.path.isfile(pretrained_model_file):
            pretrained_model_loaded = inference.load_checkpoint(pretrained_model_file)

        if not pretrained_model_loaded:
            print(f'Cannot load pretrained weights: {pretrained_model_file}')
        else:
            print(f'Pretrained weights: {pretrained_model_file}')
            print(inference.evaluate(test_dataloader))
