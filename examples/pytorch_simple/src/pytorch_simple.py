"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch and MNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole MNIST dataset, we here use a small
subset of it.

Modified to run on Amazon SageMaker. The original version is here: 
https://github.com/optuna/optuna/blob/master/examples/pytorch_simple.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

import os
import json
import argparse
import logging
import sys
from secrets import get_secret

import optuna

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 10
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_TEST_EXAMPLES = BATCHSIZE * 10

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def define_model(trial):
    # We optimize the number of layers, hidden untis and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_uniform("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def get_mnist(args):
    # Load MNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_dir, train=True, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_dir, train=False, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    return train_loader, test_loader


def objective(trial):

    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_uniform("lr", 1e-5, 1e-1)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the MNIST dataset.
    train_loader, test_loader = get_mnist(args)

    # Training of the model.
    model.train()
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)

            # Zeroing out gradient buffers.
            optimizer.zero_grad()
            # Performing a forward pass.
            output = model(data)
            # Computing negative Log Likelihood loss.
            loss = F.nll_loss(output, target)
            # Performing a backward pass.
            loss.backward()
            # Updating the weights.
            optimizer.step()

    save_model(model, '/tmp', trial.number)
    
    # Validation of the model.
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # Limiting testing data.
            if batch_idx * BATCHSIZE >= N_TEST_EXAMPLES:
                break
            data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability.
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / N_TEST_EXAMPLES
    
    trial.set_user_attr('accuracy', accuracy)
    trial.set_user_attr('job_name', args.training_env['job_name'])
    return accuracy

def model_fn(model_dir):
    from optuna.trial import FixedTrial
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = torch.load(os.path.join(model_dir, 'params.pth'))
    model = define_model(FixedTrial(params)).to(device)
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

def save_model(model, model_dir, trial_number):
    logger.info("Saving the model_{}.".format(trial_number))
    path = os.path.join(model_dir, 'model_{}.pth'.format(trial_number))
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # To configure Optuna db
    parser.add_argument('--host', type=str)
    parser.add_argument('--db-name', type=str, default='optuna')
    parser.add_argument('--db-secret', type=str, default='demo/optuna/db')
    parser.add_argument('--study-name', type=str, default='chainer-simple')
    parser.add_argument('--region-name', type=str, default='us-east-1')
    parser.add_argument('--n-trials', type=int, default=100)
    
    # Data, model, and output directories These are required.
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--training-env', type=str, default=json.loads(os.environ['SM_TRAINING_ENV']))
    
    args, _ = parser.parse_known_args()    

    secret = get_secret(args.db_secret, args.region_name)
    connector = 'mysqlconnector'
    db = 'mysql+{}://{}:{}@{}/{}'.format(connector, secret['username'], secret['password'], args.host, args.db_name)

    study = optuna.study.load_study(study_name=args.study_name, storage=db)
    study.optimize(objective, n_trials=args.n_trials)

    logger.info("Number of finished trials: {}".format(len(study.trials)))

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value: {}".format(trial.value))

    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))
        
    # retrieve and save the best model
    from optuna.trial import FixedTrial
    try:
        model = define_model(FixedTrial(trial.params)).to(DEVICE)
        with open(os.path.join('/tmp', 'model_{}.pth'.format(trial.number)), 'rb') as f:
            model.load_state_dict(torch.load(f))
            
        path = os.path.join(args.model_dir, 'model.pth')
        torch.save(model.cpu().state_dict(), path)
        torch.save(trial.params, os.path.join(args.model_dir, 'params.pth'))
        logger.info('    Model saved: model_{}.npz'.format(trial.number))
    except Exception as e: 
        logger.info('    Save failed: {}'.format(e))