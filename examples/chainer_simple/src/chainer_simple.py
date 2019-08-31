# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import os
import json 
from secrets import get_secret

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers
import numpy as np
import pkg_resources

if pkg_resources.parse_version(chainer.__version__) < pkg_resources.parse_version('4.0.0'):
    raise RuntimeError('Chainer>=4.0.0 is required for this example.')


# N_TRAIN_EXAMPLES = 3000
# N_TEST_EXAMPLES = 1000
BATCHSIZE = 128
EPOCH = 10


def create_model(trial):
    # We optimize the numbers of layers and their units.
    n_layers = trial.suggest_int('n_layers', 1, 3)

    layers = []
    for i in range(n_layers):
        n_units = int(trial.suggest_loguniform('n_units_l{}'.format(i), 4, 128))
        layers.append(L.Linear(None, n_units))
        layers.append(F.relu)
    layers.append(L.Linear(None, 10))

    return chainer.Sequential(*layers)

def create_optimizer(trial, model):
    # We optimize the choice of optimizers as well as their parameters.
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'MomentumSGD'])
    if optimizer_name == 'Adam':
        adam_alpha = trial.suggest_loguniform('adam_alpha', 1e-5, 1e-1)
        optimizer = chainer.optimizers.Adam(alpha=adam_alpha)
    else:
        momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
        optimizer = chainer.optimizers.MomentumSGD(lr=momentum_sgd_lr)

    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    return optimizer


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    # Model and optimizer
    model = L.Classifier(create_model(trial))
    optimizer = create_optimizer(trial, model)
    
    train_iter = chainer.iterators.SerialIterator(train, BATCHSIZE)
    test_iter = chainer.iterators.SerialIterator(test, BATCHSIZE, repeat=False, shuffle=False)

    # Trainer
    updater = chainer.training.StandardUpdater(train_iter, optimizer)
    trainer = chainer.training.Trainer(updater, (EPOCH, 'epoch'))
    trainer.extend(chainer.training.extensions.Evaluator(test_iter, model))
    log_report_extension = chainer.training.extensions.LogReport(log_name=None)
    trainer.extend(chainer.training.extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(log_report_extension)

    # Run!
    trainer.run()

    # Set the user attributes such as loss and accuracy for train and validation sets with SageMaker training job name. 
    log_last = log_report_extension.log[-1]
    for key, value in log_last.items():
        trial.set_user_attr(key, value)
        
    trial.set_user_attr('job_name', args.training_env['job_name'])
    serializers.save_npz(os.path.join('/tmp', 'model_{}.npz'.format(trial.number)), model)
    
    # Return the validation accuracy
    return log_report_extension.log[-1]['validation/main/accuracy']

def model_fn(model_dir):
    """
    This function is called by the Chainer container during hosting when running on SageMaker with
    values populated by the hosting environment.
    
    This function loads models written during training into `model_dir`.

    Args:
        model_dir (str): path to the directory containing the saved model artifacts

    Returns:
        a loaded Chainer model
    
    For more on `model_fn`, please visit the sagemaker-python-sdk repository:
    https://github.com/aws/sagemaker-python-sdk
    
    For more on the Chainer container, please visit the sagemaker-chainer-containers repository:
    https://github.com/aws/sagemaker-chainer-containers
    """
    
    from optuna.trial import FixedTrial
    
    chainer.config.train = False
    params = np.load(os.path.join(model_dir, 'params.npz'))['arr_0'].item()
    model = L.Classifier(create_model(FixedTrial(params)))
    serializers.load_npz(os.path.join(model_dir, 'model.npz'), model)
    return model.predictor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # for HPO
    parser.add_argument('--host', type=str)
    parser.add_argument('--db-name', type=str, default='optuna')
    parser.add_argument('--db-secret', type=str, default='demo/optuna/db')
    parser.add_argument('--study-name', type=str, default='chainer-simple')
    parser.add_argument('--n-trials', type=int, default=10)

    # Data, model, and output directories These are required.
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--training-env', type=str, default=json.loads(os.environ['SM_TRAINING_ENV']))
    parser.add_argument('--region-name', type=str, default='us-east-1')
    
    args, _ = parser.parse_known_args()
    
    num_gpus = int(os.environ['SM_NUM_GPUS'])
    
    # Load data downloaded from S3
    train_data = np.load(os.path.join(args.train, 'train.npz'))['data']
    train_labels = np.load(os.path.join(args.train, 'train.npz'))['labels']

    test_data = np.load(os.path.join(args.test, 'test.npz'))['data']
    test_labels = np.load(os.path.join(args.test, 'test.npz'))['labels']

    train = chainer.datasets.TupleDataset(train_data, train_labels)
    test = chainer.datasets.TupleDataset(test_data, test_labels)

    model_dir = args.model_dir
    
    # Define an Optuna study. 
    import optuna
    secret = get_secret(args.db_secret, args.region_name)
    connector = 'mysqlconnector'
    db = 'mysql+{}://{}:{}@{}/{}'.format(connector, secret['username'], secret['password'], args.host, args.db_name)
    
    study = optuna.study.load_study(study_name=args.study_name, storage=db)
    study.optimize(objective, n_trials=args.n_trials)

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    print('  User attrs:')
    for key, value in trial.user_attrs.items():
        print('    {}: {}'.format(key, value))

    # resave the best model
    try:
        model = L.Classifier(create_model(FixedTrial(trial.params)))
        serializers.load_npz(os.path.join('/tmp', 'model_{}.npz'.format(trial.number)), model)
        serializers.save_npz(os.path.join(model_dir, 'model.npz'), model)        
        np.savez(os.path.join(model_dir, 'params.npz'), trial.params)
        
        print('    Saved:')
    except Exception as e: 
        print('    Save failed:', e)