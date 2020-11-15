import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision

# To view tensorboard metrics
# tensorboard --logdir=logs --port=6006 --bind_all
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from evolver import CrossoverType, MutationType, VectorEvolver
from unet import UNet
from dataset_utils import PartitionType
from cuda_utils import maybe_get_cuda_device, clear_cuda
from landcover_dataloader import get_landcover_dataloaders

from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, ConfusionMatrix, mIoU
from ignite.handlers import ModelCheckpoint
from ignite.utils import setup_logger
from ignite.engine import Engine


# Define directories for data, logging and model saving.
base_dir = os.getcwd()
dataset_name = "landcover_large"
dataset_dir = os.path.join(base_dir, "data/" + dataset_name)

experiment_name = "backprop_single_point_finetuning_frozen_batchnorm_ReduceLROnPlateau_0001"
model_name = "best_model_9_validation_accuracy=0.8940.pt"
model_path = os.path.join(base_dir, "logs/" + dataset_name + "/" + model_name)
log_dir = os.path.join(base_dir, "logs/" + dataset_name + "_" + experiment_name)

# Create DataLoaders for each partition of Landcover data.
dataloader_params = {
    'batch_size': 16,
    'shuffle': True,
    'num_workers': 6,
    'pin_memory': True}

partition_types = [PartitionType.TRAIN, PartitionType.VALIDATION, 
                   PartitionType.FINETUNING, PartitionType.TEST]
data_loaders = get_landcover_dataloaders(dataset_dir, 
                                         partition_types,
                                         dataloader_params,
                                         force_create_dataset=False)

finetuning_loader = data_loaders[2]
test_loader = data_loaders[3]

# Get GPU device if available.
device = maybe_get_cuda_device()

# Determine model and training params.
params = {
    'max_epochs': 10,
    'n_classes': 4,
    'in_channels': 4,
    'depth': 5,
    'learning_rate': 0.0001,
    'log_steps': 1,
    'save_top_n_models': 4
}

clear_cuda()    
model = UNet(in_channels = params['in_channels'],
             n_classes = params['n_classes'],
             depth = params['depth'])
model.load_state_dict(torch.load(model_path))

model.to(device)

# Create Trainer or Evaluators
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=params['learning_rate'])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


# Determine metrics for evaluation.
metrics = {
        "accuracy": Accuracy(), 
        "loss": Loss(criterion),
        "mean_iou": mIoU(ConfusionMatrix(num_classes = params['n_classes'])),
}

def backprop_step(engine, batch):
    model.eval()
    model.zero_grad()
    batch_x, batch_y = batch
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    outputs = model(batch_x)
    loss = criterion(outputs[:, :, 127:128,127:128], batch_y[:,127:128,127:128])
    loss.backward()
    optimizer.step()
    return loss.item()
 

# Create Trainer or Evaluators
trainer = Engine(backprop_step)
train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
validation_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

trainer.logger = setup_logger("Trainer")
train_evaluator.logger = setup_logger("Train Evaluator")
validation_evaluator.logger = setup_logger("Validation Evaluator")


# Tensorboard Logger setup below based on pytorch ignite example
# https://github.com/pytorch/ignite/blob/master/examples/contrib/mnist/mnist_with_tensorboard_logger.py
@trainer.on(Events.EPOCH_COMPLETED)
def compute_metrics(engine):
    """Callback to compute metrics on the train and validation data."""
    train_evaluator.run(finetuning_loader)
    validation_evaluator.run(test_loader)
    scheduler.step(validation_evaluator.state.metrics['loss'])

def score_function(engine):
    """Function to determine the metric upon which to compare model."""
    return engine.state.metrics["accuracy"]
    
# Setup Tensor Board Logging    
tb_logger = TensorboardLogger(log_dir=log_dir)

tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=params['log_steps']),
    tag="training",
    output_transform=lambda loss: {"batchloss": loss},
    metric_names="all",
)

for tag, evaluator in [("training", train_evaluator), ("validation", validation_evaluator)]:
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag=tag,
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )

tb_logger.attach_opt_params_handler(trainer, 
                                    event_name=Events.ITERATION_COMPLETED(every=params['log_steps']), 
                                    optimizer=optimizer)

tb_logger.attach(trainer, log_handler=WeightsScalarHandler(model), 
                 event_name=Events.ITERATION_COMPLETED(every=params['log_steps']))

tb_logger.attach(trainer, log_handler=WeightsHistHandler(model), 
                 event_name=Events.EPOCH_COMPLETED(every=params['log_steps']))

tb_logger.attach(trainer, log_handler=GradsScalarHandler(model), 
                 event_name=Events.ITERATION_COMPLETED(every=params['log_steps']))

tb_logger.attach(trainer, log_handler=GradsHistHandler(model), 
                 event_name=Events.EPOCH_COMPLETED(every=params['log_steps']))

model_checkpoint = ModelCheckpoint(
    log_dir,
    n_saved=params['save_top_n_models'],
    filename_prefix="best",
    score_function=score_function,
    score_name="validation_accuracy",
    global_step_transform=global_step_from_engine(trainer),
)

validation_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})
trainer.run(finetuning_loader, max_epochs=params['max_epochs'])
tb_logger.close()
