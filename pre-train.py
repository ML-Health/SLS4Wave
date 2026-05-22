import torch
import os
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tools import *
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import pandas as pd
from S4model import *
from utils.nt_xent import NTXentLoss

import argparse

parser = argparse.ArgumentParser(description="Pre-train model parameters")
parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate for the optimizer')
parser.add_argument('--adam_weight_decay', type=float, default=0.0001, help='Weight decay for Adam optimizer')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--max_epoch', type=int, default=50, help='Maximum number of epochs')
parser.add_argument('--data_length', type=int, default=1250, help='Length of the input data')
parser.add_argument('--infor', type=str, default="", help='Additional information')
parser.add_argument('--weighted_class', type=float, default=1.0, help='Weight for the positive class in BCE loss')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint file for resuming training')

args = parser.parse_args()

def calculate_f1_score(fp, tp, tn, fn):
    precision = tp / ((tp + fp) +0.000001)
    recall = tp / ((tp + fn) +0.000001)
    f1_score = 2 * (precision * recall) / (precision + recall+0.0001)
    return f1_score*100


SEED = 1234567

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


samples, alarm, groundtruth,_ = torch.load(
    'datasets/pre-train.pt',weights_only=False)

params_training = {
    'learning_rate': args.learning_rate,
    'adam_weight_decay': args.adam_weight_decay,
    'batch_size': args.batch_size,
    'max_epoch': args.max_epoch,
    'data_length': args.data_length,
    'weighted_class': args.weighted_class,
    'infor':""
}



current_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', str(current_time)+'_pretrain')

if not os.path.exists(model_path):
    os.makedirs(model_path)

save_path = os.path.join(model_path, 'results.txt')

logger = get_logger(logpath=save_path, filepath=os.path.abspath(__file__))

logger.info(params_training)

model_save_path = os.path.join(model_path, 'model_pretrain.pt')

dataset_train = Dataset_train(samples, alarm, groundtruth)

params = {'batch_size': params_training['batch_size'],
          'shuffle': True,
          'num_workers': 4 if os.name == 'posix' else 0}

iterator_train = DataLoader(dataset_train, **params)

wavenet_config={
    "in_channels": 4,
    "out_channels": 1,
    "num_res_layers": 6,
    "res_channels": 256,
    "skip_channels": 256,
    "diffusion_step_embed_dim_in": 256,
    "diffusion_step_embed_dim_mid": 512,
    "diffusion_step_embed_dim_out": 512,
    "s4_lmax": 1250,
    "s4_d_state": 64,
    "s4_dropout": 0.1,
    "s4_bidirectional": 1,
    "s4_layernorm": 1
}

model = SSSM(**wavenet_config)


logger.info(model)
logger.info("Paramerters: {} M".format(sum(x.numel() for x in model.parameters()) / 1000000))
if torch.cuda.is_available():
    model = nn.DataParallel(model)

model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=params_training['learning_rate'],
                             weight_decay=params_training['adam_weight_decay'])  # optimize all cnn parameters
loss_ce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([params_training['weighted_class']]).to(device))
loss_obj = NTXentLoss(device, params_training['batch_size'], temperature=0.1, use_cosine_similarity=True)
num_epochs = params_training['max_epoch']

results_trainloss = []

start_epoch = 1
if args.checkpoint:
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    logger.info(f"Resumed training from checkpoint: {args.checkpoint}, starting at epoch {start_epoch}")


for t in range(1, 1 + num_epochs):
    train_loss = 0
    differ_loss_val = 0
    model = model.train()
    train_TP, train_FP, train_TN, train_FN = 0, 0, 0, 0
    for b, batch in enumerate(iterator_train, start=1):
        loss, _, _ = new_pretrain_model(batch, model, loss_obj, device,length=params_training['data_length'])
                                                                     

        train_loss += loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()
    train_loss /= b

    checkpoint_path = os.path.join(model_path, f'checkpoint_epoch_{t}.pt')
    torch.save({
        'epoch': t,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss
    }, checkpoint_path)
    logger.info(f"Checkpoint saved at {checkpoint_path}")
    if t%10==0:
        torch.save(model, model_save_path)
        logger.info("model saved")
        logger.info("model saved in Epoch: {}, Train loss: {}".format(t, train_loss))   
    results_trainloss.append(train_loss)
    logger.info("Epoch: {}, Train loss: {}".format(t, train_loss))
