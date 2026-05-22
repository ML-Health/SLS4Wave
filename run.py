import torch
import os
import logging
import argparse
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
from sklearn.metrics import roc_curve, auc,roc_auc_score
import pandas as pd
from S4model import *
from utils.filter import filter

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def calculate_f1_score(fp, tp, tn, fn):
    precision = tp / ((tp + fp) +0.000001)
    recall = tp / ((tp + fn) +0.000001)
    f1_score = 2 * (precision * recall) / (precision + recall+0.0001)
    return f1_score*100

def normalize_tensor_per_channel(tensor):

    normalized_tensor = torch.zeros_like(tensor)
    
    for channel in range(tensor.shape[1]):

        mean = tensor[:, channel, 72499:75000].mean(dim=1, keepdim=True)
        std = tensor[:, channel, 72499:75000].std(dim=1, keepdim=True)
        
        epsilon = 1e-5
        std = std + epsilon
        
        normalized_tensor[:, channel, :] = (tensor[:, channel, :] - mean) / std
        
    return normalized_tensor

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate S4-based classifier")
    parser.add_argument('--dataset', type=str, default='VTac', choices=['VTac','mimic','2015'], help='Dataset name')
    parser.add_argument('--data_dir', type=str, help='Base directory containing dataset .pt files')
    parser.add_argument('--sub_dataset', type=str, default='Ventricular_Tachycardia', help='2015 sub-dataset name')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epoch', type=int, default=60)
    parser.add_argument('--data_length', type=int, default=1250)
    parser.add_argument('--learning_rate', type=float, default=2e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--weighted_class', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pretrain', action='store_true', help='Load pretrained model weights if true')
    parser.add_argument('--pretrained_path', type=str, default='')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save models and logs')
    return parser.parse_args()


args = parse_args()
SEED = args.seed
# torch.backends.cudnn.enabled = False    
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

dataset = args.dataset
if dataset == 'mimic':
    # input shape (X,4,75000)
    base = args.data_dir
    train_path = os.path.join(base, 'train.pt')
    test_path = os.path.join(base, 'test.pt')
    val_path = os.path.join(base, 'val.pt')

    samples, alarm, groundtruth,_ = torch.load(train_path)
    samples_t, alarm_t, groundtruth_t,records = torch.load(test_path)
    samples_v, alarm_v, groundtruth_v,records = torch.load(val_path)

    samples = normalize_tensor_per_channel(samples)
    samples_t = normalize_tensor_per_channel(samples_t)
    samples_v = normalize_tensor_per_channel(samples_v)

    samples[:,2,:]=0            # MIMIC dataset has no PLTEH, set this channel to 0
    samples_t[:,2,:]=0
    samples_v[:,2,:]=0
    
if dataset == 'VTac':
    # input shape (X,4,75000)
    base = args.data_dir
    train_path = os.path.join(base, 'train.pt')
    test_path = os.path.join(base, 'test.pt')
    val_path = os.path.join(base, 'val.pt')

    samples, groundtruth, names = torch.load(train_path)
    samples_t, groundtruth_t,records = torch.load(test_path)
    samples_v,  groundtruth_v,records = torch.load(val_path)
    
if dataset == "2015":
    # input shape (X,4,37500)

    base = args.data_dir
    train_path = os.path.join(base, 'train.pt')
    test_path = os.path.join(base, 'test.pt')
    val_path = os.path.join(base, 'val.pt')

    train_path = args.train_path
    test_path = args.test_path
    val_path = args.val_path

    samples,groundtruth = torch.load(train_path)
    samples_t,groundtruth_t = torch.load(test_path)
    samples_v,groundtruth_v = torch.load(val_path)
    groundtruth = groundtruth.reshape(-1,1)
    groundtruth_t = groundtruth_t.reshape(-1,1)
    groundtruth_v = groundtruth_v.reshape(-1,1)

alarm=torch.zeros(groundtruth.shape)                # keep it for other 2015 datasets like VFIB, AFIB, etc.
alarm_t=torch.zeros(groundtruth_t.shape)
alarm_v=torch.zeros(groundtruth_v.shape)

finetune = args.pretrain
params_training = {
    'seed': SEED,
    'framework': 'S4',
    "differ_loss_weight": 1.5,
    'weighted_class': args.weighted_class,  # class weight
    'learning_rate': args.learning_rate,
    'adam_weight_decay': args.weight_decay,
    'batch_size': args.batch_size,   
    'max_epoch': args.max_epoch,
    'data_length': args.data_length,
    'dataset': dataset,
    "pretrain": finetune,
    "infor":"vtac pre d=2",
}

current_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

base_save_dir = args.save_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
model_path = os.path.join(base_save_dir, str(current_time), dataset)

if not os.path.exists(model_path):
    os.makedirs(model_path)
save_path = os.path.join(model_path, 'results.txt')

logger = get_logger(save_path, os.path.abspath(__file__))

logger.info(params_training)
if dataset == "2015":
    logger.info(args.sub_dataset)


model_save_path = os.path.join(model_path, f'model_{SEED}.pt')  

signal_train, signal_test = samples, samples_t
alarm_train, alarm_test = alarm, alarm_t
y_train, y_test = groundtruth, groundtruth_t

signal_val= samples_v
alarm_val= alarm_v
y_val= groundtruth_v

dataset_train = Dataset_train(signal_train, alarm_train, y_train)
dataset_eval = Dataset_train(signal_val, alarm_val, y_val)
dataset_test = Dataset_train(signal_test, alarm_test, y_test)

params = {'batch_size': params_training['batch_size'],
            'shuffle': False,
            'num_workers': 0 if os.name == 'posix' else 0}
eval_params = {'batch_size': params_training['batch_size'],
                'shuffle': False,
                'num_workers': 0 if os.name == 'posix' else 0}
iterator_train = DataLoader(dataset_train, **params)
iterator_val = DataLoader(dataset_eval, **eval_params)
iterator_test = DataLoader(dataset_test, **eval_params)

wavenet_config={
    "in_channels": 4,
    "out_channels": 1,
    "num_res_layers": 6,
    "res_channels": 256,
    "skip_channels": 256,
    "embed_dim_in": 256,
    "embed_dim_mid": 512,
    "embed_dim_out": 512,
    "s4_lmax": 1250,
    "s4_d_state": 64,
    "s4_dropout": 0.0,
    "s4_bidirectional": 1,
    "s4_layernorm": 1
}

model = SSSM(**wavenet_config)

if finetune==True:
    model=torch.load(args.pretrained_path,weights_only=False)
    def torchmodify(name) :
        a=name.split('.')
        for i,s in enumerate(a) :
            if s.isnumeric() :
                a[i]="_modules['"+s+"']"
        return '.'.join(a)
    
    for name, module in model.named_modules() :
        if isinstance(module,nn.GELU) :
            exec('model.'+torchmodify(name)+'=nn.GELU()')

    # if use multi-gpu pretrain
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

logger.info(model)
logger.info("Paramerters: {} M".format(sum(x.numel() for x in model.parameters()) / 1000000))
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=params_training['learning_rate'],
                                weight_decay=params_training['adam_weight_decay'])  # optimize all cnn parameters
loss_ce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([params_training['weighted_class']]).to(device))


num_epochs = params_training['max_epoch']

results_trainloss = []
results_evalloss = []
results_score = []
results_TPR = []
results_TNR = []
results_acc = []
results_F1 = []
results_PPV = []    
max_score = 0

min_eval_loss = float('inf')
for t in range(1, 1 + num_epochs):
    train_loss = 0
    differ_loss_val = 0
    model = model.train()
    train_TP, train_FP, train_TN, train_FN = 0, 0, 0, 0
    for b, batch in enumerate(tqdm(iterator_train),
                                start=1):  
        # signal_train, alarm_train, y_train, signal_test, alarm_test, y_test = batch
        loss, Y_train_prediction, y_train = train_model(batch, model, loss_ce, device, params_training['data_length'],
                                                                        weight=params_training["differ_loss_weight"])

        train_loss += loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()
        # scheduler.step()

    train_loss /= b
    differ_loss_val /= b
    y_tests=[]
    Y_eval_predictions=[]
    y_pred_final_all=[]
    y_test_final_all=[]
    eval_loss = 0
    model = model.eval()
    types_TP = [0, 0, 0, 0, 0]
    types_FP = [0, 0, 0, 0, 0]
    types_TN = [0, 0, 0, 0, 0]
    types_FN = [0, 0, 0, 0, 0]
    with torch.no_grad():

        for b, batch in enumerate(tqdm(iterator_val), start=1):
            loss, Y_eval_prediction, y_test, alarm_types = eval_model(batch, model, loss_ce, params_training['data_length'],device)
            types_TP, types_FP, types_TN, types_FN,pre = evaluation_test(alarm_types, Y_eval_prediction,
                                                                        y_test, types_TP, types_FP, types_TN, types_FN)
            eval_loss += loss.item()
            y_pred_final_all.extend(Y_eval_prediction.tolist())
            y_test_final_all.extend(y_test.tolist())
            
    # compute AUC

    auc = roc_auc_score(y_test_final_all, y_pred_final_all)*100

    eval_loss /= b
    acc = 100 * (sum(types_TP) + sum(types_TN)) / (sum(types_TP) + sum(types_TN) + sum(types_FP) + sum(types_FN))
    score = 100 * (sum(types_TP) + sum(types_TN)) / (
            sum(types_TP) + sum(types_TN) + sum(types_FP) + 5 * sum(types_FN))
    TPR = 100 * sum(types_TP) / (sum(types_TP) + sum(types_FN))
    TNR = 100 * sum(types_TN) / (sum(types_TN) + sum(types_FP))
    F1 = calculate_f1_score(sum(types_FP),sum(types_TP),sum(types_TN),sum(types_FN))
    PPV = 100 * sum(types_TP) / (sum(types_TP) + sum(types_FP)+1)
    if score > max_score:
        max_score = score
        torch.save(model.state_dict(), model_save_path)
        logger.info('model saved at ' + str(t)+ ' epoch' + ' with score ' + str(score) + ' and auc ' + str(auc))

    logger.info(params_training['framework'] + " Epoch " + str(t) +
                " train_loss: " + str(round(train_loss, 5)) +
                " eval_loss: " + str(round(eval_loss, 5)))
    logger.info('TPR: ' + str(round(TPR, 3)) + ' TNR: ' +
                str(round(TNR, 3)) + ' Score: ' + str(round(score, 3)) + ' Acc: ' + str(round(acc, 3))+' F1 :'+str(round(F1, 3))+ ' PPV :'+str(round(PPV, 3)))
    logger.info('AUC: ' + str(round(auc, 3)))
    results_trainloss.append(train_loss)
    results_evalloss.append(eval_loss)

    results_TPR.append(TPR)
    results_TNR.append(TNR)
    results_score.append(score)
    results_acc.append(acc)
    results_F1.append(F1)
    results_PPV.append(PPV)

    if t==num_epochs:
        model.load_state_dict(torch.load(model_save_path))
        y_tests=[]
        Y_eval_predictions=[]
        eval_loss = 0
        model = model.eval()
        types_TP = [0, 0, 0, 0, 0]
        types_FP = [0, 0, 0, 0, 0]
        types_TN = [0, 0, 0, 0, 0]
        types_FN = [0, 0, 0, 0, 0]
        y_pred_final_all=[]
        y_test_final_all=[]
        with torch.no_grad():

            for b, batch in enumerate(tqdm(iterator_test), start=1):
                loss, Y_eval_prediction, y_test, alarm_types = eval_model(batch, model, loss_ce,params_training['data_length'], device)
                types_TP, types_FP, types_TN, types_FN,pre = evaluation_test(alarm_types, Y_eval_prediction,
                                                                            y_test, types_TP, types_FP, types_TN, types_FN)
                eval_loss += loss.item()
                y_tests.append(y_test.cpu().numpy())
                Y_eval_predictions.append(pre.cpu().numpy())
                y_pred_final_all.extend(Y_eval_prediction.tolist())
                y_test_final_all.extend(y_test.tolist())

        auc_ = roc_auc_score(y_test_final_all, y_pred_final_all)*100
        eval_loss /= b
        acc = 100 * (sum(types_TP) + sum(types_TN)) / (sum(types_TP) + sum(types_TN) + sum(types_FP) + sum(types_FN))
        score = 100 * (sum(types_TP) + sum(types_TN)) / (
                sum(types_TP) + sum(types_TN) + sum(types_FP) + 5 * sum(types_FN))
        TPR = 100 * sum(types_TP) / (sum(types_TP) + sum(types_FN))
        TNR = 100 * sum(types_TN) / (sum(types_TN) + sum(types_FP))
        F1 = calculate_f1_score(sum(types_FP),sum(types_TP),sum(types_TN),sum(types_FN))
        PPV = 100 * sum(types_TP) / (sum(types_TP) + sum(types_FP)+1)    
        
        index = results_score.index(max(results_score))
        result = ' TPR: ' + str(round(results_TPR[index], 3)) + ' TNR: ' + \
                str(round(results_TNR[index], 3)) + ' Score: ' + str(round(results_score[index], 3)) + ' Acc: ' + str(
            round(results_acc[index], 3)) + ' F1: ' + str(round(results_F1[index], 3)) + ' PPV: ' + str(round(results_PPV[index], 3))+ ' AUC: ' + str(round(auc_, 4))


        Test_result='TPR: ' + str(round(TPR, 3)) + ' TNR: ' + \
            str(round(TNR, 3)) + ' Score: ' + str(round(score, 3)) + ' Acc: ' + str(
    round(acc, 3)) + ' F1: ' + str(round(F1, 3)) + ' PPV: ' + str(round(PPV, 3)) + ' AUC: ' + str(round(auc_, 5))

logger.info('\n')
logger.info(Test_result)
