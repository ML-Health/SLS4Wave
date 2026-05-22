import torch
import os
import logging
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from nets import *
import time
from tools import *
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import pandas as pd
from S4model import *

print(f"CUDA AVAILABLE: {torch.cuda.is_available()}")
def calculate_f1_score(fp, tp, tn, fn):
    precision = tp / ((tp + fp) +0.000001)
    recall = tp / ((tp + fn) +0.000001)
    f1_score = 2 * (precision * recall) / (precision + recall+0.0001)
    return f1_score*100

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate on 2015 dataset")
    parser.add_argument('--dataset', type=str, default='2015', help='2015 or mimic')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing train.pt/val.pt/test.pt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--data_length', type=int, default=1250)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--weighted_class', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrained_path', type=str, default='')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--sub_dataset', type=str, default='Ventricular_Tachycardia', help='sub-dataset name')
    return parser.parse_args()

args = parse_args()
SEED = args.seed
# torch.backends.cudnn.enabled = False    # disable cuDNN
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True  # ensure experiment reproducibility
    torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base = args.data_dir
train_path = os.path.join(base, 'train.pt')
test_path = os.path.join(base, 'test.pt')
val_path = os.path.join(base, 'val.pt')

samples, alarm, groundtruth = torch.load(train_path, weights_only=False)
samples_t, alarm_t, groundtruth_t = torch.load(test_path, weights_only=False)
samples_v, alarm_v, groundtruth_v = torch.load(val_path, weights_only=False)

finetune = args.pretrain
params_training = {
    'seed': SEED,
    'framework': 'S4',
    "differ_loss_weight": 1.5,
    'weighted_class': args.weighted_class,
    'learning_rate': args.learning_rate,
    'adam_weight_decay': args.weight_decay,
    'batch_size': args.batch_size,   
    'max_epoch': args.max_epoch,
    'data_length': args.data_length,
    'dataset': args.dataset,  
    "pretrain": finetune,
    'data_dir': args.data_dir,
}


current_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

base_save_dir = args.save_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
model_path = os.path.join(base_save_dir, str(current_time), params_training['dataset'])

if not os.path.exists(model_path):
    os.makedirs(model_path)
save_path = os.path.join(model_path, 'results.txt')

logger = get_logger(save_path, os.path.abspath(__file__))

logger.info(params_training)

model_save_path = os.path.join(model_path, f'model_{SEED}.pt')

VT_index=[]
VFIB_index=[]
for i in range(len(samples)):
    if alarm[i][0]==1 :
        VT_index.append(i)
    if alarm[i][0]==0 :
        VFIB_index.append(i)  

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
            'num_workers': 8 if os.name == 'posix' else 0}
eval_params = {'batch_size': params_training['batch_size'],
                'shuffle': False,
                'num_workers': 8 if os.name == 'posix' else 0}
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
    "s4_dropout": 0.1,
    "s4_bidirectional": 1,
    "s4_layernorm": 1
}

model = SSSM(**wavenet_config)

class CompatGELU(nn.GELU):
    """
    A GELU module that is compatible with older versions of PyTorch.
    Model was saved in torch 1.x and loaded in torch 2.x
    """
    def __init__(self, *args, **kwargs):
        self.approximate = kwargs.pop('approximate', None)
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return F.gelu(x)
    
    def extra_repr(self):
        # so logger.info(model) stops crashing
        return f"approximate={repr(self.approximate)}"


def patch_gelu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, CompatGELU())
        else:
            patch_gelu(child)

if finetune==True:
    model=torch.load(args.pretrained_path, weights_only=False)
    patch_gelu(model)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

logger.info(model)
logger.info("Paramerters: {} M".format(sum(x.numel() for x in model.parameters()) / 1000000))
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=params_training['learning_rate'],
                                weight_decay=params_training['adam_weight_decay'])  
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
        loss, Y_train_prediction, y_train = train_model(batch, model, loss_ce, device, length=args.data_length, weight=params_training["differ_loss_weight"])

        train_loss += loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

    train_loss /= b
    differ_loss_val /= b
    y_tests=[]
    Y_eval_predictions=[]
    eval_loss = 0
    model = model.eval()
    types_TP = [0, 0, 0, 0, 0]
    types_FP = [0, 0, 0, 0, 0]
    types_TN = [0, 0, 0, 0, 0]
    types_FN = [0, 0, 0, 0, 0]
    with torch.no_grad():

        for b, batch in enumerate(tqdm(iterator_val), start=1):
            loss, Y_eval_prediction, y_test, alarm_types = eval_model(batch, model, loss_ce, args.data_length, device)
            types_TP, types_FP, types_TN, types_FN,pre = evaluation_test(alarm_types, Y_eval_prediction,
                                                                        y_test, types_TP, types_FP, types_TN, types_FN)
            eval_loss += loss.item()
            y_tests.append(y_test.cpu().numpy())
            Y_eval_predictions.append(pre.cpu().numpy())

    y_tests=np.concatenate(y_tests,0)
    Y_eval_predictions=np.concatenate(Y_eval_predictions,0)
    fpr, tpr, thresholds = roc_curve(y_tests, Y_eval_predictions)
    roc_auc = auc(fpr, tpr)

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
        logger.info('save model!!\n')

    logger.info(params_training['framework'] + " Epoch " + str(t) +
                " train_loss: " + str(round(train_loss, 5)) +
                " eval_loss: " + str(round(eval_loss, 5)))
    logger.info('TPR: ' + str(round(TPR, 3)) + ' TNR: ' +
                str(round(TNR, 3)) + ' Score: ' + str(round(score, 3)) + ' Acc: ' + str(round(acc, 3))+' F1 :'+str(round(F1, 3))+ ' PPV :'+str(round(PPV, 3)))
    logger.info('AUC: ' + str(round(roc_auc, 3)))
    results_trainloss.append(train_loss)
    results_evalloss.append(eval_loss)

    results_TPR.append(TPR)
    results_TNR.append(TNR)
    results_score.append(score)
    results_acc.append(acc)
    results_F1.append(F1)
    results_PPV.append(PPV)

    if t==num_epochs:
        model.load_state_dict(torch.load(model_save_path),weights_only=False)
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
                loss, Y_eval_prediction, y_test, alarm_types = eval_model(batch, model, loss_ce, args.data_length, device)
                types_TP, types_FP, types_TN, types_FN,pre = evaluation_test(alarm_types, Y_eval_prediction,
                                                                            y_test, types_TP, types_FP, types_TN, types_FN)
                eval_loss += loss.item()
                y_tests.append(y_test.cpu().numpy())
                Y_eval_predictions.append(pre.cpu().numpy())
                y_pred_final_all.extend(Y_eval_prediction.tolist())
                y_test_final_all.extend(y_test.tolist())

        y_tests=np.concatenate(y_tests,0)
        Y_eval_predictions=np.concatenate(Y_eval_predictions,0)
        fpr, tpr, thresholds = roc_curve(y_tests, Y_eval_predictions)
        roc_auc = auc(fpr, tpr)

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
            round(results_acc[index], 3)) + ' F1: ' + str(round(results_F1[index], 3)) + ' PPV: ' + str(round(results_PPV[index], 3))+ ' AUC: ' + str(round(roc_auc, 4))

        TP_str = ', '.join(map(str, types_TP))
        TN_str = ', '.join(map(str, types_TN))
        FP_str = ', '.join(map(str, types_FP))
        FN_str = ', '.join(map(str, types_FN))
        logger.info(" TP: "+TP_str+" TN: "+TN_str+" FP: "+FP_str+" FN: "+FN_str)

        Test_result='TPR: ' + str(round(TPR, 3)) + ' TNR: ' + \
            str(round(TNR, 3)) + ' Score: ' + str(round(score, 3)) + ' Acc: ' + str(
    round(acc, 3)) + ' F1: ' + str(round(F1, 3)) + ' PPV: ' + str(round(PPV, 3)) + ' AUC: ' + str(round(roc_auc, 5))
        logger.info(str(calculate_metrics(types_TP, types_FP, types_TN, types_FN,y_test_final_all, y_pred_final_all,alarm_t)))


logger.info('\n')
logger.info(Test_result)


