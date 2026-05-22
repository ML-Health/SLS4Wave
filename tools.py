import logging
import torch
from torch.nn.functional import threshold
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
import numpy as np
from utils.filter import filter
from utils.nt_xent import NTXentLoss
import torch.nn as nn

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger


class Dataset_train(Dataset):
    # 'Characterizes a dataset for PyTorch'
    def __init__(self, signal_train, alarm_train, y_train):
        # 'Initialization'

        self.strain = signal_train  # signal
        self.atrain = alarm_train  # alarm
        self.ytrain = y_train  # label (true/false)

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.ytrain)

    def __getitem__(self, index):
        # 'Generates one sample of data'
        # Select sample
        return self.strain[index], self.atrain[index], self.ytrain[index]

def new_pretrain_model_multi_node(batch, model, loss_ce, device, length, weight):  
    # used for multi-node training
    signal_train, _, y_train = batch

    length = length
    loss_obj = NTXentLoss(device, signal_train.shape[0], temperature=0.1, use_cosine_similarity=True)

    # Ensure loss is computed on GPU
    loss_ce = torch.nn.CosineEmbeddingLoss().to(device)

    # Move data to GPU
    signal_train_1 = signal_train[:, :, 33750 - length: 33750].to(device)  # normal heart beat time interval
    signal_train_2 = signal_train[:, :, 37500 - length: 37500].to(device)  # 10s event
    signal_train_4 = filter(signal_train[:, :, 37500 - length: 37500])  # abnormal heart beat time interval after augmentation
    signal_train_4 = signal_train_4[:, :, :].to(device)
    y_train = y_train.to(device)

    # Ensure model is compatible with DDP
    if isinstance(model, nn.parallel.DistributedDataParallel):
        feature_1, Y_train_prediction = model.module(signal_train_1)
        feature_2, Y_train_prediction = model.module(signal_train_2)
        feature_4, Y_train_prediction = model.module(signal_train_4)
    else:
        feature_1, Y_train_prediction = model(signal_train_1)
        feature_2, Y_train_prediction = model(signal_train_2)
        feature_4, Y_train_prediction = model(signal_train_4)

    # Compute CLOC-like loss
    loss_1 = loss_obj(feature_1, feature_2)
    loss_2 = loss_obj(feature_2, feature_4)
    loss = loss_1 + loss_2 * 10  # only use CLOC-like loss

    return loss, Y_train_prediction, y_train

def new_pretrain_model(batch, model, loss_ce, device,
                weight):  # signal_train, alarm_train, y_train, signal_test, alarm_test, y_test = batch
    
    # t1,t2=0.5,0.5
    signal_train,  _,y_train = batch
    # print(signal_train.shape)
    # signal_train = signal_train[:,:,::2]
    length = 2500
    loss_obj = NTXentLoss(device, signal_train.shape[0], temperature=0.1, use_cosine_similarity=True)
    # loss_triplet = torch.nn.TripletMarginLoss(margin=2)
    loss_ce = torch.nn.CosineEmbeddingLoss()
    signal_train_1 = signal_train[:, :, 33750:35000].to(device)                     # normal heart beat time interval
    signal_train_2 = signal_train[:, :, 36250:37500].to(device)    
    signal_train_3 = signal_train[:, :, 32500:33750].to(device)
    # alarm_train = alarm_train.to(device)

    # signal_train_3 = find_interval(signal_train_1,signal_train_2).to(device)        # abnormal heart beat time interval

    signal_train_4 = filter(signal_train[:, :, 36250:37500])                            # abnormal heart beat time interval after augmentation
    signal_train_4 = signal_train_4[:,:,:].to(device)
    y_train = y_train.to(device)

    feature_1,Y_train_prediction = model(signal_train_1)
    feature_2,Y_train_prediction = model(signal_train_2)
    feature_3,Y_train_prediction = model(signal_train_3)
    feature_4,Y_train_prediction = model(signal_train_4)

    # tar = -torch.ones(feature_1.shape[0]).to(device)
    # loss_1 = loss_ce(feature_1, feature_2,tar)                          # maximize the distance between the normal and abnormal. 
    # epsilon = 1e-8
    # loss_1 = -torch.sum(torch.log(1/torch.exp(cosine_sim + epsilon)))/feature_1.shape[0]

    # use CLOC-like loss
    loss_1 = loss_obj(feature_1, feature_2)                                         # minimize the distance between normal heart beat and augmented heart beat
    loss_2 = loss_obj(feature_2, feature_4)                                         # minimize the distance between normal heart beat and augmented heart beat
    # loss_2  = loss_obj(feature_2, feature_3) 
    loss = loss_1   + loss_2 *10                                   #only use CLOC-like loss

    # use triplet loss
    # loss_1 = loss_triplet(feature_1, feature_3, feature_2)
    # loss_2 = loss_obj(feature_2,feature_4)
    # loss = loss_1 + loss_2
    # loss = loss_2                                                   # only use noise loss
    
    return loss, Y_train_prediction, y_train

def train_model(batch, model, loss_ce, device, length,
                weight): 
    # signal_train, alarm_train, y_train, signal_test, alarm_test, y_test = batch
    signal_train, alarm_train, y_train = batch

    length = length
    signal_train = signal_train.to(device)
    signal_train = signal_train[:,:,::2].to(device)  # downsample to 125Hz
    
    signal_train = signal_train[:, :, -length-7500:-7500]
    alarm_train = alarm_train.to(device)
    y_train = y_train.to(device)

    _,Y_train_prediction = model(signal_train)

    loss = loss_ce(Y_train_prediction, y_train)

    return loss, Y_train_prediction, y_train


def eval_model(batch, model, loss_ce, length,
               device):  
    # signal_train, alarm_train, y_train, signal_test, alarm_test, y_test = batch
    signal_train, alarm_train, y_train = batch
    length = length
    signal_train = signal_train.to(device)
    signal_train = signal_train[:,:,::2].to(device)  # downsample to 125Hz
    
    signal_train = signal_train[:, :,  -length-7500:-7500]
    alarm_train = alarm_train.to(device)
    y_train = y_train.to(device)

    _,Y_train_prediction = model(signal_train)

    loss = loss_ce(Y_train_prediction, y_train)

    return loss, Y_train_prediction, y_train, alarm_train


def evaluation(Y_eval_prediction, y_test, TP, FP, TN, FN):  # b 2   set 0 is false alarm and 1 is true alarm
    pre = (Y_eval_prediction >= 0).int()
    for i, j in zip(pre, y_test):
        if i.item() == 1 and j.item() == 1:  # 1 -> 1
            TP += 1
        if i.item() == 1 and j.item() == 0:  # 0 -> 1
            FP += 1
        if i.item() == 0 and j.item() == 0:  # 0 -> 0  # false classified to false
            TN += 1
        if i.item() == 0 and j.item() == 1:  # 1 -> 0  # true alarm classified to false alarm
            FN += 1
    return TP, FP, TN, FN


def evaluate_rule_based(rule_based_results, y_test):
    TP = FP = TN = FN = 0
    for i, j in zip(rule_based_results, y_test):
        if i.item() == 1 and j.item() == 1:  # 1 -> 1
            TP += 1
        if i.item() == 1 and j.item() == 0:  # 0 -> 1
            FP += 1
        if i.item() == 0 and j.item() == 0:  # 0 -> 0  # false classified to false
            TN += 1
        if i.item() == 0 and j.item() == 1:  # 1 -> 0  # true alarm classified to false alarm
            FN += 1
    return 100 * TP / (TP + FN), 100 * TN / (TN + FP), 100 * (TP + TN) / (TP + TN + FP + 5 * FN), 100 * (TP + TN) / (
            TP + TN + FP + FN)


def evaluation_test(alarm_types, Y_eval_prediction, y_test, types_TP, types_FP, types_TN,
                    types_FN):  # b 2   set 0 is false alarm and 1 is true alarm
    pre = (Y_eval_prediction >= 0).int()
    for i, j, k in zip(pre, y_test, alarm_types):
        idx = torch.argmax(k).item()
        if i.item() == 1 and j.item() == 1:  # 1 -> 1
            types_TP[idx] += 1
        if i.item() == 1 and j.item() == 0:  # 0 -> 1
            types_FP[idx] += 1
        if i.item() == 0 and j.item() == 0:  # 0 -> 0  # false classified to false
            types_TN[idx] += 1
        if i.item() == 0 and j.item() == 1:  # 1 -> 0  # true alarm classified to false alarm
            types_FN[idx] += 1
    return types_TP, types_FP, types_TN, types_FN,pre


def evaluate_raise_threshold(alarm_type, prediction, groundtruth, types_TP, types_FP, types_TN, types_FN, threshold):
    prediction = torch.sigmoid(prediction)

    alarm_type = alarm_type.argmax(1).item()
    if alarm_type == 2:
        pre = 1 if prediction >= threshold else 0
    else:
        pre = 1 if prediction >= 0.5 else 0

    if pre == 1 and groundtruth == 1:
        types_TP[alarm_type] += 1
    elif pre == 1 and groundtruth == 0:
        types_FP[alarm_type] += 1
    elif pre == 0 and groundtruth == 1:
        types_FN[alarm_type] += 1
    elif pre == 0 and groundtruth == 0:
        types_TN[alarm_type] += 1

    return types_TP, types_FP, types_TN, types_FN
def calculate_metrics(TP, FP, TN, FN,y_tests, Y_eval_predictions,alarms):
    TP = np.array(TP)
    FP = np.array(FP)
    TN = np.array(TN)
    FN = np.array(FN)
    ills=['Ventricular_Tachycardia', 'Ventricular_Flutter_Fib','Asystole','Tachycardia','Bradycardia']
    TPR = 100*(TP / (TP + FN + 1e-6))
    TNR = 100*(TN / (TN + FP + 1e-6))
    PPV = 100*(TP / (TP + FP + 1e-6))
    ACC = 100*((TP + TN) / (TP + TN + FP + FN + 1e-6))
    score = 100*((TP + TN) / (TP + TN + FP + FN*5 + 1e-6))
    precision = 100*(TP / (TP + FP + 1e-6))
    recall = 100*(TP / (TP + FN + 1e-6))
    F1 = 2 * precision * recall / (precision + recall + 1e-6)
    aucs = calculate_aucs(y_tests, Y_eval_predictions,alarms)
    results=   {}
    results[ills[0]] = 'TPR: ' + str(round(TPR[0], 3)) + ' TNR: ' +  str(round(TNR[0], 3)) + ' Score: ' + str(round(score[0], 3)) + ' Acc: ' + str(round(ACC[0], 3))+' F1 : '+str(round(F1[0], 3))+ ' PPV :'+str(round(PPV[0], 3))+ ' AUC :'+str(round(aucs[0], 3))
    results[ills[1]] = 'TPR: ' + str(round(TPR[1], 3)) + ' TNR: ' +  str(round(TNR[1], 3)) + ' Score: ' + str(round(score[1], 3)) + ' Acc: ' + str(round(ACC[1], 3))+' F1 : '+str(round(F1[1], 3))+ ' PPV :'+str(round(PPV[1], 3))+ ' AUC :'+str(round(aucs[1], 3))
    results[ills[2]] = 'TPR: ' + str(round(TPR[2], 3)) + ' TNR: ' +  str(round(TNR[2], 3)) + ' Score: ' + str(round(score[2], 3)) + ' Acc: ' + str(round(ACC[2], 3))+' F1 : '+str(round(F1[2], 3))+ ' PPV :'+str(round(PPV[2], 3))+ ' AUC :'+str(round(aucs[2], 3))
    results[ills[3]] = 'TPR: ' + str(round(TPR[3], 3)) + ' TNR: ' +  str(round(TNR[3], 3)) + ' Score: ' + str(round(score[3], 3)) + ' Acc: ' + str(round(ACC[3], 3))+' F1 : '+str(round(F1[3], 3))+ ' PPV :'+str(round(PPV[3], 3))+ ' AUC :'+str(round(aucs[3], 3))
    results[ills[4]] = 'TPR: ' + str(round(TPR[4], 3)) + ' TNR: ' +  str(round(TNR[4], 3)) + ' Score: ' + str(round(score[4], 3)) + ' Acc: ' + str(round(ACC[4], 3))+' F1 : '+str(round(F1[4], 3))+ ' PPV :'+str(round(PPV[4], 3))+ ' AUC :'+str(round(aucs[4], 3))
    return results

def calculate_aucs(y_tests, Y_eval_predictions,alarms):
    from sklearn.metrics import roc_curve, auc
    aucs=[]
    my_list = [[] for _ in range(5)]
    y_tests = torch.tensor(y_tests)
    Y_eval_predictions = torch.tensor(Y_eval_predictions)
    for i in range(5):
        for j in range(len(y_tests)):
            if alarms[j,i]==1.0:
                my_list[i].append(j)
    for i in range(5):
        y_test = y_tests[my_list[i]]
        Y_eval_prediction = Y_eval_predictions[my_list[i]]
        fpr, tpr, thresholds = roc_curve(y_test, Y_eval_prediction)
        auc_score = auc(fpr, tpr)
        aucs.append(auc_score*100)
    return aucs