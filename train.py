from __future__ import division
from __future__ import print_function

SKIP_TRAINING = False

BASE_DIR = '.'
DATA_DIR = f'{BASE_DIR}/old_processed_data'

MODEL_CHECKPOINT_PATH='checkpoint_model.pt'
# device = 'cpu'
device = 'mps'

from util import count_files

import os
import copy
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from model import FAST
import torch.optim as optim
from evaluator import evaluate
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

seed = 123456789
np.random.seed(seed)
writer = SummaryWriter('logs/')

parser = argparse.ArgumentParser()

# TODO: handle cuda arg
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')

parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=123456789, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-2, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.20, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda=(device == 'cuda')


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_text_path = f"{DATA_DIR}/train_text/"
train_time_path = f"{DATA_DIR}/train_timestamps/"
train_price_path = f"{DATA_DIR}/train_price/"
train_mask_path = f"{DATA_DIR}/train_mask/"
train_gt_path = f"{DATA_DIR}/train_gt/"
no_of_train_samples = count_files(train_text_path)

val_text_path = f"{DATA_DIR}/val_text/"
val_time_path = f"{DATA_DIR}/val_timestamps/"
val_price_path = f"{DATA_DIR}/val_price/"
val_mask_path = f"{DATA_DIR}/val_mask/"
val_gt_path = f"{DATA_DIR}/val_gt/"
no_of_validation_samples = count_files(val_text_path)

test_text_path = f"{DATA_DIR}/test_text/"
test_time_path = f"{DATA_DIR}/test_timestamps/"
test_price_path = f"{DATA_DIR}/test_price/"
test_mask_path = f"{DATA_DIR}/test_mask/"
test_gt_path = f"{DATA_DIR}/test_gt/"
no_of_test_samples = count_files(test_text_path)

no_stocks = 2
# no_stocks = count_files(f'{DATA_DIR}/raw/tweet')

# TODO: change this
no_of_train_samples=1
no_of_validation_samples=1
no_of_test_samples=1


def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

def loss_rank(pred, base_price, ground_truth, mask, alpha, no_stocks):
    return_ratio = torch.div((pred - base_price), base_price)
    reg_loss = weighted_mse_loss(return_ratio, ground_truth, mask)
    all_ones = torch.ones(no_stocks,1).to(device)
    pre_pw_dif =  (torch.matmul(return_ratio, torch.transpose(all_ones, 0, 1)) - torch.matmul(all_ones, torch.transpose(return_ratio,0,1)))
    gt_pw_dif = (torch.matmul(all_ones, torch.transpose(ground_truth,0,1)) - torch.matmul(ground_truth, torch.transpose(all_ones,0,1)))
    
    print("the shape of ground truth is ", ground_truth.shape)

    mask_pw_previous_code = torch.matmul(mask, torch.transpose(mask,0,1))
    mask_pw = mask @ mask.t()
    assert (mask_pw == mask_pw_previous_code).all()


    rank_loss = torch.mean(F.relu(((pre_pw_dif*gt_pw_dif)*mask_pw)))
    loss = reg_loss + alpha*rank_loss
    del mask_pw, gt_pw_dif, pre_pw_dif, all_ones
    return loss, reg_loss, rank_loss, return_ratio

def train(
        epoch,
        save_step=max(no_of_train_samples//10, 1),
        ):

    tra_loss = 0.0
    tra_reg_loss = 0.0
    tra_rank_loss = 0.0
    model.train()
    optimizer.zero_grad()


    for i in range(no_of_train_samples):
        train_text = torch.tensor(np.load(train_text_path+str(i).zfill(10)+'.npy'), dtype=torch.float32).to(device)
        train_timestamps = torch.tensor(np.load(train_time_path+str(i).zfill(10)+'.npy'), dtype=torch.float32).to(device)
        output = model(train_text, train_timestamps)
        mask_batch = (np.load(train_mask_path+str(i).zfill(10)+'.npy'))#.to(device)
        price_batch = (np.load(train_price_path+str(i).zfill(10)+'.npy'))#.to(device)
        gt_batch = (np.load(train_gt_path+str(i).zfill(10)+'.npy'))#.to(device)
        cur_loss, cur_reg_loss, cur_rank_loss, curr_rr_train = loss_rank(output, torch.FloatTensor(price_batch).to(device), 
                                                                                    torch.FloatTensor(gt_batch).to(device), 
                                                                                    torch.FloatTensor(mask_batch).to(device), 
                                                                                    float(0.2), int(no_stocks))
        cur_loss.backward()
        # print('[INFO] Training: loss: ', cur_loss)
        optimizer.step()
        tra_loss += cur_loss.detach().cpu().item()
        tra_reg_loss += cur_reg_loss.detach().cpu().item()
        tra_rank_loss += cur_rank_loss.detach().cpu().item()
        # print('[INFO] METRICS -- Training Loss:',
                # tra_loss / (no_of_train_samples),
                # tra_reg_loss / (no_of_train_samples),
                # tra_rank_loss / (no_of_train_samples))
        
        del price_batch
        del gt_batch
        del mask_batch 

        if (i+1) % save_step == 0:
            torch.save(model, MODEL_CHECKPOINT_PATH)


def test_dict():
    with torch.no_grad():
        cur_valid_pred = np.zeros(
            [no_stocks, no_of_validation_samples],
            dtype=float)
        cur_valid_gt = np.zeros(
            [no_stocks, no_of_validation_samples],
            dtype=float)
        cur_valid_mask = np.zeros(
            [no_stocks, no_of_validation_samples],
            dtype=float)
        val_loss = 0.0
        val_reg_loss = 0.0
        val_rank_loss = 0.0
        
        model.eval()
        for i in range(no_of_validation_samples):
            val_text = torch.tensor(np.load(val_text_path+str(i).zfill(10)+'.npy'), dtype=torch.float32).to(device)
            val_timestamps = torch.tensor(np.load(val_time_path+str(i).zfill(10)+'.npy'), dtype=torch.float32).to(device)
            
            output_val = model(val_text, val_timestamps)
            
            mask_batch = (np.load(val_mask_path+str(i).zfill(10)+'.npy'))#.to(device)
            price_batch = (np.load(val_price_path+str(i).zfill(10)+'.npy'))#.to(device)
            gt_batch = (np.load(val_gt_path+str(i).zfill(10)+'.npy'))#.to(device)

            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = loss_rank(output_val, torch.FloatTensor(price_batch).to(device), 
                                                                                        torch.FloatTensor(gt_batch).to(device), 
                                                                                        torch.FloatTensor(mask_batch).to(device), 
                                                                                        float(0.2), int(1.0))
            cur_rr = cur_rr.detach().cpu().numpy().reshape((no_stocks,1))
            val_loss += cur_loss.detach().cpu().item()
            val_reg_loss += cur_reg_loss.detach().cpu().item()
            val_rank_loss += cur_rank_loss.detach().cpu().item()

            cur_valid_pred[:, i] = \
                copy.copy(cur_rr[:, 0])
            cur_valid_gt[:, i] = \
                copy.copy(gt_batch[:, 0])
            cur_valid_mask[:, i] = \
                copy.copy(mask_batch[:, 0])

            del price_batch
            del gt_batch
            del mask_batch
            
        # print('[INFO] METRICS -- Validation MSE:',
        #     val_loss / (no_of_validation_samples),
        #     val_reg_loss / (no_of_validation_samples),
        #     val_rank_loss / (no_of_validation_samples))
        # cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
        # print('\t [INFO] METRICS -- Validation preformance:', cur_valid_perf)
        

        cur_test_pred = np.zeros(
            [no_stocks, no_of_test_samples],
            dtype=float)
        cur_test_gt = np.zeros(
            [no_stocks, no_of_test_samples],
            dtype=float)
        cur_test_mask = np.zeros(
            [no_stocks, no_of_test_samples],
            dtype=float
        )
        test_loss = 0.0
        test_reg_loss = 0.0
        test_rank_loss = 0.0
        
        model.eval()
        for i in range(no_of_test_samples):
            test_text = torch.tensor(np.load(test_text_path+str(i).zfill(10)+'.npy'), dtype=torch.float32).to(device)
            test_timestamps = torch.tensor(np.load(test_time_path+str(i).zfill(10)+'.npy'), dtype=torch.float32).to(device)
            output_test = model(test_text, test_timestamps)
            mask_batch = (np.load(test_mask_path+str(i).zfill(10)+'.npy'))#.to(device)
            price_batch = (np.load(test_price_path+str(i).zfill(10)+'.npy'))#.to(device)
            gt_batch = (np.load(test_gt_path+str(i).zfill(10)+'.npy'))#.to(device)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = loss_rank(output_test, torch.FloatTensor(price_batch).to(device), 
                                                                                        torch.FloatTensor(gt_batch).to(device), 
                                                                                        torch.FloatTensor(mask_batch).to(device), 
                                                                                        float(0.2), int(1.0))
            cur_rr = cur_rr.detach().cpu().numpy().reshape((no_stocks,1))
            test_loss += cur_loss.detach().cpu().item()
            test_reg_loss += cur_reg_loss.detach().cpu().item()
            test_rank_loss += cur_rank_loss.detach().cpu().item()

            cur_test_pred[:, i] = \
                copy.copy(cur_rr[:, 0])
            cur_test_gt[:, i] = \
                copy.copy(gt_batch[:, 0])
            cur_test_mask[:, i] = \
                copy.copy(mask_batch[:, 0])

            del price_batch
            del gt_batch
            del mask_batch


        # print('[INFO] METRICS -- Test:',
        #     test_loss / (no_of_test_samples),
        #     test_reg_loss / (no_of_test_samples),
        #     test_rank_loss / (no_of_test_samples))
        cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
        print('\t[INFO] METRICS -- Test performance:', cur_test_perf)
        
        # df = pd.DataFrame(columns=['mrr','irr','sr','ndcg'])
        # df = df.append(cur_test_perf,ignore_index=True)
        df = pd.DataFrame([cur_test_perf], columns=['mrr', 'irr', 'sr', 'ndcg'])

        df.to_csv(f'{BASE_DIR}/fast_test_results.csv')
        

model = FAST(no_stocks).to(device)

lr_list = [1e-3, 5e-4, 3e-5]
for lr in lr_list:
    if args.cuda:
        model.cuda()
    else:
        model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    if SKIP_TRAINING:
        model = torch.load(MODEL_CHECKPOINT_PATH)
    else:
        for epoch in tqdm(range(args.epochs)):
            train(epoch)

    results = test_dict()