import argparse
#from main_bs import *
import torch

import os
import argparse
import torch
from tqdm import tqdm
import time
ptlm_path = '/raid/data_stubowenx/supports/'


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def train_bs(cuda_device, ratio = 0.3):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    #max_mem = int(total * 0.1)
    block_mem = int(total * ratio)
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    del x

parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--cuda_id', default="0", type=str)
parser.add_argument('--base_path', default="./dataset", type=str)
parser.add_argument('--dataset', default='WebNLG', type=str)
parser.add_argument('--train', default="train", type=str)
parser.add_argument('--bert_learning_rate', default=3e-5, type=float)
parser.add_argument('--other_learning_rate', default=(3e-5)*5, type=float)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--lr_schedual', default="linear_decay", type=str) #['constant', 'cosine_decay', 'linear_decay']
parser.add_argument('--schedual_lr_rate', default=0.1, type=float)
parser.add_argument('--temperature', default=0.01, type=float)
parser.add_argument('--num_train_epochs', default=20000, type=int)
parser.add_argument('--cl_alpha', default=0.01, type=float)
parser.add_argument('--mrd_alpha', default=0.1, type=float)
parser.add_argument('--num_neg_samples', default=32, type=int)
parser.add_argument('--file_id', default='999', type=str)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--seed', default=2022, type=int)

parser.add_argument('--max_len', default=100, type=int)
parser.add_argument('--warmup', default=0.0, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--max_grad_norm', default=1.0, type=float)
parser.add_argument('--min_num', default=1e-7, type=float)
parser.add_argument('--bert_vocab_path', default= ptlm_path + "/bert-base-cased/vocab.txt", type=str)
parser.add_argument('--bert_config_path', default=ptlm_path + "/bert-base-cased/config.json", type=str)
parser.add_argument('--bert_model_path', default= ptlm_path + "/bert-base-cased/pytorch_model.bin", type=str)



parser.add_argument('--bert', default="4", type=str)
parser.add_argument('--bs', default="0.3", type=float)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.bert
train_bs(args.bert, args.bs)
for _ in range(1000):
    time.sleep(3600*24)
#print('Done')
setup_seed(args.seed)
if args.train=="train":
    train(args)
else:
    test(args)
