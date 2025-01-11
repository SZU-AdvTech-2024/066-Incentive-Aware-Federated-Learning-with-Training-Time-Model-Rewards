import copy
from typing import List
import numpy as np
import torch

from src.fl.aggregator import average_weights
from src.fl.selector import BaseSelector
from src.fl.selector.randomauctionselector import RandomAuctionSelector
from src.utils import parse_yaml
import src.task as TK
import src.fl.sampler as FLS
from src.fl.sampler import LabelSkewSampler
import src.fl.selector as SLT
import src.fl.algo as FLAO
from src.logger_iafl import LoggerFL
from src.fl.client import Client
from src.fl.client import AuctionClient
from src.fl.aggregator import ServerAgg

import argparse
import random

# get the config file path
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='/scientific research/work/FL/config.yaml', help='the path of config file')
args = parser.parse_args()

# load the config file
config = parse_yaml(args.config)
task_config = config['Task']

# get the dataset
train_dataset, test_dataset = getattr(TK.baseloader, task_config['Dataset']['name'])(**task_config['Dataset']['args'])

# get the model
global_model = getattr(TK.models, task_config['Model']['name'])(**task_config['Model']['args'])
global_model.to(task_config['Trainer']['args']['device'])

# get the trainer
global_trainer = getattr(TK.trainer, task_config['Trainer']['name'])(**task_config['Trainer']['args'])

# get the sampler
sampler_config = config['Sampler']
# sampler = LabelSkewSampler(train_dataset, **sampler_config['args'])
sampler = getattr(FLS, sampler_config['name'])(train_dataset, **sampler_config['args'])

user_idxes = sampler.sample()

# get the selector
selector = getattr(SLT, config['Selector']['name'])(**config['Selector']['args'])
# create noisy level list
alpha = config['Noise']['alpha']
sigma = config['Noise']['sigma']
mu = config['Noise']['mu']
lower = config['Noise']['lower']
upper = config['Noise']['upper']

if alpha == 0.0:
    noisy_levels = [0 for _ in range(len(user_idxes))]
else:
    #noisy_levels = np.random.uniform(0, alpha, len(user_idxes))
    #noisy_levels = np.random.binomial(1, alpha, len(user_idxes))
    num_clients = len(user_idxes)  # 客户端数量
    noise_levels = np.random.normal(loc=mu, scale=sigma, size=num_clients)
    # 手动截断，将超出范围的值裁剪到 [lower, upper] 之间
    noisy_levels = np.clip(noise_levels, lower, upper)
    '''num_clients = len(user_idxes)  # 客户端数量
    # 将每个客户端的噪声程度设置为 0
    noisy_levels = np.zeros(num_clients)'''

print("noisy_levels", noisy_levels)

# crete the logger
logger = LoggerFL(config['save_path'])

#print('the noisy levels mean:', np.mean(noisy_levels))

#print('the noisy levels mean:', np.mean(np.random.uniform(0, 1, 1000)))

# create the client pool
client_pool:List[Client] = []


distribute_first = True

def model_op(model1,model2,op=1):
    temp = {}
    for k in model1.state_dict().keys():
        temp[k] = torch.zeros_like(model1.state_dict()[k])

    for k in model1.state_dict().keys():
        temp[k] = model1.state_dict()[k]+model2.state_dict()[k]*op
    ret_model = copy.deepcopy(model1)
    ret_model.load_state_dict(temp)
    return ret_model
def fl_train_Avg(global_model, global_trainer, client_pools, selector: BaseSelector, test_dataset, fl_config, logger=None):
    rounds = fl_config['rounds']
    acc_list = []
    loss_list = []
    for round in range(rounds):
        print(f'Round {round} start')
        # store the local models
        local_models = []
        acc_list.append([])
        loss_list.append([])

        # get the feedback, such as utility, from the clients
        selector.con_feedback()
        winner = selector.select()
        print(f'Winner: {winner}')
        for idx in winner:
            client = client_pools[idx]
            client.receive_model(copy.deepcopy(global_model))
            model = client.train()
            local_models.append(model)
            acc,loss = client.eval()
            acc_list[-1].append(acc)
            loss_list[-1].append(loss)
        # global model
        global_model = average_weights(global_model,local_models)

        print(f'Round {round} end')

        payment_list, bids = selector.get_data()
        if logger is not None:
            pass
            logger.update(loss_list[-1], acc_list[-1], bids, payment_list)
def fl_train(global_model, global_trainer, client_pools, selector: BaseSelector, test_dataset, fl_config, logger=None):

    agg = ServerAgg(copy.deepcopy(global_model))

    # TODO: input the probability q
    q = 0.1

    # TODO: initiate phase, 初始化每一个模型的参数，还有初始化的ref model
    gradients = []
    for i in range(len(client_pools)):
        zero_gradient = copy.deepcopy(global_model)
        zero_dict = {}
        for k in zero_gradient.state_dict().keys():
            zero_dict[k] = torch.zeros_like(zero_gradient.state_dict()[k])
        zero_gradient.load_state_dict(zero_dict)
        gradients.append(zero_gradient)

    rounds = fl_config['rounds']
    acc_list = []
    loss_list = []
    for round in range(rounds):
        print(f'Round {round} start')
        # store the local models
        local_models = []
        acc_list.append([])
        loss_list.append([])

        # TODO: 调用server agg
        delta_model_list, ref_model = agg.server_agg(gradients, client_pools)
        for i, client in enumerate(client_pools):
            coin = random.uniform(0,1)
            # 如果小于q，模型直接拿到ref model
            if coin <= q:
                model = ref_model
            # 如果大于q，也就是1-q的概率触发，那么用自己的上一轮模型，与模型变化量相加
            else:
                model = client.get_model()
                model = model_op(model, delta_model_list[i])
            client.receive_model(model)
            cp_model = copy.deepcopy(model)
            trained_model = client.train()
            acc,loss = client.eval()
            acc_list[-1].append(acc)
            loss_list[-1].append(loss)
            client.receive_model(cp_model)
            gradient = model_op(trained_model,cp_model,op=-1)
            gradients[i] = gradient
        # get the feedback, such as utility, from the clients
        selector.feedback()
        # select the clients
        winner = selector.select()
        print(f'Winner: {winner}')

        print(f'Round {round} end')

        payment_list, bids = selector.get_data()
        if logger is not None:
            pass
            logger.update(loss_list[-1], acc_list[-1], bids, payment_list)


# ------------------------------------------------Fed Avg------------------------------------------------
def federated_average(global_model, global_trainer, user_idxes, selector: BaseSelector, train_dataset, test_dataset,
                      fl_config, logger=None):
    '''
    :param global_model:
    :param global_trainer:
    :param user_idxes:
    :param selector:
    :param train_dataset:
    :param test_dataset:
    :param fl_config:
    :return:
    '''
    # create the client pools
    client_pools: List[AuctionClient] = []
    for i, (user_idx, noisy_level) in enumerate(zip(user_idxes, noisy_levels)):
        # TODO: 增加一个初始化ref model的环节，实现的时候就直接用glboal_model去初始化了，反正都是随机初始化的
        client = AuctionClient(i, train_dataset, user_idx, global_trainer, noisy_rate=noisy_level,test_size=0.3)
        # ensure every client has the model
        client.receive_model(copy.deepcopy(global_model))
        client_pools.append(client)
    # initiate the selector
    selector.initiate(client_pools)
    if config['algo']=='iafl':
        fl_train(global_model, global_trainer, client_pools, selector, test_dataset, fl_config, logger=logger)
    elif config['algo'] =='fed_avg':
        fl_train_Avg(global_model, global_trainer, client_pools, selector, test_dataset, fl_config, logger=logger)

federated_average(global_model, global_trainer, user_idxes, selector, train_dataset, test_dataset, config['FL'],logger=logger)
logger.close()

