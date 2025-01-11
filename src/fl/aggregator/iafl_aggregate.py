import numpy as np
from math import ceil
import random
import torch
import copy

def compute_max_gamma_prime(gamma_list, N):
    # 计算 γ'_{i,t}
    gamma_primes = [gamma - (gamma - 1) / N for gamma in gamma_list]
    # 找到最大值
    max_gamma_prime = max(gamma_primes)
    return max_gamma_prime

def model_op(model1,model2,op=1):
    temp = {}
    for k in model1.state_dict().keys():
        temp[k] = torch.zeros_like(model1.state_dict()[k])

    for k in model1.state_dict().keys():
        temp[k] = model1.state_dict()[k]+model2.state_dict()[k]*op
    ret_model = copy.deepcopy(model1)
    ret_model.load_state_dict(temp)
    return ret_model

class ServerAgg:
    def __init__(self,ref_model):
        self.ref_model = ref_model



    def server_agg(self, gradients, client_pools, kappa=0.5):
        # TODO: 实现server aggregate 算法
        '''
        :param gradients: 长度为N，每一个Client的梯度
        :return: 长度为N的\Delta \theta_i^t，还有一个\theta_{ref}^t
        '''
        # Step 1: 获取每个客户端的数据量
        data_volume_list = []
        for client in client_pools:
            data_volume = client.get_data_volume()
            data_volume_list.append(data_volume)
        max_data_volume = max(data_volume_list) + 250

        # Step 2: 计算每个客户端的比率 gamma_i
        gamma_list = [(data_volume / max_data_volume) ** (1 - kappa) for data_volume in data_volume_list]

        # Step 3: 初始化最终聚合的梯度
        total_clients = len(client_pools)
        final_aggregated_gradient_list = []

        # Step 4: 按照 γ_i 比率聚合
        for i, gamma_i in enumerate(gamma_list):
            # 确定需要的梯度数量
            num_gradients = ceil(gamma_i * (total_clients - 1)) + 1  # 包含自己的梯度

            # 获取其他客户端的梯度（排除自己）
            other_gradients = [grad for j, grad in enumerate(gradients) if j != i]

            # 随机选取所需数量的梯度
            selected_gradients = other_gradients[:num_gradients - 1]
            selected_gradients.append(gradients[i])  # 包含自己的梯度

            gradients_weight = {}
            final_aggregated_gradient = gradients[0]
            for k in final_aggregated_gradient.state_dict().keys():
                gradients_weight[k] = torch.zeros_like(final_aggregated_gradient.state_dict()[k])
            for k in gradients_weight.keys():
                for model in selected_gradients:
                    gradients_weight[k] += model.state_dict()[k]
                gradients_weight[k] = torch.div(gradients_weight[k], len(selected_gradients))
            final_aggregated_gradient.load_state_dict(gradients_weight)
            final_aggregated_gradient_list.append(final_aggregated_gradient)

        max_gamma_prime = compute_max_gamma_prime(gamma_list, 30)
        # 计算选择的元素数量
        num_samples = ceil(max_gamma_prime * 30)
        # 随机选择 gamma*N 个元素
        ref_gradients = random.sample(gradients, num_samples)

        gradients_weight = {}
        ref_aggregated_gradient = gradients[0]
        for k in ref_aggregated_gradient.state_dict().keys():
            gradients_weight[k] = torch.zeros_like(ref_aggregated_gradient.state_dict()[k])
        for k in gradients_weight.keys():
            for model in ref_gradients:
                gradients_weight[k] += model.state_dict()[k]
            gradients_weight[k] = torch.div(gradients_weight[k], len(ref_gradients))
        ref_aggregated_gradient.load_state_dict(gradients_weight)
        ref_model = model_op(self.ref_model, ref_aggregated_gradient, op=1)
        self.ref_model = ref_model
        return final_aggregated_gradient_list, ref_model

