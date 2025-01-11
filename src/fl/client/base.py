from .fldataset import FLDataset
import torch


# the fed prox element
class Prox:
    def __init__(self, global_model, mu):
        '''
        :param global_model:
        :param mu:
        '''
        self.mu = mu
        self.global_model = global_model

    def __call__(self, model):
        '''
        add the prox item when train in client
        :param model:
        :return:
        '''
        proximal_term = 0.0
        for w, w_t in zip(model.parameters(), self.global_model.parameters()):
            proximal_term += (w - w_t).norm(2)
        return (self.mu / 2) * proximal_term


# the client used for training
class Client:
    def __init__(self, id, dataset, idxes, trainer, mu=None, noisy_rate=0.0, test_size=0.0):
        self.id = id
        self.dataset = FLDataset(dataset, idxes, noisy_rate=noisy_rate)
        self.test_dataset = None
        # TODO: 根据需要将数据拆分为训练集和测试集
        if test_size != 0.0:
            train_size = int((1 - test_size) * len(self.dataset))
            test_size = len(self.dataset) - train_size
            self.dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])
        self.trainer = trainer

        self.model = None
        # save the model in the last round
        self.last_model = None
        self.mu = mu

    # receive the global model, can be used in first round or the round be selected
    def receive_model(self, model):
        '''
        receive the global model
        :param model:
        :return:
        '''
        self.last_model = self.model
        self.model = model

    # get the local model
    def train(self):
        '''
        local training
        :return:
        '''
        print('Client {} training'.format(self.id))
        # here will save the trained model in the local, in order to calculate the utility
        if self.mu is not None:
            # use the FedProx algo
            add_item = Prox(self.model, self.mu)
            self.model = self.trainer.update(self.dataset, self.model, add_item)
        else:
            self.model = self.trainer.update(self.dataset, self.model)
        return self.model

    def eval(self):
        if self.test_dataset is None:
            raise ValueError('No test dataset, please input the non-zero test_size')
        acc, loss = self.trainer.eval(self.test_dataset, self.model)
        return acc, loss

    def get_loss(self):
        '''
        count the loss, used for the utility computation
        :return:
        '''
        return self.trainer.get_loss(self.dataset, self.model)
    def get_data_volume(self):

        return len(self.dataset)