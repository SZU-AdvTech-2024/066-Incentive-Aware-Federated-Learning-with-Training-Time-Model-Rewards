import math
import random

import numpy as np

from .base import Client

class AuctionClient(Client):
    def __init__(self, id, dataset, idxes, trainer, mu=None, noisy_rate=0.0,test_size = 0.0):
        super().__init__(id, dataset, idxes, trainer, mu, noisy_rate,test_size)
        pass

    def send_bid(self):
        noisy_rate = self.dataset.noisy_rate
        if noisy_rate >= 0.5:
            base_value = noisy_rate + 5
        elif noisy_rate >= 0.2:
            base_value = noisy_rate + 6
        else:
            base_value = noisy_rate + 6.8

        lower_bound = base_value * 0.8
        upper_bound = base_value * 1.2
        return round(random.uniform(lower_bound, upper_bound), 2)
    def send_iafl_bid(self):
        return round(random.uniform(5, 8), 2)

    def get_model(self):
        return self.model


