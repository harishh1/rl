
from conf import *
import numpy as np
class Exploration:
    def __init__(self):
        self.rate = 1
        self.decay_step = 1
        self.rate_decay = conf['exploration_rate_decay']
        self.rate_min = conf['exploration_rate_min']

    def decay(self):
        if conf['exp'] == 0:
            self.exponential_decay()
        elif conf['exp'] == 1:
            self.episilon_greedy()
        
        self.decay_step += 1
        return self.rate
        
    def exponential_decay(self):
         self.rate *= self.rate_decay
         self.rate = max(self.rate, self.rate_min)
    
    def episilon_greedy(self):
         # decrease exploration_rate
         self.rate = self.rate_min + \
            (1 - self.rate_min) * \
                np.exp(-self.rate_decay * self.decay_step)

