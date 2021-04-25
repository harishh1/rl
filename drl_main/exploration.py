
from conf import *

class Exploration:
    def __init__(self):
        self.rate = 1
        self.rate_decay = conf['exploration_rate_decay']
        self.rate_min = conf['exploration_rate_min']

    def decay(self, ep, step):
        if conf['exp'] == 0:
            self.exponential_decay(ep,step)
        return self.rate
        
    def exponential_decay(self,ep,step):
         self.rate *= self.rate_decay
         self.rate = max(self.rate, self.rate_min)

        # decrease exploration_rate
        #self.exploration_rate = self.exploration_rate_min + \
        #    (1 - self.exploration_rate_min) * \
        #        np.exp(-self.exploration_rate_decay * self.ep)

        
