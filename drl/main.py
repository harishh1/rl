from env import Env
from packages import *

class Drl(Env):
    def __init__(self, env_name):
        super().__init__(env_name)
    def action(self):
        self.use_cuda = torch.cuda.is_available()
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')
            


cartpole = Drl('CartPole-v1')
i = cartpole.reset().numpy()
