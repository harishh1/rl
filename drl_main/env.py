import collections
from packages import *
from conf import *

'''
Creates Env
preprocessing the observation or image
step, reset functions of env
'''

class Env():
    def __init__(self, env_name):
        print('this is env')
        self.env_name = env_name
        self.image_memory = collections.deque(maxlen= conf['image_memory_len'])
        self.env = gym.make(env_name)
        self.reset()
    def step(self,action):
        next_state, reward, done, _ = self.env.step(action)
        self.image_memory.appendleft(self.get_screen())
        next_state = np.array(self.image_memory) 
        return next_state, reward, done, _

    def reset(self):
        self.env.reset()
        self.org_shape = self.env.render(mode='rgb_array').shape #used for the transformation of image
        for i in range(conf['image_memory_len']):
            self.image_memory.append(self.get_screen())

        s = np.array(self.image_memory)
        self.state_dim = s.shape
        self.action_dim = self.env.action_space.n
        return s

    def gray_scale(self,observation):
        #[H, W, C] to [C, H, W]
        observation = np.transpose(observation, (2,0,1))
        observation = torch.tensor(observation.copy(), dtype = torch.float)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

    def resize(self, observation):
        transform = T.Compose(
            [T.Resize(self.org_shape[:2]), T.Normalize(0,255)]
        )
        observation = transform(observation)
        return observation

    def get_screen(self):
        return self.resize(
            self.gray_scale(
                self.env.render(mode='rgb_array')
                )
            ).squeeze().numpy()