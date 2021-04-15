from packages import *

'''
Creates Env
preprocessing the observation or image
step, reset functions of env
'''

class Env():
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)
    def step(self,action):
        next_state, reward, done, _ = self.env.step(action)
        next_state = self.get_screen()
        return next_state, reward, done, _

    def reset(self):
        self.env.reset()
        self.org_shape = self.env.render(mode='rgb_array').shape
        s = self.get_screen()
        self.state_dims = s.shape
        self.action_dims = self.env.action_space.n
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
            )