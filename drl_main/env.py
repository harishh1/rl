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

    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


    def get_screen(self):
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))

        # Cart is in the lower half, so strip off the top and bottom of the screen
        _ , screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]

        return self.resize(
            self.gray_scale(
                screen
                )
            ).squeeze().numpy()