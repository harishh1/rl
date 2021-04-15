from packages import *
from preprocess import *

env = gym.make('CartPole-v1')
im = env.reset()
next_state, reward, done, info = env.step(0)
print(next_state.shape)

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

print(env.step(0))




