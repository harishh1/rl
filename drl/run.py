from packages import *
from env import Drl
from neural_nets import Drl

cartpole = Drl('CartPole-v0')
print(cartpole.state_dims)
print(cartpole.action_dims)
print(cartpole)



