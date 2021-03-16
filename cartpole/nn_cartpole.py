import sys
sys.path.append('../Base/')

from imports import *
from sequential_models import FCQ
from training_startegy import EGreedyExpStrategy, GreedyStrategy
from env import get_make_env_fn
from Base import ReplayBuffer
from dqn import DQN

SEEDS = (12,)

environment_settings = {
        'env_name' : 'CartPole-v1',
        'gamma': 1.00,
        'max_minutes': 20,
        'max_episodes': 3,
        'goal_mean_100_reward': 475
    }

model_hidden_layers = [512,128]

#training strategy params
init_epsilon = 1.0
min_epsilon=0.3
decay_steps=20000

#replay buffer
max_size=50000
batch_size=64
n_warmup_batches = 5
update_target_every_steps = 10
value_optimizer_lr = 0.0005

dqn_results = []
log.info(environment_settings)
for seed in SEEDS:

    #Neural Net
    value_model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims = model_hidden_layers)

    #Optimizer
    value_optimizer_fn = lambda net, lr: optim.RMSprop(net.parameters(), lr=lr)

    #Training Strategy
    training_strategy_fn = lambda: EGreedyExpStrategy(init_epsilon=init_epsilon,  
                                                      min_epsilon=min_epsilon, 
                                                      decay_steps=decay_steps)
    #Testing Strategy
    evaluation_strategy_fn = lambda: GreedyStrategy()

    #Memory
    replay_buffer_fn = lambda: ReplayBuffer(max_size=max_size, batch_size=batch_size)
    
    #Environment
    env_name, gamma, max_minutes,     max_episodes, goal_mean_100_reward = environment_settings.values()

    #Agent with all above functions
    agent = DQN(replay_buffer_fn,
                value_model_fn,
                value_optimizer_fn,
                value_optimizer_lr,
                training_strategy_fn,
                evaluation_strategy_fn,
                n_warmup_batches,
                update_target_every_steps)

    make_env_fn, make_env_kargs = get_make_env_fn(env_name=env_name)

    #Taining the agent!!!
    result= agent.train(
        make_env_fn, make_env_kargs, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)

    dqn_results.append(result)







