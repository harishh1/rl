conf = {
"from_pixel": True,
"SEEDS": [34],
"model_hidden_layers": [512,128],
"init_epsilon": 1.0,
"min_epsilon":0.1,
"decay_steps":2000,
"epsilon_decay_ratio": .0005,
"max_size":10000,
"batch_size": 32,
"n_warmup_batches": 1,
"update_target_every_steps": 10,
"value_optimizer_lr": 0.00025,
"environment_settings":{
        'env_name' : 'CartPole-v1',
        'gamma': .95,
        'max_minutes': 10,
        'max_episodes': 1000,
        'goal_mean_100_reward': 475
        }
}

from_pixel = conf['from_pixel']
environment_settings = conf['environment_settings']
SEEDS = conf['SEEDS']
model_hidden_layers = conf['model_hidden_layers']
init_epsilon = conf['init_epsilon']
min_epsilon = conf['min_epsilon']
decay_steps = conf['decay_steps']
max_size = conf['max_size']
batch_size = conf['batch_size']
n_warmup_batches = conf['n_warmup_batches']
update_target_every_steps = conf['update_target_every_steps']
value_optimizer_lr = conf['value_optimizer_lr']
epsilon_decay_ratio = conf['epsilon_decay_ratio']