conf = {
"from_pixel": False,
"SEEDS": [12],
"model_hidden_layers": [512,128],
"init_epsilon": 1.0,
"min_epsilon":0.3,
"decay_steps":20000,
"max_size":50000,
"batch_size": 2,
"n_warmup_batches": 1,
"update_target_every_steps": 10,
"value_optimizer_lr": 0.0005,
"environment_settings":{
        'env_name' : 'CartPole-v0',
        'gamma': 1.00,
        'max_minutes': 10,
        'max_episodes': 4,
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