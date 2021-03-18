from_pixel = False
environment_settings ={
        'env_name' : 'CartPole-v1',
        'gamma': 1.00,
        'max_minutes': 20,
        'max_episodes': 5,
        'goal_mean_100_reward': 475
        }

SEEDS = (12,)
model_hidden_layers = [512,128]
init_epsilon = 1.0
min_epsilon=0.3
decay_steps=20000
max_size=50000
if from_pixel:
    batch_size= 1
else:
    batch_size= 64
n_warmup_batches = 5
update_target_every_steps = 10
value_optimizer_lr = 0.0005

