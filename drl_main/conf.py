conf = {
"pixel": False,
"seeds":[23, 65, 44, 89, 98],

#LEARNING
"burnin":1000,  # min. experiences before training
"learn_every":1,  # no. of experiences between updates to Q_online
"sync_every": 10, # no. of episode between Q_target & Q_online sync
"lr":0.00025,

#act
"exp":1, # || 0 --> exponential decay | 1 --> episilon greedy || 
"exploration_rate_decay": .0005,
"exploration_rate_min": .001,

# ENV
"env_name":'CartPole-v1',
"episodes":50,
"log_every_ep":10,
"image_memory_len":4,

#REPLAY MEMORY
"memory" : 4000,
"batch_size": 64,
"gamma": 0.95,

"save_every":20
}
'''
save every change 100
ep
'''

