conf = {
"pixel": False,

#LEARNING
"burnin":1000,  # min. experiences before training
"learn_every":1,  # no. of experiences between updates to Q_online
"sync_every": 10, # no. of episode between Q_target & Q_online sync
"lr":0.00025,

#act
"exp":0,
"exploration_rate_decay": .999,
"exploration_rate_min": .001,

# ENV
"env_name":'CartPole-v1',
"episodes":1000,
"log_every_ep":25,
"image_memory_len":4,

#REPLAY MEMORY
"memory" : 2000,
"batch_size": 64,
"gamma": 0.95
}




