conf = {
"pixel": False,

#LEARNING
"burnin":1000,  # min. experiences before training
"learn_every":1,  # no. of experiences between updates to Q_online
"sync_every": 10, # no. of episode between Q_target & Q_online sync
"lr":0.0005,
"exploration_rate_decay": .001,
"exploration_rate_min": .3,

# ENV
"env_name":'CartPole-v1',
"episodes":3000,
"log_every_ep":10,
"image_memory_len":4,

#REPLAY MEMORY
"memory" : 5000,
"batch_size": 64,
"gamma": 0.95
}




