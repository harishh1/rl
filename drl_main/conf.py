conf = {

#LEARNING
"burnin":500,  # min. experiences before training
"learn_every":1,  # no. of experiences between updates to Q_online
"sync_every": 50, # no. of experiences between Q_target & Q_online sync
"lr":0.00025,

"exploration_rate_decay": .9999,
"exploration_rate_min": .01,
# ENV
"env_name":'CartPole-v1',
"episodes":1000,
"log_every_ep":20,
"image_memory_len":4,

#REPLAY MEMORY
"memory" : 3000,
"batch_size": 32,
"gamma": 0.95
}




