from main import *
from logger import MetricLogger
from conf import *

use_cuda = torch.cuda.is_available()
print(f'Using cuda: {use_cuda} \n')


env_name = conf['env_name']
episodes = conf['episodes']
log_every_ep = conf['log_every_ep']
seeds = conf["seeds"]
if len(seeds) == 0: seeds = [np.random.randint(1,100)]

save_dir = Path("results") / env_name /datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

chk_dir = save_dir / 'chk'
chk_dir.mkdir()

arr_dir = save_dir / 'res_arrays'
arr_dir.mkdir()


for seed in seeds:
    
    conf["seed"] = seed
    conf["save_dir"] = save_dir
    env = Drl(env_name)
    logger = MetricLogger(save_dir, seed)

    for e in range(episodes):
        
        state = env.reset()
        # Play the game!
        while True:

            # Run agent on the state

            action = env.act(state)

            # Agent performs action
            next_state, reward, done, info = env.step(action)

            # Remember
            env.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = env.learn()

            #Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state
            
            # Check if end of game
            if done:
            #if done or info["flag_get"]:
                break
        logger.log_episode()

        if e % log_every_ep == 0:
            logger.record(episode = e, epsilon = env.exploration_rate, step = env.curr_step)
    logger.save()