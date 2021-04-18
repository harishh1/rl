from main import *
from logger import MetricLogger

env_name = 'CartPole-v1' 
episodes = 40
log_every_ep = 20


use_cuda = torch.cuda.is_available()
print(f'Using cuda: {use_cuda} \n')

save_dir = Path("results") / env_name /datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
env = Drl(env_name)

logger = MetricLogger(save_dir)


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

    if e % 20 == 0:
        print(logger.ep_rewards)
        logger.record(episode = e, epsilon = env.exploration_rate, step = env.curr_step)