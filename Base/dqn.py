from numpy.core.fromnumeric import std
from imports import *
from Base import get_gif_html

class DQN():
    def __init__(self,
                 replay_buffer_fn,
                 value_model_fn,
                 value_optimizer_fn,
                 value_optimizer_lr,
                 training_strategy_fn,
                 evaluation_strategy_fn,
                 n_warmup_batches,
                 update_target_every_steps
                 ):
        self.replay_buffer_fn = replay_buffer_fn
        self.value_model_fn = value_model_fn 
        self.value_optimizer_fn = value_optimizer_fn 
        self.value_optimizer_lr = value_optimizer_lr 
        self.training_strategy_fn = training_strategy_fn 
        self.evaluation_strategy_fn = evaluation_strategy_fn 
        self.n_warmup_batches = n_warmup_batches 
        self.update_target_every_steps = update_target_every_steps

        #CNN
        self.image_memory = None
        self.rem_steps = 4

        #stats
        self.action_track = {'explore':0, 'exploit':0}
        self.st_ep_reward = None
        self.st_ep_steps = None 

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences

        max_a_q_sp = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))

        q_sa = self.online_model(states).gather(1, actions)

        td_error = q_sa - target_q_sa
        value_loss = td_error.pow(2).mul(0.5).mean()
        #log.info(f'Loss: {value_loss}')
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def interaction_step(self, state, env):
        action = self.training_strategy.select_action(self.online_model, state)
        new_state, reward, is_terminal, info = env.step(action)

        


        is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
        is_failure = is_terminal and not is_truncated


        tmp_state = state
        tmp_new_state = new_state
        
        if self.from_pixel:
            s = self.getImage(env)
            new_state = self.update_image_memory(s)

            tmp_state = np.expand_dims(state, axis=0)
            tmp_new_state = np.expand_dims(new_state, axis=0)
        

        experience = (tmp_state, action, reward,tmp_new_state, float(is_failure))
        self.replay_buffer.store(experience)        
        return new_state, is_terminal, reward
    def update_network(self):
        for target, online in zip(self.target_model.parameters(), self.online_model.parameters()):
            target.data.copy_(online.data) 

    def reset(self,env):
        s = env.reset()
        if self.from_pixel:
            frame = self.getImage(env)
            for i in range(self.rem_steps):
                self.update_image_memory(frame)
            s = self.image_memory
        return s

    def update_image_memory(self,new_frame):
        self.image_memory = np.roll(self.image_memory, 1, axis = 0)
        self.image_memory[0,:,:] = new_frame
        return self.image_memory

    
    def train(self, make_env_fn, make_env_kargs, seed, gamma, 
              max_minutes, max_episodes, goal_mean_100_reward, from_pixel, get_image_fn = False):
        training_start, last_debug_time = time.time(), float('-inf')

        self.checkpoint_dir = tempfile.mkdtemp()
        
        log.info('\n\n\n\n')
        log.info('Training started')

        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed =seed
        self.gamma = gamma
        self.from_pixel = from_pixel
        self.getImage = get_image_fn

        self.st_ep_reward = np.zeros(max_episodes)

        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        torch.manual_seed(self.seed) ; np.random.seed(self.seed); random.seed(self.seed)
        log.info(f'ENV: {self.make_env_kargs["env_name"]}')
        log.info(f'Seed :{seed}')

        #CNN
        if from_pixel:
            log.info(f'Learning from PIXEL')
            env.reset()
            env.close()
            nA = env.action_space.n
            nS = self.getImage(env).shape 
            self.image_memory = np.zeros((self.rem_steps, nS[0], nS[1]))
            nS = (1, self.rem_steps, nS[0], nS[1])
        else:
            nS, nA = env.observation_space.shape[0], env.action_space.n 

        log.info(f'Input Shape: {nS}, Output Shape: {nA}')
        log.info(f'Target update steps: {self.update_target_every_steps}')
        self.target_model = self.value_model_fn(nS, nA)
        self.online_model = self.value_model_fn(nS, nA)
        self.update_network()

        log.info(f'\n\nModel Summary: \n{summary(self.target_model, (4,160,240))}')
        self.value_optimizer = self.value_optimizer_fn(self.online_model, self.value_optimizer_lr)
        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn()
        self.evaluation_strategy = self.evaluation_strategy_fn()
        
        log.info(f'Replay buffer:\n  Size: {self.replay_buffer.max_size}\n  Batch size: {self.replay_buffer.batch_size} \n')        
        log.info(f'Decay Epsilon steps: {self.training_strategy.decay_steps}')        
        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        
        
        log.info(f'Max Episodes: {max_episodes}')
        training_start_ts = time.time()
        for episode in range(max_episodes):
            episode_start = time.time()
            
            log.info(f'\n\nepisode# : {episode}')

            state, is_terminal = self.reset(env), False
            for step in count():
                state, is_terminal, reward = self.interaction_step(state, env)
                self.st_ep_reward[episode] += reward 

                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                if len(self.replay_buffer) > min_samples:
                    experiences = self.replay_buffer.sample()
                    self.temp = experiences

                    #converting np to tensor
                    experiences = self.online_model.load(experiences)
                    self.optimize_model(experiences)
                
                if step % self.update_target_every_steps == 0:
                    self.update_network()

                if is_terminal:
                    gc.collect()
                    break
                if self.training_strategy.exploratory_action_taken:
                    self.action_track['explore'] += 1
                else:
                    self.action_track['exploit'] += 1
            self.save_checkpoint(episode-1, self.online_model)



            #stats

            self.payload = {}
            self.payload['time'] = np.zeros(max_episodes)
            elapsed = time.time() - episode_start
            self.payload['time'][episode] = elapsed         
            
            log.info(f'Total Reward: {self.st_ep_reward[episode]}')
            #log.info(f'episilon value: {self.training_strategy.epsilon}')
            log.info(f'action track: \n{self.action_track}')
            #log.info(f'buffer Size: {len(self.replay_buffer)}')
            log.info(f'Episode Time: {elapsed}')
            

        self.payload['reward'] = self.st_ep_reward
        log.info(f'Training Time: {(time.time() - training_start_ts)/ 60} mins')
        log.info(f'Checkpoint dir: {self.checkpoint_dir}')

        return self.payload

    def get_cleaned_checkpoints(self, n_checkpoints=5):
        # try: 
        #     return self.checkpoint_paths
        # except AttributeError:
        #     self.checkpoint_paths = {}

        self.checkpoint_paths = {}
        paths = glob.glob(os.path.join(self.checkpoint_dir, '*.tar'))
        paths_dic = {int(path.split('.')[-2]):path for path in paths}
        last_ep = max(paths_dic.keys())
        # checkpoint_idxs = np.geomspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=np.int)-1
        checkpoint_idxs = np.linspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=np.int)-1

        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                os.unlink(path)


        return self.checkpoint_paths

    def demo_last(self, title='Fully-trained {} Agent', n_episodes=3, max_n_videos=3):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        last_ep = max(checkpoint_paths.keys())
        self.online_model.load_state_dict(torch.load(checkpoint_paths[last_ep]))

        self.evaluate(self.online_model, env, n_episodes=n_episodes)
        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def demo_progression(self, title='{} Agent progression', max_n_videos=5):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)
        checkpoint_paths = self.get_cleaned_checkpoints()
        for i in sorted(checkpoint_paths.keys()):
            self.online_model.load_state_dict(torch.load(checkpoint_paths[i],map_location=torch.device('cpu')))
            self.evaluate(self.online_model, env, n_episodes=1)

        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            subtitle_eps=sorted(checkpoint_paths.keys()),
                            max_n_videos=max_n_videos)
        #del env
        return HTML(data=data)

    def save_checkpoint(self, episode_idx, model):
        torch.save(model.state_dict(), 
                   os.path.join(self.checkpoint_dir, 'model.{}.tar'.format(episode_idx)))

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = []
        for _ in range(n_episodes):
            s, d = self.reset(eval_env), False
            rs.append(0)
            for _ in count():
                a = self.evaluation_strategy.select_action(eval_policy_model, s)
                s, r, d, _ = eval_env.step(a)
                if self.from_pixel:
                    s = self.getImage(eval_env)
                    s = self.update_image_memory(s)
                rs[-1] += r
                if d: break
        return np.mean(rs), np.std(rs)

        
