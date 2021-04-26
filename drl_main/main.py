from packages import *
from env import Env
from neural_nets import Conv_net, Values_net
from conf import *
from exploration import Exploration
#main
class Drl(Env):
    def __init__(self,env_name):
        super().__init__(env_name)
        self.exp = Exploration()

        self.seed = conf["seed"]
        self.save_dir = conf['save_dir']
        
#act
class Drl(Drl):
    def __init__(self,env_name):
        super().__init__(env_name)        

        self.use_cuda = torch.cuda.is_available()

        # DNN to predict the most optimal action - we implement this in the Learn section
        if self.pixel:
            self.net = Conv_net(self.state_dim, self.action_dim).float()
        else:
            self.net = Values_net(self.state_dim, self.action_dim).float()

        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = 1
        self.curr_step = 0

        self.save_every = conf['save_every']  # no. of experiences between saving Net

    def act(self, state):
        """
    Given a state, choose an epsilon-greedy action and update value of step.

    Inputs:
    state(LazyFrame): A single observation of the current state, dimension is (state_dim)
    Outputs:
    action_idx (int): An integer representing which action will perform
    """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()


        if self.curr_step > conf['burnin']:
            self.exploration_rate = self.exp.decay()
        

        # increment step
        self.curr_step += 1
        return action_idx

#cache and recall
class Drl(Drl):
    def __init__(self,env_name):
        super().__init__(env_name)
        
        self.memory = deque(maxlen=conf['memory'])
        self.batch_size = conf['batch_size']

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = state.__array__()
        next_state = next_state.__array__()

        self.memory.append((state, next_state, action, reward, done))
        if done:
            self.ep += 1
            if self.ep % self.save_every == 0:
                print(self.ep)
                self.save()

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)

        t_batch = []
        for state, next_state, action, reward, done in batch:
            if self.use_cuda:
                state = torch.tensor(state).cuda()
                next_state = torch.tensor(next_state).cuda()
                action = torch.tensor([action]).cuda()
                reward = torch.tensor([reward]).cuda()
                done = torch.tensor([done]).cuda()
            else:
                state = torch.tensor(state)
                next_state = torch.tensor(next_state)
                action = torch.tensor([action])
                reward = torch.tensor([reward])
                done = torch.tensor([done])
            t_batch.append((state, next_state, action, reward, done))
        batch = t_batch
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

#td estimate
class Drl(Drl):
    def __init__(self,env_name):
        super().__init__(env_name)
        self.gamma = conf['gamma']

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

#Loss and optimization
class Drl(Drl):
    def __init__(self,env_name):
        super().__init__(env_name)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=conf['lr'])
        #self.loss_fn = torch.nn.SmoothL1Loss()
        self.loss_fn = torch.nn.MSELoss()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

#save checkpoint
class Drl(Drl):
    def __init__(self,env_name):
        super().__init__(env_name)
    def save(self):
        save_path = (
            self.save_dir / f"chk/{conf['seed']}_net_{int(self.ep // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"Net saved to {save_path} at step {self.ep}")

#all together
class Drl(Drl):
    def __init__(self,env_name):
        super().__init__(env_name)
        self.burnin = conf['burnin']  # min. experiences before training
        self.learn_every = conf['learn_every']  # no. of experiences between updates to Q_online
        self.sync_every = conf['sync_every']  # no. of experiences between Q_target & Q_online sync
        self.ep = 1

    def learn(self):
        if self.ep % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)
    
    def evaluate(self, num_ep=1):

        rs = []
        for i in range(num_ep):
            state = self.env.reset()
        
            while True:
                action_values = self.net(state, model="online")
                action_idx = torch.argmax(action_values, axis=1).item()

                next_state,reward, done, _ = self.env.step(action_idx)
                state = next_state
                rs.append(reward)
                if done:
                    break;
        return np.mean(rs), np.std(rs)
                

    