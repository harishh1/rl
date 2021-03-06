from packages import *
from conf import *

class MetricLogger:
    def __init__(self, save_dir, seed):
        self.save_log = save_dir / "log"
        seeds = conf['seeds']
        if len(seeds):s = f"Seed: {seed}, Pos: {seeds.index(seed)+1} out of {len(seeds)}\n" 
        else: s= 'Running on a random seed'
        with open(self.save_log, "w") as f:
            f.write(
                f"{s} \n"
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.mean_ep_reward_plot = save_dir / "reward_plot.png"
        #self.mean_ep_length_plot = save_dir / "length_plot.png"
        self.mean_ep_loss_plot = save_dir / "loss_plot.png"
        self.mean_ep_q_plot = save_dir / "q_plot.png"
        self.ep_explr_plot =  save_dir / "exploration.png"

        self.save_file = save_dir / f'res_arrays/{seed}_res_arr.npy'

        self.mean_ep_reward = []
        self.mean_ep_length = []
        self.mean_ep_loss = []
        self.mean_ep_q = []


        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        self.ep_explr = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        last_n_rec = conf['log_every_ep']
        self.mean_ep_reward.append( np.round(np.mean(self.ep_rewards[-last_n_rec:]), 3))
        self.mean_ep_length.append( np.round(np.mean(self.ep_lengths[-last_n_rec:]), 3))
        self.mean_ep_loss.append( np.round(np.mean(self.ep_avg_losses[-last_n_rec:]), 3))
        self.mean_ep_q.append( np.round(np.mean(self.ep_avg_qs[-last_n_rec:]), 3))
        self.ep_explr.append(epsilon)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{self.mean_ep_reward[-1]:15.3f}{self.mean_ep_length[-1]:15.3f}{self.mean_ep_loss[-1]:15.3f}{self.mean_ep_q[-1]:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["mean_ep_reward", "mean_ep_loss", "mean_ep_q","ep_explr"]:
            plt.plot(np.array(range(len(getattr(self, f"{metric}"))))*last_n_rec, getattr(self, f"{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()
    def save(self):
        res = {
            'reward': self.ep_rewards
        }
        np.save(self.save_file, res)
        print(self.save_file.name)