from imports import *

class EGreedyExpStrategy():
    def __init__(self, init_epsilon = 1.0, min_epsilon=0.1, decay_steps = 20000, decay_ratio = .0005):

        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.decay_steps = decay_steps
        self.min_epsilon = min_epsilon
        self.epsilons = .01 /np.logspace(-2, 0, decay_steps, endpoint = False) -.01

        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0
        self.exploratory_action_taken = None
        self.rand_values = []

    def _epsilon_update(self):# remaining values after the decay
        self.epsilon = \
        self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        return self.epsilon

    def select_action(self, model, state):
        self.exploratory_action_taken = False
        with torch.no_grad():
            q_values = model(state).detach().cpu().data.numpy().squeeze()
        rand = np.random.rand()
        if  rand > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(len(q_values))
            self.exploratory_action_taken = True
        self.rand_values.append(rand)
        self._epsilon_update()
        #self.exploratory_action_taken = action != np.argmax(q_values)

        return action
class GreedyStrategy():
    def __init__(self):
        self.exploratory_action_taken = False

    def select_action(self, model, state):
        with torch.no_grad():
            q_values = model(state).detach().cpu().data.numpy().squeeze()
            return np.argmax(q_values) 