# Imports.
import numpy as np
import numpy.random as npr
import gc #???
import matplotlib.pyplot as plt

from SwingyMonkey import SwingyMonkey


class Learner(object):
    def __init__(self, a, b):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.eps = 0.01
        self.eta = a
        self.xi = b
        self.gamma = 0.9
        self.grav_ind = None
        self.n_dist_bins = 3
        self.n_mtop_bins = 3
        self.n_ttop_bins = 8
        self.n_vel_bins = 3
        self.dist_bins = np.linspace(-120, 480, num=self.n_dist_bins)
        self.mtop_bins = np.linspace(0, 400, num=self.n_mtop_bins)
        self.ttop_bins = np.linspace(-150, 150, num=self.n_ttop_bins)
        self.vel_bins = np.linspace(-30, 20, num=self.n_vel_bins)
        # self.w = np.zeros((4, self.nbins+1, self.nbins+1, self.nbins+1, self.nbins+1, 2))
        self.w = np.zeros((4, self.n_dist_bins+1, self.n_mtop_bins+1, self.n_ttop_bins+1, self.n_vel_bins+1, 2))
        # self.w = np.concatenate((np.ones((4, self.nbins+1, self.nbins+1, self.nbins+1, self.nbins+1, 1)), np.zeros((4, self.nbins+1, self.nbins+1, self.nbins+1, self.nbins+1, 1))), axis=5)
        self.niters = 0
        # self.D = 5
        # self.w = [np.zeros((self.D, )) for i in range(self.D)]

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.grav_ind = None

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # Don't jump at beginning to determine gravity
        if not self.last_state:
            self.last_action = 0
            self.last_state  = state
            return self.last_action

        # Determine gravity
        if self.last_action == 0:
            self.grav_ind = self.last_state['monkey']['vel'] - state['monkey']['vel'] - 1
            # print(self.grav_ind)

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        # new_action = npr.rand() < 0.1
        # epsilon greedy
        last_dist_ind = np.digitize(self.last_state['tree']['dist'], self.dist_bins)
        last_mtop_ind = np.digitize(self.last_state['monkey']['top'], self.mtop_bins)
        last_ttop_ind = np.digitize(self.last_state['tree']['top'] - self.last_state['monkey']['top'], self.ttop_bins)
        last_vel_ind = np.digitize(self.last_state['monkey']['vel'], self.vel_bins)

        dist_ind = np.digitize(state['tree']['dist'], self.dist_bins)
        mtop_ind = np.digitize(state['monkey']['top'], self.mtop_bins)
        ttop_ind = np.digitize(state['tree']['top'] - state['monkey']['top'], self.ttop_bins)
        vel_ind = np.digitize(state['monkey']['vel'], self.vel_bins)

        # last_state_ind_list = np.array([np.digitize(self.last_state['tree']['dist'], self.dist_bins), np.digitize(self.last_state['monkey']['top'], self.mtop_bins), \
        #                 np.digitize(self.last_state['tree']['top'], self.ttop_bins), np.digitize(self.last_state['monkey']['vel'], self.vel_bins)])
        # state_ind_list = np.array([np.digitize(state['tree']['dist'], self.dist_bins), np.digitize(state['monkey']['top'], self.mtop_bins), \
        #                 np.digitize(state['tree']['top'], self.ttop_bins), np.digitize(state['monkey']['vel'], self.vel_bins)])
        # last_state_ind = [[i] for i in last_state_ind_list]
        # state_ind = [[i] for i in state_ind_list]

        # print(self.Phi(self.last_state, self.last_action).shape)
        
        # update
        new_action = 0
        self.niters = self.niters + 1
        # print(self.eps * np.exp(self.niters / -100))
        if npr.rand() < self.eps * np.exp(self.niters / -10000):
            new_action = int(npr.rand() < 0.5)
        else:
            new_action = np.argmax(self.w[self.grav_ind, dist_ind, mtop_ind, ttop_ind, vel_ind, :])
        # Q = np.dot(self.w, self.Phi(self.last_state, self.last_action))
        prev = self.w[self.grav_ind, last_dist_ind, last_mtop_ind, last_ttop_ind, last_vel_ind, self.last_action]
        grad = prev - (self.last_reward + self.gamma * max(self.w[self.grav_ind, dist_ind, mtop_ind, ttop_ind, vel_ind, :]))
        # print(prev, grad)
        self.w[self.grav_ind, last_dist_ind, last_mtop_ind, last_ttop_ind, last_vel_ind, self.last_action] = prev - self.eta * np.exp(self.niters / -self.xi) * grad

        new_state  = state
        
        self.last_action = new_action
        self.last_state  = new_state
        return new_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

    gc.collect() #???
    # Select agent.

    etas = [1]
    xis = [1,10,100,1000,10000]

    for eta in etas:
        for xi in xis:
            agent = Learner(eta, xi)

            assert(np.count_nonzero(agent.w) == 0)

            # Empty list to save history.
            hist = []
            # Run games.
            nepochs = 100
            run_games(agent, hist, nepochs, 0)
            print(eta, xi, np.average(hist), np.std(hist))

            # Save history.
            # np.save('hist',np.array(hist))

            # Plot histogram
            # plt.hist(hist)
            # plt.title("Performance with "+str(nepochs)+" epochs")
            # plt.xlabel("Score")
            # output_file = '.png'
            # plt.savefig(output_file)
            # plt.show()