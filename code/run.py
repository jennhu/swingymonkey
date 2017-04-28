# for moving average
from __future__ import division
from pylab import plot, ylim, xlim, show, xlabel, ylabel, grid
from numpy import linspace, loadtxt, ones, convolve

# other imports
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from SwingyMonkey import SwingyMonkey

class Learner(object):
    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.eps = 0
        self.eta = 0.5
        self.xi = 1000
        self.gamma = 0.75
        self.grav_ind = None
        self.n_dist_bins = 5
        self.n_mtop_bins = 5
        self.n_ttop_bins = 7
        self.n_vel_bins = 5
        self.dist_bins = np.linspace(-120, 480, num=self.n_dist_bins)
        self.mtop_bins = np.linspace(0, 400, num=self.n_mtop_bins)
        self.ttop_bins = np.linspace(-150, 150, num=self.n_ttop_bins)
        self.vel_bins = np.linspace(-30, 20, num=self.n_vel_bins)
        self.w = np.zeros((4, self.n_dist_bins+1, self.n_mtop_bins+1, self.n_ttop_bins+1, self.n_vel_bins+1, 2))
        self.niters = 0

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

        # to determine gravity, don't jump at beginning
        if not self.last_state:
            self.last_action = 0
            self.last_state  = state
            return self.last_action

        # determine gravity
        if self.last_action == 0:
            self.grav_ind = self.last_state['monkey']['vel'] - state['monkey']['vel'] - 1

        # epsilon greedy
        last_dist_ind = np.digitize(self.last_state['tree']['dist'], self.dist_bins)
        last_mtop_ind = np.digitize(self.last_state['monkey']['top'], self.mtop_bins)
        last_ttop_ind = np.digitize(self.last_state['tree']['top'] - self.last_state['monkey']['top'], self.ttop_bins)
        last_vel_ind = np.digitize(self.last_state['monkey']['vel'], self.vel_bins)

        dist_ind = np.digitize(state['tree']['dist'], self.dist_bins)
        mtop_ind = np.digitize(state['monkey']['top'], self.mtop_bins)
        ttop_ind = np.digitize(state['tree']['top'] - state['monkey']['top'], self.ttop_bins)
        vel_ind = np.digitize(state['monkey']['vel'], self.vel_bins)

        # update
        new_action = 0
        self.niters = self.niters + 1
        if npr.rand() < self.eps * np.exp(self.niters / -10000):
            new_action = int(npr.rand() < 0.5)
        else:
            new_action = np.argmax(self.w[self.grav_ind, dist_ind, mtop_ind, ttop_ind, vel_ind, :])
        prev = self.w[self.grav_ind, last_dist_ind, last_mtop_ind, last_ttop_ind, last_vel_ind, self.last_action]
        grad = prev - (self.last_reward + self.gamma * max(self.w[self.grav_ind, dist_ind, mtop_ind, ttop_ind, vel_ind, :]))
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

    agent = Learner()

    # Empty list to save history.
    hist = []
    # Run games.
    nepochs = 200
    run_games(agent, hist, nepochs, 0)

    # print(np.average(hist), np.std(hist))
    # def movingaverage(interval, window_size):
    # 	window= np.ones(int(window_size))/float(window_size)
    # 	return np.convolve(interval, window, 'same')
    # x = range(1,201)
    # y = hist
    # plot(x,y,"k.")
    # y_av = movingaverage(y, 10)
    # plot(x, y_av,"r")
    # xlim(0,200)
    # xlabel("Epoch")
    # ylabel("Score")
    # grid(True)
    # show()
    # plt.savefig('movingav.png')

    # Save history.
    # np.save('hist',np.array(hist))

    # Plot histogram
    # plt.hist(hist)
    # plt.title("Performance with "+str(nepochs)+" epochs")
    # plt.xlabel("Score")
    # output_file = 'histo.png'
    # plt.savefig(output_file)
    # plt.show()