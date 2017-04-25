# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey


class Learner(object):
    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.D = 5
        self.eps = 0.2
        self.eta = 0.001
        self.grav = None
        self.gamma = 0.9
        self.w = [np.zeros((self.D, )) for i in range(self.D)]

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def Phi(self, state, action):
        return np.array([1, state['tree']['dist'], state['tree']['top']-state['monkey']['top'], state['monkey']['vel'], action])

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        # If no gravity determined yet, don't jump
        if not self.last_state:
            self.last_action = 0
            self.last_state  = state
            return 0

        # Determine gravity
        if self.last_action == 0:
            self.grav = self.last_state['monkey']['vel'] - state['monkey']['vel'] 
            print(self.grav)
        grav_ind = self.grav - 1

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        # new_action = npr.rand() < 0.1
        # epsilon greedy
        if npr.rand() < self.eps:
            new_action = npr.rand() < 0.5
        else:
            new_action = np.argmax([np.dot(self.w[grav_ind], self.Phi(state, 0)), np.dot(self.w[grav_ind], self.Phi(state, 1))])

        print(self.Phi(self.last_state, self.last_action).shape)
        
        # update
        Q = np.dot(self.w, self.Phi(self.last_state, self.last_action))
        grad = (Q - (self.last_reward + self.gamma * max(np.dot(self.w[grav_ind], self.Phi(state, 0)), np.dot(self.w[grav_ind], self.Phi(state, 1))))) * self.Phi(self.last_state, self.last_action)
        self.w[grav_ind] = self.w[grav_ind] - self.eta * grad

        new_state  = state
        
        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

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

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 200, 100)

	# Save history. 
	np.save('hist',np.array(hist))


