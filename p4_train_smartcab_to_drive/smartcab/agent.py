import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import itertools as iter
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.actions = ('forward','left','right',None)  # define tuple of actions available to agent

        def choose_action(planner_type=1):
            """ Select action based on various action selection methods, depending on the planner_type parameter 

            Parameters
            ----------
            planner_type:
                            1 .. planner that chooses random action
                            2 .. greedy planner
                            3 .. randomized planner
                            4 .. naive reward matrix planner (decide optimally if all rewards are known at given state and explore unexplored otherwise)

            """
            if (planner_type == 1):
                return random.choice(self.actions)
            elif (planner_type == 2):
                return self.actions[np.argmax(self.q[self.state_index(self.state)])]
            elif (planner_type == 3):
                return self.actions[np.argmax(self.q[self.state_index(self.state)])] if random.random() > 0.1 else random.choice(self.actions)
            elif (planner_type == 4):
                if (np.count_nonzero(self.rewards[self.state_index(self.state)]) == len(self.actions)):   # case where all actions have been explored (assumes zero reward does not occur)
                    #print 'I have learned all rewards in this state'
                    return self.actions[np.argmax(self.rewards[self.state_index(self.state)])]
                else: # case where actions still need to be explored
                    for i in range(len(self.actions)):  # identif first unexplored action and try it
                        if (self.rewards[self.state_index(self.state),i] == 0):
                            return self.actions[i]




        self.choose_action = choose_action
        # compute state representation as ['current waymark'] x ['trafficLight'] x ['directionOncoming'] x ['directionLeft']
        # Remark: leave out 'directionRight', since cars coming from the right will not impact the agent in any way
        self.states = list(iter.product(['forward','left','right'],['red','green'],['left','forward','right',None],['left','forward','right',None]))

        def state_index(input_tuple):
            """ Accept an input tuple such as ('left', 'green' , None, None) and return its
            index in the state matrix.
            """
            return self.states.index(input_tuple)

        self.state_index = state_index
        self.net_reward = 0  # initialize net reward counter
        self.q = 30 * np.ones((len(self.states),len(self.actions)))  # initialize Q-matrix
        self.rewards = np.zeros((len(self.states),len(self.actions))) # initialize rewards matrix (used only by the optional naive static learner)
        self.gamma = 0.05  # set discount factor
        self.alpha = 0.35  # set learning rate
        self.count_time = 0 # set counter for completion time
        self.count_time_list = []


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # Prepare for a new trip; reset any variables here, if required
        self.net_reward = 0  # reset net reward
        self.count_time_list.append(self.count_time)
        self.count_time = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'])

        # Select action according to your policy (choose 4 for naive reward matrix)
        action = self.choose_action(planner_type=4)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Learn policy based on state, action, reward
        self.net_reward += reward
        next_inputs = self.env.sense(self) # retrieve new state (1/2)
        next_state = (self.next_waypoint, next_inputs['light'], next_inputs['oncoming'], next_inputs['left'])  # retrieve new state (2/2)
        ind_t0, ind_t1 = (self.state_index(self.state),self.state_index(next_state))
        # transition rule for q-matrix / learning matrix
        self.q[ind_t0,self.actions.index(action)] = (1 - self.alpha) * self.q[ind_t0,self.actions.index(action)] + self.alpha * (reward + self.gamma * np.max(self.q[ind_t1]))

        # Learn optimal naive rewards (for optional naive rewards matrix learner(only used there))
        self.rewards[ind_t0,self.actions.index(action)] = reward

        """ some print statements that help explore the algorithms workings follow"""
        #print "My state is {0}".format(self.state_index(self.state))

        #print "My state is {0}".format(self.state)
        #print self.rewards[ind_t0]
        #print "Rewards matrix sum is {0}".format(np.sum(self.rewards))

        #print "Q vector is {0}".format(self.q[ind_t0] )
        #print "Q matrix sum is {0}".format(np.sum(self.q))
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        self.count_time += 1  # count updates
        #print self.count_time
        #print 'Mean time till dest is {0} and std is {1}'.format(np.mean(self.count_time_list), np.std(self.count_time_list))


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
