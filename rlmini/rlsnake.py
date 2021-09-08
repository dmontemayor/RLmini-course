"""Reinforcement Learning Snake - agent and environment"""

import numpy as np
from crpm.ssa import ssa
from crpm.ffn_bodyplan import read_bodyplan
from crpm.ffn_bodyplan import init_ffn
from crpm.ffn_bodyplan import copy_ffn
from crpm.fwdprop import fwdprop
from crpm.lossfunctions import loss
from crpm.gradientdecent import gradientdecent
from crpm.contrastivedivergence import contrastivedivergence

#from enums import *
#import random

class SnakeEnv:
    """ Snake Game Emulator """

    def __init__(self):
        """init snake environment"""
        #Env Parameters
        self.scrh = ? #screen height
        self.scrw = ? #screen width
        self.nfeat = ? #number of features

        #init snake body
        self.snake = self.newsnake()

        #initial food location
        food = [self.scrh//4, self.scrw//4]

        #init snake score
        self.score = 0

        #init state: (what does the agent see?)
        self.state = self.getstate()

    #def reward(self, states):
    #    """define reward function as a function of state. returns a scalar.
    #    """
    #    return 0 #default value

    def newsnake(self):
        """resets snake body."""

        #snake initial position
        snake_x = self.scrw//2
        snake_y = self.scrh//2

        #create snake initial body segments (3 in x direction)
        snake = [[snake_y, snake_x],
                 [snake_y, snake_x-1],
                 [snake_y, snake_x-2]]

        return snake

    def getstate(self, reinit=False):
        """returns state of environment."""
        import numpy as np

        #what if agent could see location of food?
        state = self.food

        ##what if agent could see if location was occupied with snake
        #state = np.zeros((self.scrh, self.scrw)) #init with all locations empty
        ##check if each location is occupied
        #for x in range(scrw):
        #    for y in range(scrh):
        #        loc = [y, x]
        #        if loc in self.snake:
        #            state[y, x] = 1 #location is occupied

        return state


    def take_action(self, action):
        """update snake given action specified and return new state and reward.
        """

        reward = 0 #default reward

        key = None #default action

        #update snake position
        new_head = [self.snake[0][0], self.snake[0][1]]
        if key == 0: #move down
            new_head[0] += 1
        if key == 1: #move up
            new_head[0] -= 1
        if key == 2: #move right
            new_head[1] += 1
        if key == 3: #move left
            new_head[1] -= 1

        #stack new head on snake
        self.snake.insert(0, new_head)

        #check if snake head ate food
        if self.snake[0] == self.food:
            #assign reward
            reward = 1
            #create new food at random position at least one space away from walls
            self.food = None
            while self.food is None:
                newfood = [random.randint(1, self.scrh-1), random.randint(1, self.scrw-1)]
                #update food position if new food position is not in snakes body
                self.food = newfood if newfood not in self.snake else None
        else:
            #pop off snake's tail
            tail = self.snake.pop()

        #return updated state and reward realized
        return self.getstate(), reward

class Agent:
    def __init__(self, discount=0.95, exploration_rate=1.0, exploration_rate_decay=.99, target_every=2):
        """ define deep network hyperparameters"""
        self.discount = discount # how much future rewards are valued w.r.t. current
        self.exploration_rate = exploration_rate # initial exploration rate
        self.exploration_rate_decay = exploration_rate_decay # transition from exploration to expliotation
        self.target_every = target_every #how many iterations to skip before we swap prediciton network with target network

        #retrieve the body plan
        self.bodyplan = read_bodyplan("models/snake_bodyplan.csv")

        #define prediction network
        self.prednet = init_ffn(self.bodyplan)
        self.loss = None #current prediction error

        #init the target network
        self.targetnet = init_ffn(self.bodyplan)

        #init counter used to determine when to update target network with prediction network
        self.iteration = 0

   # Ask model to estimate value for current state (inference)
    def get_value(self, state):
        """ prediction network to predict action values"""

        prediction, _state = fwdprop(state, self.prednet)
        return prediction

   # Ask model to calcualte value keeping current policy
    def get_target_value(self, state):
        """ target network to predict action values"""

        prediction, _ = fwdprop(state, self.targetnet)
        return prediction

    def get_next_action(self, state):
        """ returns e-greedy action given state"""
        greedy_action = self.greedy_action(state)
        random_action = self.random_action()
        action = np.where(np.random.rand(1) > self.exploration_rate, greedy_action, random_action)
        return action

    # Which action has bigger Q-value, estimated by our model (inference).
    def greedy_action(self, state):
        """argmax picks the higher Q-value and returns the index"""
        return np.argmax(self.get_value(state), axis=0)

    def random_action(self):
        """return random action"""
        return np.random.randint(self.bodyplan[-1]["n"], size=1)

    def train(self, state, action, reward, new_state):
        """ will train deep network model by gradient decent """

        # Ask the model for the Q values of the current state (inference)
        state_values = self.get_value(state)
        #print("state_values")
        #print(state_values)


        # Ask the model for the Q values of the new state (target)
        new_state_values = self.get_target_value(new_state)
        #print("new_state_values")
        #print(new_state_values)

        #get network input size
        nfeat = state.shape[0] #network input size
        nlabels = state_values.shape[0] #network output size (actions)
        #print("nfeat and nlabels(actions)")
        #print(nfeat)
        #print(nlabels)

        #print("actions")
        #print(action)


        #update Q Values if any patients took this action
        state_values[iact] = (reward +
                              self.discount * np.amax(new_state_values[:]))

        # Train prediction network
        nobv = 1
        traindata = ? #state[:, intrain].reshape((nfeat, nobv))
        _, self.loss, _ = gradientdecent(self.prednet,
                                         traindata,
                                         state_values[:].reshape((nlabels, nobv)),
                                         "mse", maxepoch=1, healforces=False)
        #print("loss")
        #print(self.loss)
        #print("bias")
        #print(self.prednet[-1]["bias"])
        #print("weights")
        #print(self.prednet[-1]["weight"])

    def update(self, state, action, reward, new_state, validation=None):

        # Train our model with new data
        self.train(state, action, reward, new_state, validation)


        # Finally shift our exploration_rate toward zero (less gambling)
        self.exploration_rate *= self.exploration_rate_decay

        #increment iteration counter
        self.iteration += 1

#def main q learning loop
def run_simulation(agent, simulator, maxstep, stepsize=.5, update=True,
                   enroll_rate=0, enroll_goal=10, minibatchsize=10,
                   online=True, file="buffer"):
    """
    Define Q learning orchestration

    Args:
        agent : determines policy, at every simulation step will decide which
            action to take for every patient in the simulator.
        simulator: determines the evolution of multiple patients in a simulation
            (cohort). Simulator evolves the state of each patient for one
            simulation step given the action dermined by the agent and calculates
            the reward of that state-action pair.
        maxstep: total integer number of simulation steps
        update: Boolean flag when true will allow agent to update the policy.
            When false, the agent will apply its current policy for the duration
            of the simulation.
        enroll_rate: integer number of new patients to add to the
            simulator at every simulation step.
    ChangeLog:
        + offline parameter tells wheather to read or write previous buffer data
        + file parameter is name where replay buffer will be read/written.
        + Randomly enroll new patients - Let simulator determine dropout rate
        + Have training, validation, and tesing patients
            - testing and validation patients enrolled together at 7:3 ratio
                + validation patients used for naive early stopping (overfitting)
            - testing patients enrolled once policy is set - used for reporting
    """

    #read replay buffer for offline learning
    if not online:
        buffer = np.load(file+".npz")
        pid_buffer = buffer['pid']
        group_buffer = buffer['group']
        visit_buffer = buffer['visit']
        state_buffer = buffer['state']
        action_buffer = buffer['action']
        reward_buffer = buffer['reward']
        new_state_buffer = buffer['new_state']
        #pretrain agent
        agent.pretrain(state_buffer,group_buffer)


    #define early stopping frequency
    earlystopfreq = 20#100

    #init policy bias and variance
    bias = np.empty(13)
    sigma = np.empty(13)

    #init learning and step counter
    step = 0
    learning_error = None
    continuelearning = True
    while continuelearning:

        print("- - - - S T E P - - - -  "+str(step))

        #sample policy for online learning and create new replay buffer
        if online:
            #drop out patients who need to leave study
            simulator.dropout()

            #enroll new patients if haven't reached enrollment goal
            if enroll_rate > 0 and simulator.withdrawn.shape[0] < enroll_goal:
                simulator.enroll(enroll_rate)

            #store current state
            state = np.copy(simulator.state)

            #get withdrawn npatients
            withdrawn = np.copy(simulator.withdrawn)

            #query agent for the next action
            action = agent.get_next_action(state, withdrawn)

            #take action, get new state and reward
            new_state, reward = simulator.take_action(action, simtime=stepsize)

            #get patients with valid actions (technically should be defined by withdrawn)
            patients = np.logical_not(withdrawn)
            pid = np.where(np.logical_not(withdrawn))[0]

            #init replay buffer at first simulation step
            if step == 0:
                pid_buffer = np.copy(pid)
                group_buffer = np.copy(simulator.group[patients])
                visit_buffer = np.copy(simulator.visit[patients])
                state_buffer = np.copy(state[:,patients])
                action_buffer = np.copy(action[patients])
                reward_buffer = np.copy(reward[patients])
                new_state_buffer = np.copy(new_state[:,patients])
            elif not np.all(withdrawn):#accumulate replay buffer
                pid_buffer = np.append(pid_buffer, np.copy(pid))
                group_buffer = np.append(group_buffer, np.copy(simulator.group[patients]))
                visit_buffer = np.append(visit_buffer, np.copy(simulator.visit[patients]))
                state_buffer = np.hstack((state_buffer, np.copy(state[:,patients])))
                action_buffer = np.append(action_buffer, np.copy(action[patients]))
                reward_buffer = np.append(reward_buffer, np.copy(reward[patients]))
                new_state_buffer = np.hstack((new_state_buffer, np.copy(new_state[:,patients])))

        #let agent update policy
        if update:
            #prepare mini-batch
            #Select random training experiences
            intrain = np.where(np.logical_not(group_buffer))[0]
            minsize = min(minibatchsize,intrain.shape[0])
            patients = np.random.choice(intrain, size=minsize, replace=False)
            #perform update step
            agent.update(state_buffer[:,patients],
                         action_buffer[patients],
                         reward_buffer[patients],
                         new_state_buffer[:,patients],
                         group_buffer[patients])

            #stop learning if loss function is nan
            if np.isnan(agent.loss):
                continuelearning = False

            #naive early stopping - use validation set periodically
            if step%earlystopfreq==0:
                #Select random validation experiences
                invalid = np.where(group_buffer)[0]
                #validratio = invalid.shape[0]/intrain.shape[0]
                #minsize = min((minibatchsize*invalid.shape[0])//intrain.shape[0], invalid.shape[0])
                #minsize = min(minibatchsize, invalid.shape[0])
                if invalid.shape[0]>0:#minsize>0:
                    #valid_patients = np.random.choice(invalid, size=minsize, replace=False)
                    #calculate QAgent's current pred error using targetnet
                    #Caclulate QAgent's prediction error on validation set.
                    #create arrays to hold first and second moments of error per visit interval
                    etau1 = np.empty(13)
                    etau2 = np.empty(13)
                    #loop over vist intervals
                    for idx in range(13):
                        #init visit interval cumulative error
                        etau1[idx] = 0
                        etau2[idx] = 0
                        #init visit interval sample counter
                        msecount = 0
                        #get visit interval index
                        tau = idx-6
                        #loop over validation patients
                        #invalid = np.where(group_buffer)[0]
                        for pid in np.unique(pid_buffer[invalid]):
                            #get samples pertaining to this patient
                            psamples = np.where(pid_buffer==pid)[0]
                            #loop over visits
                            for visit in np.unique(visit_buffer[psamples]):
                                #if visit+interval index exists for this patient then
                                if np.isin(visit+tau,visit_buffer[psamples]):
                                    #get visit+interval index
                                    vidx = np.where(visit_buffer[psamples]==visit+tau)[0][0]
                                    #get observed outcome for visit+interval
                                    obstau = simulator.outcome(state_buffer[:,psamples[vidx]])
                                    #get visit index
                                    vidx = np.where(visit_buffer[psamples]==visit)[0][0]
                                    #get observed outcome for visit
                                    obs = simulator.outcome(state_buffer[:,psamples[vidx]])
                                    #construct state to prognose
                                    progstate = state_buffer[:,psamples[vidx]] #start with visit state
                                    #change time to indicate visit time interval
                                    #progstate[-1] = state_buffer[-1,vtidx]-state_buffer[-1,vidx]
                                    progstate[-1] = stepsize*tau
                                    #make sure progstate is nx1
                                    progstate = progstate.reshape((-1,1))
                                    #calculate predicted return for (state(visit),time interval)
                                    rhat = np.squeeze(agent.get_target_value(progstate))
                                    #accumulate square error for visit interval
                                    resi = rhat - np.log(obstau/obs)
                                    etau1[idx] += resi
                                    etau2[idx] += resi*resi
                                    #increment visit interval sample counter
                                    msecount += 1
                                #end if
                            #end loop over visits
                        #end loop over patients
                        #save cumulative second moment
                        #normalize visit interval accumulated error
                        etau1[idx] /= msecount
                        etau2[idx] /= msecount
                        #print("increment")
                        #print(tau)
                        #print(msecount)
                    #end loop over visit intervals
                    bias = etau1
                    sigma = np.sqrt(etau2-etau1*etau1)
                    print("bias")
                    print(etau1)
                    print("MSE")
                    print(etau2)
                    print("stdev")
                    print(sigma)

                    #define current error
                    curr_err = np.mean(etau2)
                    #curr_err = np.sum(etau2*np.exp(-np.abs(etau2-etau2.shape[0]/2)))

                    #save variance if first time
                    if learning_error is None:
                        learning_error = curr_err

                    #stop learning if current error is greater than previous error
                    if curr_err>learning_error:
                        print("early stopping!")
                        continuelearning = False

                    #update learning error
                    learning_error = curr_err

        #complete learning step
        step += 1

        #stop learning after maxsteps
        if step > maxstep:
            continuelearning = False

    #save replay buffer at end of online learning
    if online:
        np.savez(file,
                 pid=pid_buffer,
                 group=group_buffer,
                 visit=visit_buffer,
                 state=state_buffer,
                 action=action_buffer,
                 reward=reward_buffer,
                 new_state=new_state_buffer
                 )

    #Return predictions and observations on validation set.
    #init Prediction array with dimensions (ninterval,npatient)
    invalid = np.where(group_buffer)[0]
    patients = np.unique(pid_buffer[invalid])
    npats = patients.shape[0]
    pred = np.empty((13,npats))
    nvisit = np.max(visit_buffer)
    obv = np.full((nvisit,npats),np.nan)
    #loop over validation patients
    for pidx in range(npats):
        #get patient id
        pid = patients[pidx]
        #get samples pertaining to this patient
        psamples = np.where(pid_buffer==pid)[0]
        #loop over visits to get outcomes
        for visit in np.unique(visit_buffer[psamples]):
            #get visit index
            vidx = np.where(visit_buffer[psamples]==visit)[0][0]
            #get observed outcome for visit
            obv[visit-1,pidx] = simulator.outcome(state_buffer[:,psamples[vidx]])
            #make predictions based off mid visit 3 state (midpoint)
            if visit == 0:
                #construct state to prognose based on visit0 state
                progstate = state_buffer[:,psamples[vidx]]
                #loop over visit intervals
                for idx in range(13):
                    #get visit interval index
                    tau = idx-6
                    #edit time to indicate prognosis time interval
                    progstate[-1] = stepsize*tau
                    #make sure progstate is nx1
                    progstate = progstate.reshape((-1,1))
                    #calculate predicted return for (state(visit3),time interval)
                    pred[idx,pidx] = agent.get_value(progstate)
                #end loop over intervals
            #end if visit number is 0
        #end loop over visits
    #end loop over patients
    print("predictions")
    print(pred)
    print("observations")
    print(obv)


    return obv, pred, bias, sigma



"""Snake Game"""

import random
import curses

def snakegame():
    """Snake Game rendered with ascii characters
    """
    #curses to initialize screen
    screen = curses.initscr()

    #set initial curser to zero so doesn't show up on screen
    curses.curs_set(0)

    #get screen width and height
    scrh, scrw = screen.getmaxyx()

    #create a new window using width and height and starting at top-left corner
    window = curses.newwin(scrh, scrw, 0, 0)

    #set to accept keypad input
    window.keypad(1)

    #refresh every 100 ms
    window.timeout(100)

    #set snake initial position in middle of screen
    snake_x = scrw//2
    snake_y = scrh//2

    #create snake initial body segments (3 in x direction)
    snake = [[snake_y, snake_x],
             [snake_y, snake_x-1],
             [snake_y, snake_x-2]]

    #have initial key entry be to the right so snake doesn't hit its self
    key = curses.KEY_RIGHT

    #initial food some where behind the snake
    food = [scrh//4, scrw//4]

    #add the diamond character on the window to represent the food
    window.addch(food[0], food[1], curses.ACS_DIAMOND)

    #init snake alive with zero points
    alive = True
    score = 0

    #main game loop
    while alive:

        window.refresh()
        #display score
        window.addstr(0, 0, str(score))

        #get action input
        next_key = window.getch()
        #Note: next_key will = -1 if no key is pressed

        #keep moving in same direction unless something was pressed
        key = key if next_key == -1 else next_key

        #update snake position
        new_head = [snake[0][0], snake[0][1]]
        if key == curses.KEY_DOWN:
            new_head[0] += 1
        if key == curses.KEY_UP:
            new_head[0] -= 1
        if key == curses.KEY_RIGHT:
            new_head[1] += 1
        if key == curses.KEY_LEFT:
            new_head[1] -= 1

        #stack new head on snake
        snake.insert(0, new_head)

        #check if snake head ate food
        if snake[0] == food:
            #increment score
            score += 1
            #create new food at random position at least one space away from walls
            food = None
            while food is None:
                newfood = [random.randint(1, scrh-1), random.randint(1, scrw-1)]
                #update food position if new food position is not in snakes body
                food = newfood if newfood not in snake else None
            #add new food character to window
            window.addch(food[0], food[1], curses.ACS_DIAMOND)
        else:
            #pop off snake's tail
            tail = snake.pop()
            #add a space character where the tail was to erase from window
            window.addch(tail[0], tail[1], ' ')

        #add head of snake to the window as checkerboard character
        window.addch(snake[0][0], snake[0][1], curses.ACS_CKBOARD)

        #check to see if snake deaded
        #head of snake's y position hits bottom or top of screen
        if snake[0][0] in [0, scrh]:
            alive = False
        #head of snake's x position hits left or right of screen
        if snake[0][1] in [0, scrw]:
            alive = False
        #snake's head hits anywhere on snakes body
        if snake[0] in snake[1:]:
            alive = False

    #end main game loop

    #close window and return score
    curses.endwin()
    return score

if __name__ == '__main__':
    snakegame()
