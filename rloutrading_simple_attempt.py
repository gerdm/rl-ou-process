import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from numpy.random import randn
from tqdm import tqdm
from math import ceil


class FFNet(nn.Module):
    def __init__(self, n ):
        super(FFNet, self).__init__()
        # 2 input layer (t,S), 1 output channel, 2 hidden layers with n units each
        self.layer1 = nn.Linear(4, n)
        nn.init.normal_(self.layer1.weight, mean=0, std=0.5)
        
        self.layer2 = nn.Linear(n, n)
        nn.init.normal_(self.layer2.weight, mean=0, std=0.5)
        
        self.layer3 = nn.Linear(n, 1)
        nn.init.normal_(self.layer3.weight, mean=0, std=0.5)

        self.relu = nn.ReLU() #ReLU

    def forward(self, x):
        h1 = self.relu(self.layer1(x))
        h2 = self.relu(self.layer2(h1))
        y = self.layer3(h2)
        
        return y
    

class DQNTrading():
    def __init__(self, kappa, sigma, xbar, phi, gamma, c,
                 T, dt, C, D, n_batches, config, buffer_size, xmin=90, xmax=110, buy_min=-5,
                 buy_max=5, inventory_min=-20, inventory_max=20,
                 eps_0=0.20, n_stdev_X=5, n_price=40, n_decimals_time=4, n_q = 1,
                 n_decimals_price=4, seed=None , K_replacement_freq = 10, delta=0.001):

        self.kappa = kappa
        self.sigma = sigma
        self.xbar = xbar
        self.xmin = xmin
        self.xmax = xmax
        self.phi = phi
        self.gamma = gamma
        self.c = c
        self.T = T
        self.dt = dt
        #self.A = A
        #self.B = B
        self.C = C
        self.D = D
        self.n_q = n_q
        self.eps_0 = eps_0
        self.n_decimals_time = n_decimals_time
        self.n_decimals_price = n_decimals_price
        self.timesteps = self._initialize_timesteps()
        self.buy_min=buy_min
        self.buy_max=buy_max
        self.inventory_min = inventory_min
        self.inventory_max = inventory_max
        self.inventory = np.arange(inventory_min, inventory_max + n_q, n_q)
        self.n_stdev_X = n_stdev_X
        self.n_price = n_price
        self.actions = np.arange(buy_min, buy_max + n_q, n_q)
        self.action_space = self._initialize_df_action()
        self.buckets = self._initialize_price_buckets()
        self.n_batches = n_batches
        self.config = config
        self.Q_M = FFNet(config)
        self.Q_T = FFNet(config)
        self.Q_T.load_state_dict(self.Q_M.state_dict())

        #self.Q_T.parameters = self.Q_M.parameters
        self.replay_buffer = deque(maxlen=buffer_size)
        self.optimizer = optim.Adam(self.Q_M.parameters(), lr=1e-5)#optim.Adam(self.Q_M.parameters(), lr=0.1,  amsgrad = True)
        self.error_history = []
        self.K_replacement_freq=K_replacement_freq
        self.delta=delta


    def _initialize_timesteps(self):
        time = np.arange(0, self.T, self.dt)
        time = np.round(time, self.n_decimals_time)
        return time



    
    def _initialize_df_action(self):
        action_space = self.actions[None, :] + self.inventory[:, None]

        return action_space


    def _initialize_price_buckets(self):
        upper_bound = self.xbar + self.n_stdev_X * self.sigma / np.sqrt(2 * self.kappa)
        lower_bound = self.xbar - self.n_stdev_X * self.sigma / np.sqrt(2 * self.kappa)
        bucket_size = self.sigma * np.sqrt(self.dt) / 0.5
        buckets = np.arange(lower_bound, upper_bound, bucket_size).round(self.n_decimals_price)
        buckets = np.linspace(lower_bound, upper_bound, self.n_price).round(self.n_decimals_price)

        return buckets


    def get_possible_actions_zero_inventory_end(self,it, q):
        if it == len(self.timesteps):
            possible_actions = np.array([0])
        elif it == len(self.timesteps) - 1:
            possible_actions = np.array([-q])
            possible_actions = np.clip(possible_actions, (self.buy_min) , (self.buy_max) ) 
        elif it == len(self.timesteps -2):
            locate_q = self.action_space[(self.inventory==q).argmax(),:]
            idx = ((self.buy_min <=  locate_q) &
                       (locate_q <= self.buy_max))
            possible_actions = locate_q[idx] - q 
        else:
            locate_q = self.action_space[(self.inventory==q).argmax(),:]
            idx = ((self.inventory_min <=  locate_q) &
                       (locate_q <= self.inventory_max))
            possible_actions = locate_q[idx] - q 
        
        return possible_actions
    
    def get_possible_actions(self, it, q):
        total_steps = len(self.timesteps)
        filter_timestep = ceil(self.inventory_max / max(self.actions))
        filter_timestep = total_steps - filter_timestep - 1

        if it == total_steps:
            return np.array([0])
    
        locate_q = self.action_space[(self.inventory==q).argmax(),:]
        idx = (self.inventory_min <=  locate_q) & (locate_q <= self.inventory_max)
        possible_actions = locate_q[idx] - q

        if it >= filter_timestep:
            take_actions = np.stack(np.meshgrid(*[self.actions for _ in range(total_steps - it)])).sum(axis=0)
            filter_actions = self.actions[np.where(q + take_actions == 0)[0]]
            possible_actions = np.intersect1d(take_actions, filter_actions)

        return possible_actions

    def exploration_rate(self, iteration):
        return self.C / (self.D + iteration)

    def simulate_ou_process(self, x0=None, nsims=1, random_shock=False):
        if random_shock == True:
            x0 = np.random.choice(self.buckets)
        else:
            x0 = self.xbar

        nsteps = len(self.timesteps)
        x = np.zeros((nsteps, nsims))
        x[0,:] = x0

        errs = np.random.randn(nsteps - 1, nsims)
        for t in range(nsteps - 1):
            x[t + 1,:] = (x[t,:] + self.dt * (self.kappa * (self.xbar - x[t,:]))
                         + np.sqrt(self.dt) * self.sigma * errs[t,:])

        x = np.clip(x, self.buckets.min()+1e-10, self.buckets.max()-1e-10)
        return x
    
    def learning_rate(self, iteration):
        return self.A / (self.B + iteration)


    def step_function(self, t, q, xt, a):
        q_prime = q + a
        Z = np.random.randn(1) * (self.sigma) / np.sqrt(2*self.kappa) * np.sqrt( 1-np.exp(-2*self.kappa*self.dt) ) 
        xt_prime = xt*np.exp(-self.kappa*self.dt) + self.xbar*(1-np.exp(-self.kappa*self.dt)) + Z
        xt_prime = xt_prime[0]
        t_prime = t + self.dt # np.round(t + self.dt, self.n_decimals_time)
        if t_prime < self.T:
            reward = q_prime * (xt_prime - xt) - self.phi * (a ** 2)
        else:
            reward = q_prime * (xt_prime - xt) - self.phi * (a ** 2) - self.c * (q_prime **2)
                
        return reward, xt_prime, t_prime, q_prime


    def dq_learn(self, n_iterations, random_shock=False):
        eps_k=1
        try:
            for iteration in tqdm(range(n_iterations)):
                if iteration % self.K_replacement_freq == 0:
                    self.Q_T.load_state_dict(self.Q_M.state_dict())
                self.d_run_episode(iteration, eps_k, random_shock=random_shock)
                eps_k = max( eps_k - self.delta , self.eps_0)
        except KeyboardInterrupt:
            print("...stoping process")    


    def d_run_episode(self, iteration, eps_k, random_shock=False):
        xt = np.random.choice(self.buckets)
        q = 0
        eps_k = max(eps_k, self.eps_0)

        for it, t in enumerate(self.timesteps):  #enumerate(self.timesteps[:-1]):     
            pr = np.random.rand()
            possible_actions = self.get_possible_actions_zero_inventory_end(it, q)
            if pr <= eps_k:      ###
                new_action = np.random.choice(possible_actions)
            else:
                selection_input = torch.tensor([[t, xt, q, act] for act in possible_actions],requires_grad=False)
                list_actions = self.Q_M.forward(selection_input.float())
                #Q_update_value = list_actions.max()
                new_action = possible_actions[list_actions.argmax()]
                
            reward, xt_prime, t_prime, q_prime = self.step_function(t, q, xt, new_action)
            element_to_store = [(t, q, xt), new_action, reward, (t_prime, q_prime, xt_prime)]
            self.replay_buffer.append(element_to_store)
            q = q_prime
            xt = xt_prime
            if iteration > self.n_batches:
                self.learn_step(iteration) #it, t

        #if iteration > self.n_batches:
        #    self.learn_step(iteration)
            
    def learn_step(self,iteration): #it, t
        self.optimizer.zero_grad() 
        
        #new_replay_buffer = [v for v in self.replay_buffer if v[0][0]==t]
        
        
        buffer_size = len(self.replay_buffer)
        
        #buffer_samples_index = np.random.randint(0, buffer_size, size=self.n_batches)
        buffer_samples_index = np.random.choice(buffer_size, size=self.n_batches, replace=False)
        input_matrix = []
        input_matrix_next = []
        
        index_zero=[]

        n_obs = self.n_batches
        n_max_actions = len(self.actions)

        input_matrix_T = torch.zeros(n_obs, n_max_actions, 4) * -float("inf")
        all_actions = torch.arange(self.buy_min, self.buy_max + self.n_q , self.n_q)
        rewards = torch.zeros(n_obs,1)
        for i, sample_index in enumerate(buffer_samples_index):
            (t, q, xt), action, reward, (t_prime, q_prime, xt_prime) = self.replay_buffer[sample_index]
            it_prime = (self.timesteps == t_prime).argmax()
            list_actions = self.get_possible_actions_zero_inventory_end(it_prime, q_prime) 
            
            if t_prime == self.T:
                index_zero.append(i)
                
            
            rewards[i] = reward
            input_matrix.append((t, q, xt, action))
            
            possible_actions = [[t_prime, q_prime, xt_prime, act] for act in list_actions]
            n_possible = len(possible_actions)
            input_matrix_T[i, :n_possible, :] = torch.tensor(possible_actions)
            input_matrix_next.append([t_prime, q_prime, xt_prime, 0])

        input_matrix = torch.tensor(input_matrix,requires_grad=False).float()
        Q_pred = self.Q_M.forward(input_matrix)
        
        input_matrix_next = torch.tensor(input_matrix_next,requires_grad=False).float()  #CHECK requires grad
        action_best = self.Q_M.forward(input_matrix_T)[..., -1]   #gd review
        action_best[action_best != action_best] = -float("inf")
        action_best = all_actions[action_best.argmax(axis=-1)]
        input_matrix_next[:, -1] = action_best 
        Q_next = self.Q_T.forward(input_matrix_next)  #Q_T ******

        Q_next[index_zero] = 0

        #print('ingredients')
        #print(Q_next)
        #print(Q_pred)        
        #print(rewards)
                       
        #loss = nn.MSELoss()((Q_pred), (rewards + self.gamma * Q_next))
        loss = torch.sum((rewards + self.gamma * Q_next - Q_pred) ** 2)
        #print(loss)
        loss.backward()

        self.optimizer.step() 
        
        self.error_history.append(float(loss))
        