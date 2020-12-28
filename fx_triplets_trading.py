import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from numpy.random import randn
from tqdm import tqdm
from math import ceil, sqrt


class FFNet(nn.Module):
    def __init__(self, n ):
        super(FFNet, self).__init__()
        self.layer1 = nn.Linear(7, n)
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

    
class PairFX:
    def __init__(self, x_0, x_min, x_max, buy_min, buy_max, inventory_min, inventory_max, n_decimals_price, d_q, n_price=40):
        self.x_0 = x_0
        self.x_min = x_min
        self.x_max = x_max
        self.buy_min = buy_min
        self.buy_max = buy_max
        self.inventory_min = inventory_min
        self.inventory_max = inventory_max
        self.d_q = d_q
        self.n_price = n_price
        self.inventory = np.arange(inventory_min, inventory_max + d_q, d_q)
        self.n_decimals_price = n_decimals_price
        self.actions = np.arange(buy_min, buy_max + d_q, d_q)
        self.buckets = self._initialize_price_buckets()
        self.action_space = self._initialize_df_action()
        

    def _initialize_price_buckets(self):
        buckets = np.linspace(self.x_min, self.x_max, self.n_price).round(self.n_decimals_price)
        return buckets    
    
    def _initialize_price_buckets_GBPUSD(self):
        buckets = np.linspace(self.x_GBPUSD_min, self.x_GBPUSD_max, self.n_price).round(self.n_decimals_price)
        return buckets        


    def _initialize_df_action(self):
        action_space = self.actions[None, :] + self.inventory[:, None]

        return action_space    
    
    
class DQN_Triplets():
    def __init__(self, a, B, rho, phi, alpha, T, dt, l_A, l_B, e_C, e_D, 
                 pair_1, pair_2, c,
                 eps_0=0.20,  n_decimals_time=4, d_q = 1,
                 n_decimals_price=5, seed=None , K_replacement_freq = 10, delta=0.001, 
                 n_batches =64, config = 32, buffer_size = 1000, gamma=0.9999):
        """
        Constructor for the Triplets Q-learning program

        Parameters
        ----------
        a: float 2x1 vector
            2x1 vector of baseline values of the exchange rates [X_EURUSD;X_GBPUSD] -- see eq. (3)
        B: float 2x2 matrix
            matrix with positive eigenvalues describing the mean-reverting forces -- see eq. (3)
        rho: float 2x1 vector
            noise parameter for the random shocks -- see eq. (3)
        phi: float 2x1 vector
            penalty parameter for the action in EURUSD and GBPUSD 
        alpha: float 2x1 vector
            terminal penalty parameter for inventory in EUR and GBP
        T: float
            Time duration
        dt: float
            timestep change
        l_A: int
            learning rate parameter A in A / (x + B)
        l_B: int
            learning rate parameter B in A / (x + B)
        e_C: int
            exploration rate parameter C in C / (x + D)
        e_D: int
            exploration rate parameter D in C / (x + D)
        """
        self.a = a
        self.B = B
        self.rho = rho
        self.phi = phi
        self.gamma = gamma
        self.c = c
        self.alpha = alpha
        self.T = T
        self.dt = dt
        self.l_A = l_A
        self.l_B = l_B
        self.e_C = e_C
        self.e_D = e_D
        self.pair_1 = pair_1
        self.pair_2 = pair_2
        
        
        self.eps_0 = eps_0
        self.n_decimals_time = n_decimals_time
        self.seed = seed 
        self.K_replacement_freq = K_replacement_freq 
        self.delta = delta
        
        self.timesteps = self._initialize_timesteps()
        
        self.n_batches = n_batches
        self.config = config
        self.buffer_size = buffer_size

        self.Q_M = FFNet(config)
        self.Q_T = FFNet(config)
        self.Q_T.load_state_dict(self.Q_M.state_dict())

        self.replay_buffer = deque(maxlen=buffer_size)
        self.optimizer = optim.Adam(self.Q_M.parameters(), lr=1e-5)#optim.Adam(self.Q_M.parameters(), lr=0.1,  amsgrad = True)
        self.error_history = []

    def exploration_rate(self, iteration):
        return self.e_C / (self.e_D + iteration)

    
    def learning_rate(self, iteration):
        return self.l_A / (self.l_B + iteration)                 
                 
    def _initialize_timesteps(self):
        time = np.arange(0, self.T, self.dt)
        time = np.round(time, self.n_decimals_time)
        return time
    
    
    def simulate_cointegrated_process(self, nsims=1, random_shock=False):
        if random_shock == True:
            x0_pair_1 = np.random.choice(self.pair_1.buckets)
            x0_pair_2 = np.random.choice(self.pair_2.buckets)
        else:
            x0_pair_1 = self.pair_1.x_0
            x0_pair_2 = self.pair_2.x_0
        
        x0 = np.stack((x0_pair_1,x0_pair_2))

        nsteps = len(self.timesteps)
        x = np.zeros((2,nsims,nsteps))
        
        x[...,0] = x0[:, None]
        
        errs = np.random.randn(2,nsims,nsteps-1) * self.rho[:,None,None]

        for t in range(nsteps - 1):
            x[:,:,t + 1] = self.B @ x[:,:,t]   + self.a[:,None] + errs[:,:,t]
            
        x[0,...] = np.clip(x[0,...], self.pair_1.x_min + 1e-10, self.pair_1.x_max - 1e-10)
        x[1,...] = np.clip(x[1,...], self.pair_2.x_min + 1e-10, self.pair_2.x_max - 1e-10)
        
        return x    
    
    
    def get_possible_actions(self,it, q, pair):
            """
            Retrieve the possible actions to take at iteration
            it if having q elements of inventory

            Parameters
            ----------
            it: int
                Iteration number
            q: float
                Number of units in inventory
            """
            total_steps = len(self.timesteps)
            filter_timestep = ceil(pair.inventory_max / max(pair.actions))
            filter_timestep = total_steps - filter_timestep - 2

            if it == total_steps:
                return np.array([0])

            locate_q = pair.action_space[(pair.inventory == q).argmax(),:]
            idx = (pair.inventory_min <=  locate_q) & (locate_q <= pair.inventory_max)
            possible_actions = locate_q[idx] - q

            if it >= filter_timestep:
                take_actions = np.stack(np.meshgrid(*[pair.actions for _ in range(total_steps - it)])).sum(axis=0)
                filter_actions = pair.actions[np.where(q + take_actions == 0)[0]]
                possible_actions = np.intersect1d(possible_actions, filter_actions)

            return possible_actions    

        
        
    def step_function(self, t, q, xt, a):
        q_prime = q + a
        Z = np.random.randn(2) * self.rho 
        
        xt_prime = self.a + self.B @ xt   + Z
        
        #xt_prime = xt_prime[0]
        t_prime = t + self.dt # np.round(t + self.dt, self.n_decimals_time)
        if t_prime < self.T:
            #import pdb;pdb.set_trace()
            reward = q_prime * (xt_prime - xt) - self.phi * (a * a)
        else:
            reward = q_prime * (xt_prime - xt) - self.phi * (a * a) - self.c * (q_prime **2)
        
        reward = np.sum(reward)
        
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
        if random_shock == True:
            x0_pair_1 = np.random.choice(self.pair_1.buckets)
            x0_pair_2 = np.random.choice(self.pair_2.buckets)
        else:
            x0_pair_1 = self.pair_1.x_0
            x0_pair_2 = self.pair_2.x_0
        
        xt = np.stack((x0_pair_1,x0_pair_2))


        
        #xt = np.random.choice(self.buckets)
        q = np.zeros(2)
        eps_k = max(eps_k, self.eps_0)

        for it, t in enumerate(self.timesteps):  #enumerate(self.timesteps[:-1]):     
            pr = np.random.rand()
            
            possible_actions_1 = self.get_possible_actions(it, q[0], self.pair_1)
            possible_actions_2 = self.get_possible_actions(it, q[1], self.pair_2)
            
            if pr <= eps_k:
                new_action_1 = np.random.choice(possible_actions_1)
                new_action_2 = np.random.choice(possible_actions_2)
                new_action = np.stack((new_action_1,new_action_2))
            else:
                aux_in = [[t, xt[0], xt[1], q[0],q[1], act1, act2] for act1 in possible_actions_1 for act2 in possible_actions_2]
                selection_input = torch.tensor(aux_in,requires_grad=False)
                list_actions = self.Q_M.forward(selection_input.float())
                #Q_update_value = list_actions.max()
                new_action = aux_in[list_actions.argmax()][-2:]
                
            reward, xt_prime, t_prime, q_prime = self.step_function(t, q, xt, np.array(new_action) )
            element_to_store = [(t, q, xt), new_action, reward, (t_prime, q_prime, xt_prime)]
            self.replay_buffer.append(element_to_store)
            q = q_prime
            xt = xt_prime
            if iteration > self.n_batches:
                self.learn_step(iteration) #it, t        
 

    def learn_step(self, iteration):
        self.optimizer.zero_grad() 
        
        buffer_size = len(self.replay_buffer)
        
        buffer_samples_index = np.random.choice(buffer_size, size=self.n_batches, replace=False)
        input_matrix = []
        input_matrix_next = []
        index_zero=[]

        n_obs = self.n_batches

        n_max_actions1 = len(self.pair_1.actions)
        n_max_actions2 = len(self.pair_2.actions)

        input_matrix_T = torch.zeros(n_obs, n_max_actions1, n_max_actions2, 7) * -float("inf")
        all_actions1 = torch.arange(self.pair_1.buy_min, self.pair_1.buy_max + self.pair_1.d_q , self.pair_1.d_q)
        all_actions2 = torch.arange(self.pair_2.buy_min, self.pair_2.buy_max + self.pair_2.d_q , self.pair_2.d_q)
        all_actions = np.stack(np.meshgrid(all_actions1, all_actions2)).reshape(2, -1)
        
        rewards = torch.zeros(n_obs, 1)
        for i, sample_index in enumerate(buffer_samples_index):
            (t, q, xt), action, reward, (t_prime, q_prime, xt_prime) = self.replay_buffer[sample_index]
            act1, act2 = action
            it_prime = (self.timesteps == t_prime).argmax()
            possible_actions_1 = self.get_possible_actions(it_prime, q_prime[0], self.pair_1)
            possible_actions_2 = self.get_possible_actions(it_prime, q_prime[1], self.pair_2)
            
            if t_prime == self.T:
                index_zero.append(i)
                
            rewards[i] = reward

            input_matrix.append([t, xt[0], xt[1], q[0],q[1], act1, act2])
            possible_actions = [[t, xt[0], xt[1], q[0],q[1], act1, act2]
                                for act1 in possible_actions_1 for act2 in possible_actions_2]

            possible_actions = torch.tensor(possible_actions)
            N1 = len(possible_actions_1)
            N2 = len(possible_actions_2)
            possible_actions = possible_actions.reshape(N1, N2, -1)
            input_matrix_T[i, :N1, :N2, :] = possible_actions
            input_matrix_next.append([t, xt[0], xt[1], q[0],q[1], 0, 0])

        input_matrix = torch.tensor(input_matrix,requires_grad=False).float()
        Q_pred = self.Q_M.forward(input_matrix)
        
        input_matrix_next = torch.tensor(input_matrix_next,requires_grad=False).float() 
        action_best = self.Q_M.forward(input_matrix_T)[..., -1]   #gd review
        action_best[action_best != action_best] = -float("inf")
        action_best = action_best.reshape(self.n_batches, -1).argmax(axis=-1)
        action_best = all_actions[:, action_best]
        input_matrix_next[:, -2:] = torch.tensor(action_best).T
        Q_next = self.Q_T.forward(input_matrix_next)

        Q_next[index_zero] = 0

        loss = torch.sum((rewards + self.gamma * Q_next - Q_pred) ** 2)
        loss.backward()

        self.optimizer.step() 
        
        self.error_history.append(float(loss))


    
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


    def get_possible_actions(self,it, q):   # zero inventory at the end
        
        if it==(len(self.timesteps)):
            possible_actions = np.array([0])
            return possible_actions
        elif it==(len(self.timesteps)-1):
            possible_actions = np.array([-q])
            possible_actions = np.clip(possible_actions, (self.buy_min) , (self.buy_max) ) 
            return possible_actions
        elif it==(len(self.timesteps)-2):
            locate_q = self.action_space[(self.inventory==q).argmax(),:]
            idx = ((self.buy_min <=  locate_q) &
                       (locate_q <= self.buy_max))
            possible_actions = locate_q[idx] - q 
            return possible_actions
        else:
            locate_q = self.action_space[(self.inventory==q).argmax(),:]
            idx = ((self.inventory_min <=  locate_q) &
                       (locate_q <= self.inventory_max))


            possible_actions = locate_q[idx] - q 
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
            
            possible_actions = [[t_prime, xt_prime, q_prime, act] for act in list_actions]
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
        


