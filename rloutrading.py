import numpy as np
import pandas as pd
import xarray as xr
from numpy.random import randn

class QTrading:
    def __init__(self, kappa, sigma, xbar, phi, gamma, c,
                 T, dt, A, B, C, D, xmin=90, xmax=110, buy_min=-5,
                 buy_max=5, inventory_min=-20, inventory_max=20):
        """
        Constructor for the Q-learning program
        
        Parameters
        ----------
        kappa: float
            mean-reverting force
        sigma: float
            noise parameter
        xbar: float
            mean-reverting value
        phi: float
            penalty parameter for the action
        gamma: float
            discount factor of the reinforcement learner
        c: foat
            penalty for inventory
        T: float
            Time duration
        dt: float
            timestep change
        A: int
            
        B: int
        
        C: int
        
        D: int
        
        """
        self.kappa = kappa
        self.sigma = sigma
        self.xbar = xbar
        self.phi = phi
        self.gamma = gamma
        self.c = c
        self.T = T
        self.dt = dt
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.timesteps = self._initialize_timesteps()
        self.inventory_min = inventory_min
        self.inventory_max = inventory_max
        self.inventory = np.arange(inventory_min, inventory_max + 1)
        self.actions = np.arange(buy_min, buy_max + 1)
        self.action_space = self._initialize_df_action()
        self.buckets = self._initialize_price_buckets()
        self.Q = self._initialize_Q_matrix()
        
    def _initialize_timesteps(self):
        time = np.arange(0, self.T, self.dt)
        
        return time
        
        
    def _initialize_df_action(self):
        action_space = self.actions[None, :] + self.inventory[:, None]
        action_space = pd.DataFrame(action_space.copy(),
                                    columns=self.actions, index=self.inventory)

        return action_space
    
    
    def _initialize_price_buckets(self, n=10, ndecimals=4):
        upper_bound = self.xbar + n * self.sigma / np.sqrt(2 * self.kappa)
        lower_bound = self.xbar - n * self.sigma / np.sqrt(2 * self.kappa)
        bucket_size = self.sigma * np.sqrt(self.dt) / 4
        buckets = np.arange(lower_bound, upper_bound, bucket_size).round(ndecimals)
        
        return buckets
        
    
    def _initialize_Q_matrix(self):
        n_timesteps = np.ceil(self.T / self.dt).astype(int)
        n_bucket_prices = len(self.buckets)
        n_inventory = len(self.inventory)
        n_actions = len(self.actions)
        
        state_space = n_timesteps, n_bucket_prices, n_inventory, n_actions
        Q = np.random.randn(*state_space) / 10
        
        
        
    def simulate_ou_process(self, x0=None, nsims=1):
        x0 = self.xbar if x0 is None else x0
        nsteps = len(self.timesteps)
        x = np.zeros((nsteps, nsims))
        x[0,:] = x0
       
        errs = np.random.randn(nsteps - 1,nsims)
        for t in range(nsteps - 1):
            x[t + 1,:] = (x[t,:] + self.dt * (self.kappa * (self.xbar - x[t,:]))
                         + np.sqrt(self.dt) * self.sigma * errs[t,:])

        return x

    
    def simulate_reward_matrix(self):
        reward_dimensions = ["timestep", "inventory", "action"]
        Xt = self.simulate_ou_process()
        Xt = Xt.ravel()
        R = (np.diff(Xt)[:, None, None] * self.inventory[None, :, None]
          - self.phi * self.actions[None, None, :])
        R[-1, :, :] = R[-1, :, :] - self.c * self.inventory[:, None]
        R = xr.DataArray(R, coords=[self.timesteps[1:],
                                    self.inventory, self.actions],
                          dims=reward_dimensions)
        
        return R
    
    
    def get_possible_actions(self, q):
        possible_actions = self.action_space.loc[q]
        mapping = ((self.inventory_min <=  possible_actions) &
                   (possible_actions <= self.inventory_max))
        possible_actions = possible_actions[mapping]
        return possible_actions
    
    
    def step_in_episde(self, R):
        pass
        
    
    def q_learn(self):
        """
        to-do: implement
        """
        pass
