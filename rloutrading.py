import numpy as np
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
        self.T = T
        self.dt = dt
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.inventory = np.arange(inventory_min, inventory_max)
        self.actions = np.arange(buy_min, buy_max)
        
    def simulate_ou_process(self):
        """
        Todo (Leo): generalizar a N simulaciones
        Puntos extra por quitar todos los for loops
        """
        x0 = self.xbar
        time = np.arange(0, self.T, self.dt)
        nsteps = len(time)
        x = np.zeros(nsteps)
        x[0] = x0
        errs = randn(nsteps - 1)
        for t in range(nsteps - 1):
            x[t + 1] = x[t] + self.dt * (self.kappa * (self.xbar - x[t])) +\
            np.sqrt(self.dt) * self.sigma * errs[t]
        
        return x


    def simulate_ou_process_forloop(self, nsims=1):
        time = np.arange(0, self.T, self.dt)
        nsteps = len(time)
        x = np.zeros((nsteps,nsims))
        x[0,:] = self.x0
       
        errs = np.random.randn(nsteps - 1,nsims)
        for t in range(nsteps - 1):
            x[t + 1,:] = x[t,:] + self.dt * (self.kappa * (self.xbar - x[t,:])) + np.sqrt(self.dt) * self.sigma * errs[t,:]
       
        return time, x    

    
    def simulate_reward_matrix(self):
        Xt = self.simulate_ou_process()
        R = np.diff(Xt)[:, None, None] * self.inventory[None, :, None] - self.phi * self.actions[None, None, :]
        
        return R
    
    def q_learn(self):
        """
        to-do: implement
        """
        pass
