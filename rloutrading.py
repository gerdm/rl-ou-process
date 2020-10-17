import numpy as np
import pandas as pd
import xarray as xr
from numpy.random import randn
from tqdm import tqdm


class QTrading:
    def __init__(self, kappa, sigma, xbar, phi, gamma, c,
                 T, dt, A, B, C, D, xmin=90, xmax=110, buy_min=-5,
                 buy_max=5, inventory_min=-20, inventory_max=20,
                 eps_0=0.01, n_stdev_X=5, n_price=40, n_decimals_time=4,
                 n_decimals_price=4):
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
            used in learning rate
        B: int
            used in learning rate
        C: int
            used in exploration rate
        D: int
            used in exploration rate        
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
        self.eps_0 = eps_0
        self.n_decimals_time = n_decimals_time
        self.n_decimals_price = n_decimals_price
        self.timesteps = self._initialize_timesteps()
        self.buy_min=buy_min
        self.buy_max=buy_max
        self.inventory_min = inventory_min
        self.inventory_max = inventory_max
        self.inventory = np.arange(inventory_min, inventory_max + 1)
        self.n_stdev_X = n_stdev_X
        self.n_price = n_price
        self.actions = np.arange(buy_min, buy_max + 1)
        self.action_space = self._initialize_df_action()
        self.buckets = self._initialize_price_buckets()
        self._Q = self._initialize_Q_matrix()
        self.big_n_Q_dimension = (len(self.timesteps) * len(self.buckets) *
                                  len(self.inventory) * len(self.actions))


    @property
    def Q(self):
        return self._Q

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


    def _initialize_Q_matrix(self):
        n_timesteps = np.ceil(self.T / self.dt).astype(int)
        n_bucket_prices = len(self.buckets)
        n_inventory = len(self.inventory)
        n_actions = len(self.actions)

        state_space = n_timesteps, n_bucket_prices, n_inventory, n_actions
        Q = np.random.rand(*state_space) / 1000000
        
        aposs=self.actions
        
        for it in range(len(self.timesteps)):
            for q in range(n_inventory):
                mask = np.ones(aposs.shape,dtype=bool)
                poss = self.get_possible_actions_zero_inventory_end(it,self.inventory[q])-self.buy_min
                mask[poss] = False

                Q[it,:,q,mask]=-np.inf
                
        return Q


    
    def get_possible_actions_zero_inventory_end(self,it, q):
        
        if it==(len(self.timesteps)-1):
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


    def simulate_reward_matrix(self, random_shock=False):    
        Xt = self.simulate_ou_process(random_shock)
        Xt = Xt.ravel()
        R = (np.diff(Xt)[:, None, None] * self.inventory[None, :, None]
          - self.phi * self.actions[None, None, :] ** 2)
        R[-1, :, :] = R[-1, :, :] - self.c * self.inventory[:, None] ** 2

        return Xt, R

    def exploration_rate(self, iteration):
        return self.C / (self.D + iteration)


    def learning_rate(self, iteration):
        return self.A / (self.B + iteration)

    def get_possible_actions(self, q):
        locate_q = self.action_space[(self.inventory == q).argmax(), :]
        idx = ((self.inventory_min <=  locate_q) &
                (locate_q <= self.inventory_max))
        
        possible_actions = locate_q[idx] - q 
        return possible_actions


    def get_position_indices(self, t, price=None, inventory=None, action=None):
        """
        (time, bucket_index, inventory_index, action_index)
        """
        if inventory is None:
            raise ValueError("inventory cannot be None")
        elif price is None and action is None:
            raise ValueError("Price and action cannot be none")

        if action is None:
            i_t = (self.timesteps == t).argmax()
            i_bucket = (self.buckets == price).argmax()
            i_q = (self.inventory == inventory).argmax()
            return i_t, i_bucket, i_q, slice(None)
        elif price is None:
            i_t = (self.timesteps == t).argmax()
            i_q = (self.inventory == inventory).argmax()
            i_action = (self.actions == action).argmax()
            return i_t, slice(None), i_q, i_action
        else:
            i_t = (self.timesteps == t).argmax()
            i_bucket = (self.buckets == price).argmax()
            i_q = (self.inventory == inventory).argmax()
            i_action = (self.actions == action).argmax()
            return i_t, i_bucket, i_q, i_action



    def run_episode(self, iteration, random_shock=False):
        Xt, R = self.simulate_reward_matrix(random_shock=random_shock)
        Xt = self.buckets[np.digitize(Xt, self.buckets)]
        q = 0
        for it, t in enumerate(self.timesteps[:-1]):
            xt = Xt[it]
            xt_prime = Xt[it + 1]
            action, q_prime, Q_update_value = self.step_in_episode(R, it, t, xt, xt_prime, q, iteration)
            selection_current = self.get_position_indices(t, xt, q, action)
            self.Q[selection_current] = Q_update_value
            q = q_prime


    def get_Q_update_value(self, R, t, t_prime, xt, xt_prime, q, q_prime,
                           action, iteration):
        alpha_k = self.learning_rate(iteration)

        i_t, _, i_q, i_action = self.get_position_indices(t=t, inventory=q_prime,
            action=action)     
        reward = R[i_t, i_q, i_action]

        selection_current = self.get_position_indices(t=t, inventory=q, price=xt, action=action)
        selection_next = self.get_position_indices(t=t_prime, inventory=q_prime, price=xt_prime)

        Q_next = self.Q[selection_next].max()
        Q_current = self.Q[selection_current]

        Q_update_value = (1 - alpha_k) * Q_current
        Q_update_value += alpha_k * (reward + self.gamma * Q_next)

        return Q_update_value


    def step_in_episode(self, R, it, t, xt, xt_prime, q, iteration):
        """

        Paramters
        ---------
        R: xr.DataArray
            Reward Matrix
        q: int
            current inventory
        iteration: int
            simulation iteration
        """
        eps_k = self.exploration_rate(iteration)
        eps_k = max(eps_k, self.eps_0)
        pr = np.random.rand()
        possible_actions = self.get_possible_actions_zero_inventory_end(it,q)
        if pr < eps_k:
            new_action = np.random.choice(possible_actions)
        else:
            selection = self.get_position_indices(t, xt, q)
            possible_actions_bool = ((self.actions[:, None] - possible_actions[None, :]) == 0).any(axis=1)
            list_actions = self.Q[selection].ravel()
            list_actions[~possible_actions_bool] = -np.inf
            new_action_ix = np.argmax(list_actions)
            new_action = self.actions[new_action_ix]

        q_prime = q + new_action
        t_prime = np.round(t + self.dt, self.n_decimals_time)


        Q_update_value = self.get_Q_update_value(R, t, t_prime, xt, xt_prime, q,
                                                 q_prime, new_action, iteration)

        return new_action, q_prime, Q_update_value


    def q_learn(self, n_iterations, random_shock=False):
        try:
            for iteration in tqdm(range(n_iterations)):
                try:
                    self.run_episode(iteration, random_shock=random_shock)
                except IndexError:
                    self.run_episode(iteration, random_shock=random_shock)
        except KeyboardInterrupt:
            print("...stoping process")
            
            
if __name__ == "__main__":
    import json
    import sys
    from datetime import datetime

    _, n_iterations, filename = sys.argv
    parameters = json.load(open(filename))
    today = datetime.now().strftime("%Y%m%d%H%M")
    filename = f"{today}-qtrading.nc"
    qtr = QTrading(**parameters)
    qtr.q_learn(n_iterations)
    qtr.Q.to_netcdf(filename)
