import numpy as np
import pandas as pd
import xarray as xr
from numpy.random import randn
from tqdm import tqdm

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
            lera
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
        self._Q = self._initialize_Q_matrix()

    @property
    def Q(self):
        return self._Q

    def _initialize_timesteps(self):
        #to-do (Leo): Correct rounding feature
        time = np.arange(0, self.T, self.dt)
        time = np.round(time, 4)
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
        coords = [self.timesteps, self.buckets, self.inventory, self.actions]
        dims = ["time", "price", "inventory", "action"]
        n_timesteps = np.ceil(self.T / self.dt).astype(int)
        n_bucket_prices = len(self.buckets)
        n_inventory = len(self.inventory)
        n_actions = len(self.actions)

        state_space = n_timesteps, n_bucket_prices, n_inventory, n_actions
        Q = np.random.randn(*state_space) / 1000
        Q = xr.DataArray(Q, coords=coords, dims=dims)
        return Q



    def simulate_ou_process(self, x0=None, nsims=1, random_shock=False):
        if x0 is None and random_shock is False:
            x0 = self.xbar
        elif random_shock is False:
            x0 = x0
        else:
            x0 = np.random.choice(self.buckets)

        nsteps = len(self.timesteps)
        x = np.zeros((nsteps, nsims))
        x[0,:] = x0

        errs = np.random.randn(nsteps - 1,nsims)
        for t in range(nsteps - 1):
            x[t + 1,:] = (x[t,:] + self.dt * (self.kappa * (self.xbar - x[t,:]))
                         + np.sqrt(self.dt) * self.sigma * errs[t,:])

        x = np.clip(x, self.buckets.min(), self.buckets.max())
        return x


    def simulate_reward_matrix(self, random_shock=False):
        # To-do (Leo): Generalize to N simulations
        reward_dimensions = ["timestep", "inventory", "action"]
        Xt = self.simulate_ou_process(random_shock)
        Xt = Xt.ravel()
        R = (np.diff(Xt)[:, None, None] * self.inventory[None, :, None]
          - self.phi * self.actions[None, None, :] ** 2)
        R[-1, :, :] = R[-1, :, :] - self.c * self.inventory[:, None] ** 2
        R = xr.DataArray(R, coords=[self.timesteps[1:],
                                    self.inventory, self.actions],
                          dims=reward_dimensions)

        return Xt, R

    def exploration_rate(self, iteration):
        return self.C / (self.D + iteration)


    def learning_rate(self, iteration):
        return self.A / (self.B + iteration)


    def get_possible_actions(self, q):
        possible_actions = self.action_space.loc[q]
        mapping = ((self.inventory_min <=  possible_actions) &
                   (possible_actions <= self.inventory_max))
        possible_actions = possible_actions[mapping]
        return possible_actions


    def run_episode(self, iteration, random_shock=False):
        Xt, R = self.simulate_reward_matrix(random_shock=False)
        Xt = self.buckets[np.digitize(Xt, self.buckets)]
        q = 0
        for ix, t in enumerate(self.timesteps[:-1]):
            xt = Xt[ix]
            xt_prime = Xt[ix + 1]
            action, q, Q_update_value = self.step_in_episode(R, t, xt, xt_prime, q, iteration)
            selection_current = dict(time=t, inventory=q, price=xt, action=action)
            self.Q.loc[selection_current] = Q_update_value



    def get_Q_update_value(self, R, t, t_prime, xt, xt_prime, q, q_prime,
                           action, iteration):
        alpha_k = self.learning_rate(iteration)
        reward = (R.sel(timestep=t_prime, inventory=q_prime,
                       action=action).values.max())

        selection_current = dict(time=t, inventory=q, price=xt, action=action)
        selection_next = dict(time=t_prime, inventory=q_prime, price=xt_prime)

        Q_next = self.Q.sel(selection_next).values.max()
        Q_current = self.Q.sel(selection_current)

        Q_update_value = (1 - alpha_k) * Q_current
        Q_update_value += alpha_k * (reward + self.gamma * Q_next)

        return Q_update_value.values.max()


    def step_in_episode(self, R, t, xt, xt_prime, q, iteration):
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
        pr = np.random.rand()
        possible_actions = self.get_possible_actions(q)
        if pr < eps_k:
            new_action, *_ = possible_actions.sample().index
        else:
            list_actions = self.Q.sel(time=t, inventory=q,
                                      price=xt, action=possible_actions.index)
            new_action = list_actions.idxmax().values.max()

        q_prime = q + new_action
        t_prime = np.round(t + self.dt, 4)


        Q_update_value = self.get_Q_update_value(R, t, t_prime, xt, xt_prime, q,
                                                 q_prime, new_action, iteration)

        return new_action, q_prime, Q_update_value


    def q_learn(self, n_iterations, random_shock=False):
        """
        to-do: implement
        """
        for iteration in tqdm(range(n_iterations)):
            self.run_episode(iteration, random_shock=random_shock)


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
