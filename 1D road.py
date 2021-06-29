import os
import random

import gym
import numpy as np
from gym.vector.utils import spaces
from gym.wrappers import Monitor
from matplotlib import pyplot as plt
from stable_baselines import PPO2, results_plotter, ACKTR, A2C, PPO1
from stable_baselines.bench import load_results
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.results_plotter import ts2xy


class Station:

    def __init__(self, max_bikes):
        self.max_bikes = max_bikes
        self.bikes = 0

    def set_bikes(self, bikes):
        self.bikes = bikes

    def has_bikes(self):
        if self.bikes > 0:
            return True
        else:
            return False

    def get_bikes(self):
        return self.bikes

    def take_bike(self):
        self.bikes = self.bikes - 1

    def add_bike(self):
        self.bikes = self.bikes + 1

    def clear(self):
        self.bikes = 0

class BikesEnv(gym.Env):

    def __init__(self, stations, bikes_in_circulation, max_hourly_customers, model_name):
        super(BikesEnv, self).__init__()

        # Save Params
        self.bikes_in_circulation = bikes_in_circulation
        self.max_hourly_customers = max_hourly_customers
        self.hour = 0

        self.data_file = open(("data/" + model_name + ".txt"), "w")
        self.temp_day_reward = 0

        self.action_space = spaces.Box(np.array([-1, -1, -1, -1, -1]), np.array([+1, +1, +1, +1, +1]))
        self.observation_space = spaces.Box(np.array([-1, -1, -1, -1, -1]), np.array([+20, +20, +20, +20, +20]))

        # create the stations
        # for our 1D env, the stations are in an ordered list.
        self.stations = []
        for i in range(stations):
            # For this original program, each station can hold all the bikes in circulation
            # this wouldn't be true for a realistic implementation.
            self.stations.append(Station(bikes_in_circulation))

        # randomize bikes
        for bike in range(bikes_in_circulation):
            self.stations[random.randint(0, stations-1)].add_bike()

    def step(self, action):
        # For the purposes of this environment, a step will simulate one hour of time,
        # with 12 hours per 'day'

        # our action needs to be a incentive vector, over all of the stations

        # for each hour, we need to generate users to take bikes
        numb_of_users = random.randint(1,self.max_hourly_customers)

        bikes_moving = 0
        failed_transactions = 0
        self.hour += 1

        # iterate through users
        for user in range(numb_of_users):

            # first, pick random station to send to
            start_station_location = random.randint(0,len(self.stations)-1)
            sent_station = self.stations[start_station_location]

            # check if bikes are at this station
            if sent_station.has_bikes():
                # Take a bike, customer satisfied.
                sent_station.take_bike()
                bikes_moving += 1

            else:
                # check the incentive region.  If both nearby regions have negative incentives, then customer
                # unsatisfied
                if start_station_location is 0:
                    if not self.stations[1].has_bikes():
                        # only check to the right
                        if action[1] > 0:
                            # do the move
                            self.stations[1].take_bike()
                            bikes_moving += 1

                        else:
                            failed_transactions += 1
                    else:
                        failed_transactions += 1

                elif start_station_location is (len(self.stations)-1):
                    if self.stations[(len(self.stations)-2)].has_bikes():
                        # only check to the left
                        if action[len(self.stations)-2] > 0:
                            # do the move
                            self.stations[len(self.stations)-2].take_bike()
                            bikes_moving += 1
                        else:
                            failed_transactions += 1
                    else:
                        failed_transactions += 1

                else:
                    if action[start_station_location-1] <= 0 and action[start_station_location+1] <= 0:
                        # fail
                        failed_transactions += 1
                    elif action[start_station_location-1] >= action[start_station_location+1] and self.stations[start_station_location-1].has_bikes():
                        # move to the left
                        self.stations[start_station_location-1].take_bike()
                        bikes_moving += 1
                    elif action[start_station_location+1] >= action[start_station_location-1] and self.stations[start_station_location+1].has_bikes():
                        # move to the right
                        self.stations[start_station_location+1].take_bike()
                        bikes_moving += 1

        hourly_success = (numb_of_users - failed_transactions) / (numb_of_users + 0.0)
        obser = self.get_state()
        if self.hour >= 12:
            done = True
        else:
            done = False

        self.temp_day_reward += hourly_success

        return np.array(obser), hourly_success, done, {"info": "yo"}

    def get_state(self):
        bikes = []
        for st in self.stations:
            bikes.append(st.get_bikes())
        return bikes

    def reset(self):

        self.data_file.write((str(self.temp_day_reward)+ "\n"))
        self.temp_day_reward = 0

        self.hour = 0
        for st in self.stations:
            st.clear()

        # randomize bikes
        for bike in range(self.bikes_in_circulation):
            self.stations[random.randint(0, len(self.stations)-1)].add_bike()

        return np.array(self.get_state())

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True


# Create environment
print("env")
env = BikesEnv(5, 15, 15, "PPO1-01")

# Instantiate the agent
print("model")
model = PPO1('MlpPolicy', env, verbose=1)

# Train the agent
print("learn")
timesteps = int(2e5)
model.learn(total_timesteps=timesteps)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(mean_reward)

