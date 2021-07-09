import datetime
import os
import random

import gym
import numpy as np
from gym.vector.utils import spaces


class Station:

    def __init__(self, number_of_bikes):
        self.bikes = number_of_bikes

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


class BikeTrip():
    def __init__(self, trip_string):
        self.locations = trip_string.strip("(").strip(")").split(',')
        self.locations[0] = int(self.locations[0])
        self.locations[1] = int(self.locations[1])

    def get_start_location(self):
        return self.locations[0]

    def get_end_location(self):
        return self.locations[1]


class BikesEnv(gym.Env):

    def __init__(self, seed_number, model_name):

        if seed_number is -1:
            self.seed_file = open(("random-seed-generations/" + str(BikesEnv.generator_seed()) + ".txt"), "r")
        else:
            self.seed_file = open(("random-seed-generations/" + str(seed_number) + ".txt"), "r")

        run_info = self.seed_file.readlines()

        # Saving Params
        self.params = run_info[1].split(",")
        for value in range(len(self.params)):
            self.params[value] = int(self.params[value].strip())

        self.starting_stations = run_info[2].split(",")
        for value in range(len(self.starting_stations)):
            self.starting_stations[value] = int(self.starting_stations[value].strip().strip('[').strip(']'))

        # Saving Trips
        self.trips = []
        for hour in range(12):
            self.trips.append(
                run_info[hour + 3][run_info[hour + 3].index(";") + 1:run_info[hour + 3].rindex(";")].split())
            for trip in range(len(self.trips[hour])):
                self.trips[hour][trip] = BikeTrip(self.trips[hour][trip])

        # Now with all the seed data, setup env.
        self.model_name = model_name
        self.env_data = open(("1-dim-results/" + model_name + ".txt"), "w")

        self.number_of_stations = self.params[0]
        self.daily_budget = self.params[1]
        self.bikes_in_circulation = self.params[2]
        self.max_hourly_customers = self.params[3]
        self.hour = 0
        self.temp_day_reward = 0
        self.temp_day_budget = self.daily_budget

        self.action_space = spaces.MultiDiscrete([15] * self.number_of_stations)
        self.observation_space = spaces.Box(np.array([0] * (self.number_of_stations + 2)),
                                            np.array(([11]+[self.bikes_in_circulation] * self.number_of_stations) + [self.daily_budget]))

        self.stations = []
        for station in range(self.number_of_stations):
            self.stations.append(Station(self.starting_stations[station]))

    @classmethod
    def generator_seed(cls, number_of_stations, daily_budget, bikes_in_circulation, max_hourly_customers):

        # Data Saving Setup
        seed_number = len(os.listdir('random-seed-generations')) + 1
        data_file = open(("random-seed-generations/" + str(seed_number) + ".txt"), "w")

        data_file.write(
            "1-Dimensional Bike Env Seed " + str(seed_number) + " generated at " + str(datetime.datetime.now()) + "\n")
        data_file.write(str(number_of_stations) + "," + str(daily_budget) + "," + str(bikes_in_circulation) + "," + str(
            max_hourly_customers) + "\n")

        # Generate Bike Start Locations
        bike_locations = [0] * number_of_stations
        for bike in range(bikes_in_circulation):
            bike_locations[random.randint(0, len(bike_locations) - 1)] += 1

        data_file.write(str(bike_locations) + "\n")

        # Generate Trips for all 12 Hours
        for hour in range(12):
            data_file.write("Hour: " + str(hour) + "; ")

            # Generate Trips in this Hour
            number_of_trips = random.randint(1, max_hourly_customers)

            for trip in range(number_of_trips):
                data_file.write("(" + str(random.randint(0, number_of_stations - 1)) + "," + str(
                    random.randint(0, number_of_stations - 1)) + ") ")

            data_file.write(";" + "\n")

        data_file.close()

        return seed_number

    def reset(self):

        self.env_data.write((str(self.temp_day_reward) + "\n"))
        self.temp_day_reward = 0
        self.temp_day_budget = self.daily_budget

        self.hour = 0
        for st in range(len(self.stations)):
            self.stations[st].set_bikes(self.starting_stations[st])

        return np.array(self.get_state())

    def step(self, action):

        trips = self.trips[self.hour]
        failed_transactions = 0

        successful_trips = []

        # Iterate through the trips
        for trip in range(len(trips)):

            success = self.complete_moving(action, trips[trip].get_start_location())

            if success:
                successful_trips.append(trips[trip])
            else:
                failed_transactions += 1

            hourly_success = (len(trips) - failed_transactions) / (len(trips) + 0.0)
            obser = self.get_state()

        self.temp_day_reward += hourly_success

        for finishing_trips in range(len(successful_trips)):
            self.stations[successful_trips[finishing_trips].get_end_location()].add_bike()

        if self.hour >= 11:
            done = True
        else:
            done = False

        self.hour += 1

        return np.array(obser), hourly_success, done, {"info": str(self.hour)}

    def get_state(self):
        bikes = [self.hour]
        for st in self.stations:
            bikes.append(st.get_bikes())
        bikes.append(self.temp_day_budget)

        return bikes

    def complete_moving(self, action, start_station_location):

        # First, check if the station sent to has bikes
        if self.stations[start_station_location].has_bikes():

            # Complete move
            self.stations[start_station_location].take_bike()
            return True

        # Check edge cases
        if start_station_location is 0:
            # Check for bikes, walk cost

            if self.stations[start_station_location + 1].has_bikes() and 5 <= action[start_station_location + 1] <= self.temp_day_budget:
                # Complete the Move
                self.stations[start_station_location + 1].take_bike()
                self.temp_day_budget -= action[start_station_location + 1]
                return True
            else:
                # Failure
                return False
        elif start_station_location is (len(self.stations) - 1):
            if self.stations[start_station_location - 1].has_bikes() and 5 <= action[start_station_location - 1] <= self.temp_day_budget:
                # Complete the Move
                self.stations[start_station_location - 1].take_bike()
                self.temp_day_budget -= action[start_station_location - 1]
                return True
            else:
                # Failure
                return False

        else:
            possible_stations = []
            if self.stations[start_station_location - 1].has_bikes():
                possible_stations.append(start_station_location - 1)
            if self.stations[start_station_location + 1].has_bikes():
                possible_stations.append(start_station_location + 1)

            if len(possible_stations) == 0:
                # Failure
                return False
            elif len(possible_stations) == 1:
                # Test
                if 5 <= action[possible_stations[0]] <= self.temp_day_budget:
                    # Complete
                    self.temp_day_budget -= action[possible_stations[0]]
                    return True
                else:
                    # Failure
                    return False
            else:
                if action[possible_stations[0]] >= action[possible_stations[1]] and 5 <= action[possible_stations[0]] <= self.temp_day_budget:
                    self.temp_day_budget -= action[possible_stations[0]]
                    return True
                elif 5 <= action[possible_stations[1]] <= self.temp_day_budget:
                    self.temp_day_budget -= action[possible_stations[1]]
                    return True
                else:
                    return False

