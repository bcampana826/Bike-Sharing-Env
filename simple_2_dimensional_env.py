import gym
from pandas import np

import seed_generation


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

class TwoDBikesEnv(gym.Env):

    def __init__(self, seed_number, model_name):

        if seed_number is -1:
            self.seed_file = open(("random-seed-generations/" + str(seed_generation.generate_simple_2d_env_with_work( 600, 60, 100 )) + ".txt"), "r")
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

        self.day_successful_trips = 0
        self.day_trips = 0

        # Saving Trips
        self.trips = []
        for hour in range(12):
            self.trips.append(
                run_info[hour + 3][run_info[hour + 3].index(";") + 1:run_info[hour + 3].rindex(";")].split())
            self.day_trips += len(self.trips[hour])
            for trip in range(len(self.trips[hour])):
                self.trips[hour][trip] = BikeTrip(self.trips[hour][trip])

        # Now with all the seed data, setup env.
        self.model_name = model_name
        self.env_data = open(("2-dim-results/" + model_name + ".txt"), "w")

        self.number_of_stations = self.params[0]
        self.daily_budget = self.params[1]
        self.bikes_in_circulation = self.params[2]
        self.max_hourly_customers = self.params[3]
        self.hour = 0
        self.temp_day_reward = 0
        self.temp_day_budget = self.daily_budget

        self.action_space = gym.spaces.MultiDiscrete([15] * self.number_of_stations)
        self.observation_space = gym.spaces.Box(np.array([0] * (self.number_of_stations + 2)),
                                                np.array(([11]+[self.bikes_in_circulation] * self.number_of_stations) + [self.daily_budget]))

        self.stations = []
        for station in range(self.number_of_stations):
            self.stations.append(Station(self.starting_stations[station]))

    def reset(self):

        data = float(self.day_successful_trips) / float(self.day_trips)

        self.env_data.write((str(data) + "\n"))
        self.temp_day_reward = 0
        self.temp_day_budget = self.daily_budget

        self.day_successful_trips = 0

        self.hour = 0
        for st in range(len(self.stations)):
            self.stations[st].set_bikes(self.starting_stations[st])

        return np.array(self.get_state())

    def step(self, action):

        trips = self.trips[self.hour]
        failed_transactions = 0

        successful_trips = []

        # USE FOR TESTING BASELINE
        # action = [0]*16


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

        self.day_successful_trips += len(successful_trips)

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
        adj, crn = self.get_adjacent_stations(start_station_location)

        # First, check if the station sent to has bikes
        if self.stations[start_station_location].has_bikes():
            # Complete move
            self.stations[start_station_location].take_bike()
            return True

        possible_adj_stations = []
        possible_crn_stations = []

        for sts in range(len(adj)):
            if self.stations[adj[sts]].has_bikes() and 5 <= action[adj[sts]] <= self.temp_day_budget:
                possible_adj_stations.append(adj[sts])

        for sts in range(len(crn)):
            if self.stations[crn[sts]].has_bikes() and 10 <= action[crn[sts]] <= self.temp_day_budget:
                possible_crn_stations.append(crn[sts])

        if len(possible_adj_stations) > 0:
            chosen = -1
            temp_incentive = -1
            for sts in range(len(possible_adj_stations)):
                if action[possible_adj_stations[sts]] > temp_incentive:
                    chosen = possible_adj_stations[sts]
                    temp_incentive = action[possible_adj_stations[sts]]

            if chosen is -1:
                return False

            self.stations[chosen].take_bike()
            self.temp_day_budget - action[possible_adj_stations[sts]]
            return True

        elif len(possible_crn_stations) > 0:
            chosen = -1
            temp_incentive = -1
            for sts in range(len(possible_crn_stations)):
                if action[possible_crn_stations[sts]] > temp_incentive:
                    chosen = possible_crn_stations[sts]
                    temp_incentive = action[possible_crn_stations[sts]]

            if chosen is -1:
                return False

            self.stations[chosen].take_bike()
            self.temp_day_budget - action[possible_crn_stations[sts]]
            return True

        return False

    def get_adjacent_stations(self, start_station_location):
        adj_stations = []
        corner_stations = []

        if start_station_location % 4 is 0:
            adj_stations.append(start_station_location + 1)

        elif (start_station_location - 3) % 4 is 0:
            adj_stations.append(start_station_location - 1)
        else:
            adj_stations.append(start_station_location + 1)
            adj_stations.append(start_station_location - 1)

        if start_station_location - 3 <= 0:
            adj_stations.append(start_station_location + 4)
        elif start_station_location + 3 >= 15:
            adj_stations.append(start_station_location - 4)
        else:
            adj_stations.append(start_station_location + 4)
            adj_stations.append(start_station_location - 4)

        for sts in range(len(adj_stations)):
            if adj_stations[sts] + 4 is start_station_location or adj_stations[sts] - 4 is start_station_location:
                if adj_stations.count(start_station_location + 1) == 1:
                    corner_stations.append(adj_stations[sts] + 1)
                if adj_stations.count(start_station_location - 1) == 1:
                    corner_stations.append(adj_stations[sts] - 1)

        return adj_stations, corner_stations


