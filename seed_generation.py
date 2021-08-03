import os
import datetime
import random


def generate_true_random_seed(number_of_stations, daily_budget, bikes_in_circulation, max_hourly_customers):
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


def generate_working_prediction_seed( daily_budget, bikes_in_circulation, max_hourly_customers):
    # statically set stations at 10 for this implementation

    # Data Saving Setup
    seed_number = len(os.listdir('random-seed-generations')) + 1
    data_file = open(("random-seed-generations/" + str(seed_number) + ".txt"), "w")

    data_file.write(
        "1-Dimensional Bike Env Seed *WORK PREDICTION* " + str(seed_number) + " generated at " + str(
            datetime.datetime.now()) + "\n")
    data_file.write(str(10) + "," + str(daily_budget) + "," + str(bikes_in_circulation) + "," + str(
        max_hourly_customers) + "\n")

    # Generate Bike Start Locations
    bike_locations = [0] * 10
    for bike in range(bikes_in_circulation):
        bike_locations[random.randint(0, len(bike_locations) - 1)] += 1

    data_file.write(str(bike_locations) + "\n")

    home_stations = [0,1,2]
    work_stations = [7,8,9]

    for hour in range(12):
        # attempting to simulate work travel.
        data_file.write("Hour: " + str(hour) + "; ")

        if hour < 3:
            # 60 percent chance start from a home, end at a work
            # 20 percent  chance start from random, end at a work
            # 20 percent chance for random
            # extra people at this time
            number_of_trips = random.randint(int(max_hourly_customers/2), max_hourly_customers)

            for trip in range(number_of_trips):
                percent = random.randint(1, 10)
                if percent <= 6:
                    data_file.write("(" + str(random.choice(home_stations)) + "," + str(
                        random.choice(work_stations)) + ") ")
                elif percent <= 8:
                    data_file.write("(" + str(random.randint(0, 10 - 1)) + "," + str(
                        random.choice(work_stations)) + ") ")
                else:
                    data_file.write("(" + str(random.randint(0, 10 - 1)) + "," + str(
                        random.randint(0, 10 - 1)) + ") ")
        elif hour >= 9:
            # 60 percent chance start from a work, end at a home
            # 20 percent  chance start from random, end at a home
            # 20 percent chance for random
            # extra people at this time
            number_of_trips = random.randint(int(max_hourly_customers / 2), max_hourly_customers)

            for trip in range(number_of_trips):
                percent = random.randint(1, 10)
                if percent <= 6:
                    data_file.write("(" + str(random.choice(work_stations)) + "," + str(
                        random.choice(home_stations)) + ") ")
                elif percent <= 8:
                    data_file.write("(" + str(random.randint(0, 10 - 1)) + "," + str(
                        random.choice(home_stations)) + ") ")
                else:
                    data_file.write("(" + str(random.randint(0, 10 - 1)) + "," + str(
                        random.randint(0, 10 - 1)) + ") ")
        else:
            # Generate Trips in this Hour
            number_of_trips = random.randint(1, int(max_hourly_customers/2))

            for trip in range(number_of_trips):
                data_file.write("(" + str(random.randint(0, 10 - 1)) + "," + str(
                    random.randint(0, 10 - 1)) + ") ")

        data_file.write(";" + "\n")

    data_file.close()

    return seed_number

def generate_simple_2d_env_with_work( daily_budget, bikes_in_circulation, max_hourly_customers ):
    # statically create a 4 by 4
    # top 3 stations are home stations
    # bottom 3 stations are work stations
    home_stations = [0,1,2,4,5,8]
    work_stations = [15,14,11,13,10,7]
    stations = 16

    # Data Saving Setup
    seed_number = len(os.listdir('random-seed-generations')) + 1
    data_file = open(("random-seed-generations/" + str(seed_number) + ".txt"), "w")

    data_file.write(
        "2-Dimensional Bike Env Seed *WORK PREDICTION* " + str(seed_number) + " generated at " + str(
            datetime.datetime.now()) + "\n")
    data_file.write(str(stations) + "," + str(daily_budget) + "," + str(bikes_in_circulation) + "," + str(
        max_hourly_customers) + "\n")

    # Generate Bike Start Locations
    bike_locations = [0] * 16
    for bike in range(bikes_in_circulation):
        bike_locations[random.randint(0, len(bike_locations) - 1)] += 1

    data_file.write(str(bike_locations) + "\n")

    for hour in range(12):
        # attempting to simulate work travel.
        data_file.write("Hour: " + str(hour) + "; ")

        if hour < 3:
            # 60 percent chance start from a home, end at a work
            # 20 percent  chance start from random, end at a work
            # 20 percent chance for random
            # extra people at this time
            number_of_trips = random.randint(int(max_hourly_customers/2), max_hourly_customers)

            for trip in range(number_of_trips):
                percent = random.randint(1, 10)
                if percent <= 6:
                    data_file.write("(" + str(random.choice(home_stations)) + "," + str(
                        random.choice(work_stations)) + ") ")
                elif percent <= 8:
                    data_file.write("(" + str(random.randint(0, stations - 1)) + "," + str(
                        random.choice(work_stations)) + ") ")
                else:
                    data_file.write("(" + str(random.randint(0, stations - 1)) + "," + str(
                        random.randint(0, stations - 1)) + ") ")
        elif hour >= 9:
            # 60 percent chance start from a work, end at a home
            # 20 percent  chance start from random, end at a home
            # 20 percent chance for random
            # extra people at this time
            number_of_trips = random.randint(int(max_hourly_customers / 2), max_hourly_customers)

            for trip in range(number_of_trips):
                percent = random.randint(1, 10)
                if percent <= 6:
                    data_file.write("(" + str(random.choice(work_stations)) + "," + str(
                        random.choice(home_stations)) + ") ")
                elif percent <= 8:
                    data_file.write("(" + str(random.randint(0, stations - 1)) + "," + str(
                        random.choice(home_stations)) + ") ")
                else:
                    data_file.write("(" + str(random.randint(0, stations - 1)) + "," + str(
                        random.randint(0, 10 - 1)) + ") ")
        else:
            # Generate Trips in this Hour
            number_of_trips = random.randint(1, int(max_hourly_customers/2))

            for trip in range(number_of_trips):
                data_file.write("(" + str(random.randint(0, stations - 1)) + "," + str(
                    random.randint(0, 10 - 1)) + ") ")

        data_file.write(";" + "\n")

    data_file.close()

    return seed_number

generate_simple_2d_env_with_work( 600, 60, 100 )