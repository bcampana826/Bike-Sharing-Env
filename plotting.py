import matplotlib.pyplot as plt
import numpy as np


def get_array_with_length(array):
    x_axis = []

    for i in range(len(array)):
        x_axis.append(i)

    return x_axis

def get_line(m, b, x_axis):
    for i in range(len(x_axis)):
        x_axis[i] = (1*m)+b

    return x_axis

def get_data(file):
    my_file = open(file)
    my_data = my_file.readlines()

    for i in range(len(my_data)):
        my_data[i] = float(my_data[i])

    x_axis = get_array_with_length(my_data)
    m, b = np.polyfit(x_axis, my_data, 1)


    return get_line(m, b, get_array_with_length(my_data)), x_axis


trpo_y, trpo_x = get_data("1-dim-results/TRPO.txt")
acktr_y, acktr_x = get_data("1-dim-results/ACKTR.txt")
ppo2_y, ppo2_x = get_data("1-dim-results/PPO2.txt")
a2c_y, a2c_x = get_data("1-dim-results/A2C.txt")
ppo1_y, ppo1_x = get_data("1-dim-results/PPO1.txt")

plt.plot(trpo_x, trpo_y, label='TRPO')
plt.plot(ppo2_x, ppo2_y, label='PPO2')
plt.plot(ppo1_x, ppo1_y, label='PPO1')
plt.plot(acktr_x, acktr_y, label='ACKTR')
plt.plot(a2c_x, a2c_y, label='A2C')
plt.axline((0, 10.28), 0)
plt.ylabel('reward')
plt.ylim(10, 12.0)
plt.legend()
plt.show()
