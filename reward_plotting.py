import matplotlib.pyplot as plt

trpo_file = open("1-dim-results/TRPO-Work.txt")
trpo_file = [float(n) for n in trpo_file]
print(trpo_file[1])
trpo_plot = []
count = 0
temp_reward = 0
for n in range(len(trpo_file)):
    count += 1

    temp_reward += float(trpo_file[n])

    if count % 10 == 0:
        trpo_plot.append((temp_reward/10.0))
        temp_reward = 0

PPO2_file = open("1-dim-results/PPO2-Work.txt")
PPO2_file = [float(n) for n in PPO2_file]
print(PPO2_file[1])
PPO2_plot = []
count = 0
temp_reward = 0
for n in range(len(PPO2_file)):
    count += 1

    temp_reward += float(PPO2_file[n])

    if count % 10 == 0:
        PPO2_plot.append((temp_reward/10.0))
        temp_reward = 0

# PPO1_file = open("1-dim-results/PPO1.txt")
# PPO1_file = [float(n) for n in PPO1_file]
# print(PPO1_file[1])
# PPO1_plot = []
# count = 0
# temp_reward = 0
# for n in range(len(PPO1_file)):
#     count += 1
#
#     temp_reward += float(PPO1_file[n])
#
#     if count % 10 == 0:
#         PPO1_plot.append((temp_reward/10.0))
#         temp_reward = 0

ACKTR_file = open("1-dim-results/ACKTR-Work.txt")
ACKTR_file = [float(n) for n in ACKTR_file]
print(ACKTR_file[1])
ACKTR_plot = []
count = 0
temp_reward = 0
for n in range(len(ACKTR_file)):
    count += 1

    temp_reward += float(ACKTR_file[n])

    if count % 10 == 0:
        ACKTR_plot.append((temp_reward/10.0))
        temp_reward = 0

A2C_file = open("1-dim-results/A2C-Work.txt")
A2C_file = [float(n) for n in A2C_file]
print(A2C_file[1])
A2C_plot = []
count = 0
temp_reward = 0
for n in range(len(A2C_file)):
    count += 1

    temp_reward += float(A2C_file[n])

    if count % 10 == 0:
        A2C_plot.append((temp_reward/10.0))
        temp_reward = 0

plt.plot(trpo_plot, color='blue',label='TRPO')
plt.plot(PPO2_plot, color='red',label='PPO2')
#plt.plot(PPO1_plot, color='green',label='PPO1')
plt.plot(A2C_plot, color='purple',label='A2C')
plt.plot(ACKTR_plot, color='Orange',label='ACKTR')
plt.plot([8.8]*1600, color='black', label='Base')
plt.legend()
plt.title("Incentive Optimization")
plt.show()

