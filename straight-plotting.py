import matplotlib.pyplot as plt

trpo_file = open("1-dim-results/TRPO-Percents.txt")
trpo_file = [float(n) for n in trpo_file]
plt.plot(trpo_file, color='blue')
plt.title("TRPO")
plt.show()

ppo2_file = open("1-dim-results/PPO2-Percents.txt")
ppo2_file = [float(n) for n in ppo2_file]
plt.plot(ppo2_file, color='orange')
plt.title("PPO2")
plt.show()

acktr_file = open("1-dim-results/ACKTR-Percents.txt")
acktr_file = [float(n) for n in acktr_file]
plt.plot(acktr_file, color='red')
plt.title("ACKTR")
plt.show()

a2c_file = open("1-dim-results/A2C-Percents.txt")
a2c_file = [float(n) for n in a2c_file]
plt.plot(a2c_file, color='purple')
plt.title("a2c")
plt.show()


