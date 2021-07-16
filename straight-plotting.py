import matplotlib.pyplot as plt

trpo_file = open("1-dim-results/TRPO.txt")
trpo_file = [float(n) for n in trpo_file]
plt.plot(trpo_file, color='blue')
plt.title("TRPO")
plt.show()

ppo2_file = open("1-dim-results/PPO2-Work.txt")
ppo2_file = [float(n) for n in ppo2_file]
plt.plot(ppo2_file, color='orange')
plt.title("PPO2")
plt.show()

ppo1_file = open("1-dim-results/PPO1.txt")
ppo1_file = [float(n) for n in ppo1_file]
plt.plot(ppo1_file, color='green')
plt.title("PPO1")
plt.show()

acktr_file = open("1-dim-results/ACKTR.txt")
acktr_file = [float(n) for n in acktr_file]
plt.plot(acktr_file, color='red')
plt.title("ACKTR")
plt.show()

a2c_file = open("1-dim-results/A2C.txt")
a2c_file = [float(n) for n in a2c_file]
plt.plot(a2c_file, color='purple')
plt.title("a2c")
plt.show()


