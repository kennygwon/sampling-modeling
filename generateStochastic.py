import random
from collections import Counter
import math
import matplotlib.pyplot as plt
import testRV as testRV
import generateRV as gen
import testRV as testRV
from scipy.stats import norm
import random


def generateBM(T,u,sigma):

	dt = .01
	variance = sigma**2

	t = [0]
	x_t = [0]
	ex_t = [0]

	for i in range(int(T/dt)):
		t.append(t[-1]+dt)
		deterministic = u*dt
		stochastic = gen.boxMuller(0,variance*dt)[0]
		x_t.append(x_t[-1] + deterministic + stochastic)
		ex_t.append(ex_t[-1] + deterministic)

	plt.plot(t,x_t)
	plt.plot(t,ex_t)
	plt.show()
	




def main():

	generateBM(100,-5,50)


main()