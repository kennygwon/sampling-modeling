#file to generate samples using generateRV.py
import random
import math
import statistics
import matplotlib.pyplot as plt
import scipy.stats as scipy
from collections import Counter
import generateRV as gen
from scipy.stats import norm
from collections import defaultdict

def plot(x, bins=10):
	#plots a historgram of exponential random variables
	plt.hist(x,bins=bins)
	plt.show()

def printStats(x, popMean, popStdev):
	#prints the sample mean and std
	#also prints the population mean and std

	print("sample mean: ", round(statistics.mean(x),3))
	print("sample stdev: ", round(statistics.pstdev(x),3))
	print("population mean: ", round(popMean,3))
	print("population stdev: ", round(popStdev,3))
	zScore = round((statistics.mean(x) - popMean)/popStdev,3)
	print("z-score: ", zScore)
	print("p-value: ",round(2*scipy.norm.cdf(-abs(zScore)),3))

def confInterval(y,alpha):
	#creates a confidence interval based on the sampled data
	#y is the sampled data and alpha determines the size of the conf interval
	#ex. y = [0.8,0.6,0.7,...] alpha=.05
	y.sort()
	n = len(y)
	print("Median: ", y[n//2])
	print(100-(alpha*100), "% Interval: (", y[round(n*(alpha/2))], ", ", y[round(n - (n*(alpha/2)))], ")")

def roundedLikelihood(data,u,sd):

	likelihood = -2*math.log(sd)
	for yi in data:
		if not norm.cdf(yi+0.5,u,sd) - norm.cdf(yi-0.5,u,sd):
			return 0
		else:
			likelihood += math.log(norm.cdf(yi+0.5,u,sd) - norm.cdf(yi-0.5,u,sd))
	return math.e**likelihood


def main():


	




main()