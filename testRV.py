#file to test the generation of rvs

import random
import math
import statistics
import matplotlib.pyplot as plt
import scipy.stats as scipy
from collections import Counter
import generateRV as gen
from scipy.stats import pearsonr 


def testExpo():
	####Expo###
	#lambda - how many events per unit time
	rate = 5
	#how many Expos to generate
	n = 500
	#list of Expos
	expos = [gen.generateExpo(rate) for i in range(n)]
	plot(expos,100)
	popMean = popStdev = 1/rate
	printStats(expos,popMean, popStdev)
	

def testGamma():
	###Gamma###
	#alpha - shape parameter
	alpha = 2
	#lambda - rate parameter
	rate = 5
	#how many Gammas to generate
	n = 500
	#list of Gammas
	gammas = [gen.generateGamma(alpha, rate) for i in range(n)]
	plot(gammas, 100)
	popMean = alpha/rate
	popStdev = math.sqrt(alpha)/rate
	printStats(gammas,popMean,popStdev)
	
def testPoisson():
	###Poisson###
	#lambda - rate parameter
	rate = 5
	#how many Poissons to generate
	n = 500
	#list of Poissons
	poissons = [gen.generatePoisson(rate) for i in range(n)]
	plot(poissons, 100)
	popMean = rate
	popStdev = math.sqrt(rate)
	printStats(poissons, popMean, popStdev)

def testChiSquare():
	###Chi-Square###
	#dof - degrees of freedom
	dof = 15
	#how many Chi-Squares to generate
	n = 500
	#list of Chi-Squares
	chiSquares = [gen.generateChiSquare(dof) for i in range(n)]
	plot(chiSquares, 100)
	popMean = dof
	popStdev = math.sqrt(2*dof)
	printStats(chiSquares, popMean, popStdev)

def testInvChiSquare():
	###Inverse Chi-Square###
	#dof - degrees of freedom
	dof = 15
	#how many inv Chi-Squares to generate
	n = 500
	#list of inv Chi-Squares
	invChiSquares = [gen.generateInvChiSquare(dof) for i in range(n)]
	plot(invChiSquares, 100)
	popMean = 1/(dof-2)
	popStdev = math.sqrt(2/(((dof-2)**2)*(dof-4)))
	printStats(invChiSquares, popMean, popStdev)

def testScaleInvChiSquare():
	###Scaled Inverse Chi-Square###
	#dof - degrees of freedom
	dof = 15
	#scale2 - tau^2 - scales the distribution horizontally and vertically
	scale2 = 3
	#how many scaled inverse Chi-Squares to generate
	n = 500
	#list of scaled inverse Chi-Squares
	scaleInvChiSquares = [gen.generateScaleInvChiSquare(dof,scale2) for i in range(n)]
	plot(scaleInvChiSquares, 100)
	popMean = (dof*scale2) / (dof-2)
	popStdev = math.sqrt(2*(dof**2)*(scale2**2)/(((dof-2)**2)*(dof-4)))
	printStats(scaleInvChiSquares, popMean, popStdev)

def testT():
	###Student's t distribution###
	#dof - degrees of freedom
	dof = 15
	#how many t's to generate
	n = 500
	#list of t's
	ts = [gen.generateT(dof) for i in range(n)]
	plot(ts, 100)
	popMean = 0
	popStdev = math.sqrt(dof/(dof-2))
	printStats(ts, popMean, popStdev)


def testBeta():
	###Beta###
	#alpha - observed successes
	alpha = 1
	#beta - observed failures
	beta = 1
	#how many Betas to generate
	n = 10000
	#list of Betas
	betas = [gen.generateBeta(alpha,beta) for i in range(n)]
	plot(betas,100)
	popMean = alpha/(alpha+beta)

	popStdev = math.sqrt((alpha*beta)/(((alpha+beta)**2)*(alpha+beta+1)))
	printStats(betas,popMean,popStdev)

def testBoxMuller():
	###Normal###
	#box muller
	#mu - the mean
	u = 5
	#variance - the variance or sigma squared
	variance = 2
	#how many Normals to generate
	n = 1000
	#list of Normals
	normals = []
	for i in range((n//2)+1):
		normals+=gen.boxMuller(u,variance)
	if len(normals) > n:
		normals.pop()
	plot(normals,100)
	popMean = u
	popStdev = math.sqrt(variance)
	printStats(normals,popMean,popStdev)

def testMetropolisHastings():
	###Normal###
	#metropolis hastings
	#mu - the mean
	u = 5
	#variance - the variance or sigma squared
	variance = 10
	#how many Normals to generate
	n = 1000000
	#list of Normals
	normals = gen.metropolisHastings(u,variance,n)
	plot(normals,100)
	popMean = u
	popStdev = math.sqrt(variance)
	printStats(normals,popMean,popStdev)

def testCorrelatedNormals():

	#test using a covariance matrix
	u = [3,6,1]
	cov = [[1,.63, .4],[.63,1,.35],[.4,.35,1]]

	n = 10000
	ex1 = ex2 = ex3 = 0
	ex1x2 = ex1x3 = ex2x3 = 0

	for i in range(n):
		x1, x2, x3 = gen.MVN(u,cov)
		ex1 += x1/n
		ex2 += x2/n
		ex3 += x3/n
		ex1x2 += x1*x2/n
		ex2x3 += x2*x3/n
		ex1x3 += x1*x3/n

	covx1x2 = ex1x2-(ex1*ex2)
	covx1x3 = ex1x3-(ex1*ex3)
	covx2x3 = ex2x3-(ex2*ex3)

	print("Cov(X1,X2): Sample = %.4f; Expected = %.4f" % (covx1x2, cov[0][1]))
	print("Cov(X1,X3): Sample = %.4f; Expected = %.4f" % (covx1x3, cov[0][2]))
	print("Cov(X2,X3): Sample = %.4f; Expected = %.4f" % (covx2x3, cov[1][2]))
	print()

	#test using a correlation matrix
	ex1 = ex2 = ex3 = 0
	ex1x2 = ex1x3 = ex2x3 = 0
	u = [3,6,1]
	corr = [[1,.63, .4],[.63,1,.35],[.4,.35,1]]
	var = [5,8,4]
	cov = gen.correlationToCovariance(corr,var)
	for i in range(n):
		x1, x2, x3 = gen.MVN(u,cov)
		ex1 += x1/n
		ex2 += x2/n
		ex3 += x3/n
		ex1x2 += x1*x2/n
		ex2x3 += x2*x3/n
		ex1x3 += x1*x3/n

	covx1x2 = ex1x2-(ex1*ex2)
	covx1x3 = ex1x3-(ex1*ex3)
	covx2x3 = ex2x3-(ex2*ex3)

	print("Cov(X1,X2): Sample = %.4f; Expected = %.4f" % (covx1x2, cov[0][1]))
	print("Cov(X1,X3): Sample = %.4f; Expected = %.4f" % (covx1x3, cov[0][2]))
	print("Cov(X2,X3): Sample = %.4f; Expected = %.4f" % (covx2x3, cov[1][2]))




def testCorr():
	cor = [[1,.63, .4],[.63,1,.35],[.4,.35,1]]
	var = [3,7,2]

	print(gen.correlationToCovariance(cor,var))




def plot(x, bins=10):
	#plots a historgram of random variables
	plt.hist(x,bins=bins)
	plt.show()

def plotCDF(x):
	#plots a line graph of the cdf of dataset x
	y = [i/len(x) for i in range(len(x))]
	plt.plot(x,y)
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

def main():
	testCorrelatedNormals()

if __name__ == "__main__":
	main()