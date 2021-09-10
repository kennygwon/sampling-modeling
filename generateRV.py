#generates a bunch of rvs using the inverse tranfsorm method

import random
import math
import statistics
import matplotlib.pyplot as plt
import scipy.stats as scipy
from collections import Counter
import numpy as np

def generateExpo(rate):
	#returns a single instance of a Expo(rate)

	#random number using unif(0,1)
	u = random.uniform(0,1)

	#uses inverse transformation to generate an expo(lambda)
	x = math.log(u)/(-rate)

	return x

def generateGamma(alpha, rate):
	#returns a single instance of a Gamma(alpha, lambda) by generating expos

	ret = 0
	for i in range(alpha):
		ret += generateExpo(rate)
	return ret

def generateBeta(alpha,beta):
	#returns a single instance of a Beta(alpha,beta) by generating gammas

	#lambda to be used to generate Gamma
	#lambda can be chosen arbitrarily as long as it's the same for both gammas
	rate = 2

	x = generateGamma(alpha,rate)
	y = generateGamma(beta,rate)

	return x/(x+y)

def generatePoisson(rate):
	#returns a single instance of a Gamma(lambda) by generating expos

	#generate expos which represent interarrival times
	#until our we reach one unit of time
	ret = -1
	waitingTime = 0
	while waitingTime < 1:
		ret += 1
		waitingTime += generateExpo(rate)

	return ret

def generateChiSquare(dof):
	#returns a single instance of a Chi-Square(degrees of freedom) by generating Normals

	#generate pairs of Normals using boxMuller and multiply them together
	#do this dof number of times
	ret = 0
	#generates two randoms at once
	for i in range(dof//2):
		x,y = boxMuller(0,1)
		ret += x**2 + y**2
	#generates the last random if dof is odd
	if dof//2:
		x,_ = boxMuller(0,1)
		ret += x**2

	return ret

def generateInvChiSquare(dof):
	#returns a single instance of an inverse ChiSquare(dof) by generating a Chi-Square
	#if X ~ Chi-Square(dof) then Y=1/X ~ Inv-Chi-Square(dof)

	return 1/generateChiSquare(dof)

def generateScaleInvChiSquare(dof, scale2):
	#returns a single instance of a scaled inverse ChiSquare(dof, scale^2) by generating an inverse Chi-Square
	#if X~InvChi-Square(v) then Y=Xt^2v~ScaleInvChi-Square(v,t^2) where v is the dof

	return dof*scale2*generateInvChiSquare(dof)

def generateStandardT(dof):
	#generates a single instance of the t-distribution(dof) by generating a std. normal and a chi-square
	#t = z/sqrt(v/n) where z is standard normal and v is chi-square with n dof

	z = boxMuller(0,1)[0]
	v = generateChiSquare(dof)
	return z/(math.sqrt(v/dof))

def generateT(u,variance,dof):

	t = generateStandardT(dof)
	return (t*math.sqrt(variance)/math.sqrt(dof+1))+u


def boxMuller(u,variance):
	#returns two independent Normals(u,variance) by using the Box Muller method

	u1 = random.uniform(0,2*math.pi)
	#u2 = random.uniform(0,1)
	#r = sqrt(-2*math.log(u2))
	r = math.sqrt(2*generateExpo(1))

	n1 = r*math.cos(u1)
	n2 = r*math.sin(u1)
	x = u + (n1*math.sqrt(variance))
	y = u + (n2*math.sqrt(variance))
	return [x,y]

def metropolisHastings(mu,variance,n):
	#uses MCMC metropolis hastings to generate Normals(u,variance)

	# y = x + e
	# e~Unif(-delta,delta) where delta is a small value
	delta = .5

	normals = [mu]
	while len(normals)<n:
		x = normals[-1]
		y = random.uniform(-delta,delta) + x
		a = min((normalPDF(y,mu,variance) / normalPDF(x,mu,variance)),1)
		u = random.uniform(0,1)
		if u <= a:
			normals.append(y)
		else:
			normals.append(x)
	return normals



def normalPDF(x,u,variance):
	#returns the pdf at some point x with distribution N(u,variance)
	return ((1/(math.sqrt(2*math.pi*variance)))*math.exp((-((x-u)**2))/(2*variance)))
	
	#unnormalized pdf (to test Metropolis Hastings)
	#return (math.exp((-((x-u)**2))/(2*variance)))


def MVN(u,cov):
	#returns an n*1 list of MVN normals given an 1*n list of means and n*n covariance matrix
	n = len(u)

	#creates a list of standard normals N(0,1) of size n
	z = []
	while len(z) < n:
		z += boxMuller(0,1)
	if len(z) > n:
		z.pop()

	#turns input arrays into numpy arrays
	u = np.array([u])
	z = np.array([z])
	cov = np.array(cov)

	#perform cholesky decomposition
	lower = np.linalg.cholesky(cov)
	upper = np.transpose(lower)

	#X=u+(R^T)Z
	#since I used 1*n u and x arrays
	#formula becomes X = u+Z(R)
	#where R is the upper traingular matrix in LU decomposition
	x = np.add(u, np.matmul(z,upper))

	#converts numpy array back to array and returns
	x = x[0].tolist()
	return x

def correlationToCovariance(corr, var):
	#converts a correlation matrix to a covariance matrix
	#covariance matrix can then be used to generate MVN's

	cov = []

	n = len(var)
	for i in range(n):
		covLine = []
		for j in range(n):
			if i==j:
				covLine.append(var[i])
			else:
				covLine.append(corr[i][j]*math.sqrt(var[i]*var[j]))
		cov.append(covLine)

	return cov
