import random
import matplotlib.pyplot as plt
import generateRV as gen
import statistics
import numpy as np
import math
import random


def generateSimpleLRData():

	#randomly generate y-int
	b0 = random.uniform(-20,20)
	#randomly generate slope
	b1 = random.uniform(-4,4)
	#randomly generate the variance of y given x
	yVar = random.uniform(50,100)

	#print(b0,b1)

	xList = []
	yList = []
	for i in range(100):
		x = random.uniform(-20,20)
		expectedY = b1*x
		y = gen.boxMuller(expectedY,yVar)[0]
		xList.append(x)
		yList.append(y + b0)

	#plt.scatter(xList,yList)
	#plt.show()

	return xList, yList

def generateMultipleLRData():

	#randomly generates the population parameters
	b0 = random.uniform(-20,20)
	b1 = random.uniform(5,10)
	b2 = random.uniform(-10,0)
	b3 = random.uniform(-1,2)
	b4 = random.uniform(10,12)
	var1 = random.uniform(20,30)
	var2 = random.uniform(20,70)
	var3 = random.uniform(40,80)
	var4 = random.uniform(50,100)

	"""
	print(b0)
	print(b1)
	print(b2)
	print(b3)
	print(b4)
	"""

	xList = []
	yList = []
	for i in range(100):
		x1 = random.uniform(-20,20)
		x2 = random.uniform(-20,20)
		x3 = random.uniform(-20,20)
		x4 = random.uniform(-20,20)
		y = b0
		expectedY = b1*x1
		y += gen.boxMuller(expectedY, var1)[0]
		expectedY = b2*x2
		y += gen.boxMuller(expectedY, var2)[0]
		expectedY = b3*x3
		y += gen.boxMuller(expectedY, var3)[0]
		expectedY = b4*x4
		y += gen.boxMuller(expectedY, var4)[0]
		xList.append([1,x1,x2,x3,x4])
		yList.append([y])

	return xList,yList

def multipleLinearRegression(xList,yList):

	x = np.array(xList)
	y = np.array(yList)
	xTranspose = np.transpose(x)
	left = np.linalg.inv(np.matmul(xTranspose,x))
	right = np.matmul(xTranspose,y)
	ret = np.matmul(left,right)
	
	s = "Y = " + str(round(ret[0][0],3))
	for i in range(1,ret.shape[0]):
		if ret[i][0] <0:
			s += " - " + str(abs(round(ret[i][0],3))) + "(X" + str(i) + ")"
		else:
			s += " + " + str(round(ret[i][0],3)) + "(X" + str(i) + ")"

	print(s)


def sampleVariance(xList):
	ret = 0
	xMean = statistics.mean(xList)
	for x in xList:
		ret += (x-xMean)**2
	return ret / (len(xList)-1)

def sampleCovariance(xList, yList):
	ret = 0
	xMean = statistics.mean(xList)
	yMean = statistics.mean(yList)
	for i in range(len(xList)):
		ret += (xList[i]-xMean)*(yList[i]-yMean)
	return ret / (len(xList)-1)

def linearRegression(xList, yList):
	b1 = sampleCovariance(xList,yList) / sampleVariance(xList)
	b0 = statistics.mean(yList) - (b1*statistics.mean(xList))
	print("Linear Regression Formula: y = %.3fx + %.3f" % (b1, b0))


	#plots the data points and regression line
	plt.scatter(xList,yList)
	xMin = min(xList)
	yMin = b1*xMin + b0
	xMax = max(xList)
	yMax = b1*xMax + b0
	plt.plot([xMin, xMax], [yMin, yMax])
	plt.show()


def main():


	#xList, yList = generateSimpleLRData()
	#linearRegression(xList,yList)

	# xList,yList = generateMultipleLRData()
	# multipleLinearRegression(xList,yList)



main()