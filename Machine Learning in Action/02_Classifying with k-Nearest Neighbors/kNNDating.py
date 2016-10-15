from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
	group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
	labels = ['A','A','B','B']
	return group, labels

	
def classifyKNN(testpoint, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMatrix = tile(testpoint,(dataSetSize,1)) - dataSet
	squareDiffMatrix = diffMatrix ** 2
	squareDistances = squareDiffMatrix.sum(axis=1)
	distances = squareDistances**0.5
	sortedDistancesIndices = distances.argsort() #Returns indices that would sort array

	classCount = {}
	for i in range(k):
		voteILabel = labels[sortedDistancesIndices[i]]
		classCount[voteILabel] = classCount.get(voteILabel,0) + 1

	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]


def fileToMatrix(filename):
	fr = open(filename)
	numberOfLines = len(fr.readlines())
	#n rows, 3 columns
	returnMatrix = zeros((numberOfLines,3))
	classLabelVector=[]

	fr = open(filename)
	index = 0
	for line in fr.readlines():
		line = line.strip()
		listFromLine = line.split('\t')
		#Class label is at index 3 of listFromLine
		returnMatrix[index,:] = listFromLine[0:3]
		#Negative indexing in Python
		classLabelVector.append(listFromLine[-1])
		index+=1
	return returnMatrix,classLabelVector

def createScatterplot(filename):
	datingMatrix, datingLabels = fileToMatrix(filename)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(datingMatrix[:,1], datingMatrix[:,2])
	plt.show()
