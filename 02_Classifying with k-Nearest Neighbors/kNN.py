from numpy import *
import operator

def createDataSet():
	group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
	labels = ['A','A','B','B']
	# Labels for the corresponding members of group
	return group, labels

	
#Function to classify our test data using kNN and our sample dataset created using createDataSet()
def classifyKNN(testpoint, dataSet, labels, k):
	#Returns a tuple of dimensions of dataSet
	dataSetSize = dataSet.shape[0]
	#Repeat testpoint in a 2D (actually 1D matrix)
	#Then calculate the difference
	diffMatrix = tile(testpoint,(dataSetSize,1)) - dataSet
	#Calculating Euclidean Distance
	squareDiffMatrix = diffMat ** 2
	squareDistances = squareDiffMatrix.sum(axis=1)
	distances = squareDistances**0.5
	sortedDistancesIndices = distances.argsort() #Returns indices that would sort array

	#Keeping count of our labels for kNN with a dictionary
	classCount = {}
	for i in range(k):
		#Getting labels of k-nearest neighbours
		voteILabel = labels[sortedDistancesIndices[i]]
		classCount[voteILabel] = classCount.get(voteILabel,0) + 1

	#Sorting our dictionary based on count
	sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]


