from numpy import *

def randomArrayGenerator(n):
	myrandom = random.rand(n,n)
	print(myrandom)
	return myrandom

def arrayToMat(array):
	myMat =  mat(array)
	print(myMat)
	return myMat

def inverseMat(mat):
	myInv = mat.I 
	print(myInv)
	return myInv

myArray = randomArrayGenerator(5)
myMatrix = arrayToMat(myArray)
myInverse = inverseMat(myMatrix)
