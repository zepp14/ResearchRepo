import time
NumeralCounter = 0
import numpy as np
import matplotlib
from scipy.optimize import minimize
from scipy import optimize
from scipy.special import perm

from sklearn.cluster import KMeans

import itertools

import matplotlib.pyplot as plt
#matplotlib.pygui(true)
print(matplotlib.is_interactive())

import platform
print(platform.python_version(), "Cool")
import zeppNumLib


import sdmiFuncLib as sfl





# TESTING ALGORITHM with C
r =3
alph =4
Def = [0, 0]
G_circ = [0, -8,  8]

#generate invaders
numOfInv = (5,2)

randoNum = np.random.random_sample(numOfInv)

xBound=[-15,15]
yBound=[2,15]
I_array = np.zeros(numOfInv)

for i, num in enumerate(randoNum):
    I_array[i,0] = xBound[1]*2*num[0] + xBound[0]
    I_array[i,1] = ((yBound[1]-yBound[0] ))*(num[1]) + yBound[0]


#I_array=I_array[1:]
#known problem case:

I_array = [[12.94835922, 9.56366445],
 [ 6.19156147,13.66096074],
 [ 1.23189119, 4.61296585],
 [ 9.41023588,13.65894484],
 [-9.04671401, 6.84800002]]

I_array =[[ -8.31768302, 11.81752362],
 [-12.13492404, 14.1299304 ],
 [-11.49394499,  8.76449495],
 [  7.0313238 ,  8.93215585],
 [  9.91886253 ,12.84529787]]

I_array =[[ 7.67996157,  4.03912888],
 [ 4.72170895,  4.67518833],
 [-6.38772838,  6.04776876],
 [ 5.39405897,  3.33603151],
 [ 0.05843514,  5.78576477],
 [ 5,  4.78576477]]
start_Time = time.process_time()
print(I_array)


numOfInv = (5,2)

randoNum = np.random.random_sample(numOfInv)

xBound=[-15,15]
yBound=[2,15]
I_array = np.zeros(numOfInv)

for i, num in enumerate(randoNum):
    I_array[i,0] = xBound[1]*2*num[0] + xBound[0]
    I_array[i,1] = ((yBound[1]-yBound[0] ))*(num[1]) + yBound[0]



typeEff = 1
I_dict, I_list = sfl.createInvDict(I_array)
#Enumeration
sfl.NumeralCounter = 0
bestOrder,_,meanVect, StdData = sfl.computeBestOrderEnum(I_array, r, alph, Def, G_circ,eMethod=typeEff)
print('Enumeration, Number of Minimizations solved : ', sfl.NumeralCounter)
totEffB, P_star = sfl.plotTrial(Def, I_dict, bestOrder, G_circ, alph, r, plotFlag=1,axisSz=[-17,17,-10,25],effMethod=typeEff)
pathSc1 = sfl.flightPathScore(P_star, Def)
print("path score: ", pathSc1)
print('Best Order : ',bestOrder)
print('Best Score : ',totEffB)
print("path score: ", pathSc1)
print("meanVector: ", meanVect)
print("StdData: ", StdData)
end_Time = time.process_time()
print("Process took (sec): ",((end_Time - start_Time)))
plt.show()
#Clustering
sfl.NumeralCounter=0
predictedOrder =[]
predictedOrder = ['a' for i in range(len(I_list))]
numSolved = 0

iter = 0
distance_trav = 0;
CurrPos = Def
I_in_q = I_list

predictedOrder,  I_in_q, CurrPos, distance_trav = sfl.findOrderAlgo(predictedOrder, G_circ, alph, r, CurrPos, distance_trav, I_in_q,I_dict,effMethod=typeEff)

print("____________________________")
print('Clustering, Number of Minimizations solved: ', sfl.NumeralCounter)
totEff,P_star1 = sfl.plotTrial(Def, I_dict, predictedOrder, G_circ, alph, r, plotFlag=1,axisSz=[-17,17,-5,25],effMethod=typeEff)
print('Est Order : ', predictedOrder)
print('Est Score : ', totEff)
pathSc1 = sfl.flightPathScore(P_star1, Def)
print("path score: ", pathSc1)

score = sfl.computeNormalizeScore([pathSc1, totEff], meanVect, StdData)

print("Combine score: ", score)
plt.show()
