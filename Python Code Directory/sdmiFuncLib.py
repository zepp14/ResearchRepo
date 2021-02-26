
#import pkgs

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
import platform
import zeppNumLib


def generalParaEq(th, I_a, Def, alph, r, T):
    a = alph
    Vd = 1
    Pdx = Def[0] - I_a[0]
    Pdy = Def[1] - I_a[1]
    X = -(np.cos(th)*(a*r + Pdx*np.cos(th) + Pdy*np.sin(th) - (T**2*Vd**2 - Pdx**2*np.sin(th)**2 - Pdy**2*np.cos(th)**2 + Pdx**2*a**2 + Pdy**2*a**2 + r**2 - 2*T*Vd*r + 2*Pdx*a*r*np.cos(th) + 2*Pdy*a*r*np.sin(th) + 2*Pdx*Pdy*np.cos(th)*np.sin(th) - 2*Pdx*T*Vd*a*np.cos(th) - 2*Pdy*T*Vd*a*np.sin(th))**(1/2) - T*Vd*a))/(a**2 - 1)
    Y = -(np.sin(th)*(a*r + Pdx*np.cos(th) + Pdy*np.sin(th) - (T**2*Vd**2 - Pdx**2*np.sin(th)**2 - Pdy**2*np.cos(th)**2 + Pdx**2*a**2 + Pdy**2*a**2 + r**2 - 2*T*Vd*r + 2*Pdx*a*r*np.cos(th) + 2*Pdy*a*r*np.sin(th) + 2*Pdx*Pdy*np.cos(th)*np.sin(th) - 2*Pdx*T*Vd*a*np.cos(th) - 2*Pdy*T*Vd*a*np.sin(th))**(1/2) - T*Vd*a))/(a**2 - 1)
    return  X + I_a[0], Y + I_a[1]

def paraEqPath(Ia, D, alph, r, T, bounds=[-np.pi,np.pi], num=314 ):
    th = np.linspace(bounds[0], bounds[1], num)

    K = zeppNumLib.paraPathFunc((Ia[0],Ia[1]),(D[0],D[1]),r,T,alph)
    K1 = np.array([K[0:100], K[100:200]])

    return K1[0,:], K1[1,:]


def f(x,I_a, Def, alph, r, T,G_circ):
    X,Y = generalParaEq(x, I_a, Def, alph, r, T)

    return (X -G_circ[0] )**2+(Y -G_circ[1] )**2 - G_circ[2]**2

def createInvDict(I_array):
    sz = np.size(I_array,0)
    num = np.arange(sz)
    I_list = []
    I_dict = {}
    for i, num in enumerate(num):
        I_list.append('I' + str(num))
        I_dict[I_list[i]] = I_array[i]

    return I_dict, I_list


def plotTrial(Def, I_dict, I_list, G_circ, alph, r, axisSz=[-10,10,-5,10], plotFlag=1, T = 0, effMethod = 0):
    plt.axis('equal')
    plt.axis(axisSz)
    global NumeralCounter;
    Start = Def
    G_List = np.zeros([len(I_list),])
    Eff_List = np.zeros([len(I_list),])

    if plotFlag == 1:
        plt.scatter(Def[0], Def[1], c='y', s=40, marker="^")
        theta = np.linspace(-np.pi, np.pi, 300)
        X_circ = G_circ[0] + G_circ[2]*np.cos(theta)
        Y_circ = G_circ[1] + G_circ[2]*np.sin(theta)
        plt.plot(X_circ, Y_circ,c='b')

    else:
        pass

    P_star = np.zeros([len(I_list),2])
    I_pos =  np.zeros([len(I_list),2])
    for i, Invader in enumerate(I_list):
        I_pos[i,:] = I_dict[Invader]
        Xp, Yp = paraEqPath(I_pos[i,:], Start, alph, r, T);
        NumeralCounter = NumeralCounter + 1
        phi_a = zeppNumLib.miniConvex((I_pos[i,0],I_pos[i,1]),(Start[0],Start[1]),r,T,alph,(G_circ[0],G_circ[1],G_circ[2]))
        P_star[i,:] = [0,0]
        P_star[i,0], P_star[i,1] = generalParaEq(phi_a, I_pos[i,:], Start, alph, r, T)

        Dist1 = ((P_star[i,0] - Start[0])**2 + (P_star[i,1] - Start[1])**2)**(1/2)

        T = T + Dist1

        Start = P_star[i,:]
        # Original Efficency Formula
        G_List[i] = (P_star[i,0] - G_circ[0])**2 + (P_star[i,1] - G_circ[1])**2 - G_circ[2]**2
        if effMethod == 0:
            Eff_List[i] = (G_List[i] / Dist1)
        elif effMethod == 1:
            Eff_List[i] = (G_List[i])
        elif effMethod == 2:
            Eff_List[i] = (1 / Dist1)

        if plotFlag == 1:
            plt.plot(Xp, Yp, 'm--')
            plt.scatter( I_pos[i,0], I_pos[i,1],c='r',s=120 )
            plt.scatter( P_star[i,0], P_star[i,1], s=120,c='g',marker='d')
        else:
            pass
    if plotFlag == 1:
        plt.plot([Def[0], P_star[0,0]], [Def[1], P_star[0,1]], 'y--')
        plt.plot(P_star[:,0], P_star[:,1], 'y--', label="Flight Path")
        plt.scatter(Def[0], Def[1], c='y', s=120, marker="^", label="Defender")
        plt.plot(np.nan, np.nan, 'm--',label="Dominant Regions")
        plt.scatter( np.nan, np.nan, s=120,c='g',marker='d',label="Capture Points")
        plt.scatter(  np.nan, np.nan,c='r',s=120,label="Invaders" )
    TotEff = np.sum(Eff_List)

    return TotEff, P_star


def loadTrialData(filename, trialNo):
    data = np.load(filename,allow_pickle=True)
    NumCount = data[trialNo][0]
    BestOrder = data[trialNo][1]
    BestScore = data[trialNo][2]
    Iarr = data[trialNo][3]
    ListOfScores = data[trialNo][4]

    return  BestOrder,BestScore, Iarr, NumCount, ListOfScores

def flightPathScore(P_star, Def):
    score = 0
    length = np.size(P_star, 0)
    if(length <= 2):
        score = zeppNumLib.dirCostFunc((Def[0],Def[1]), (P_star[0,0],P_star[0,1]), (P_star[1,0],P_star[1,1]))
    elif(length > 2):
        arrayOfPose = np.vstack([np.array(Def),P_star])
        Arr = np.array([0,1,2])
        rng = length -  Arr[2]
        for i in range(0, rng + 1):
            K = Arr + i
            score = score + zeppNumLib.dirCostFunc((arrayOfPose[K[0],0],arrayOfPose[K[0],1]), (arrayOfPose[K[1],0],arrayOfPose[K[1],1]), (arrayOfPose[K[2],0],arrayOfPose[K[2],1]))
    return score

def numInTarget(P_star,G_circ):
    inTarg = []
    for P in P_star:
        PT = (P[0] - G_circ[0] )**2+(P[1] - G_circ[1] )**2 - G_circ[2]**2
        if PT < 0:
            inTarg.append(1)
        else:
            inTarg.append(0)

    return sum(inTarg)

def computeBestOrderEnum2(I_array, r, alph, Def, G_circ, dist=0, eMethod = 0 ):
    I_dict, I_list = createInvDict(I_array)
    szInvArr = len(I_dict)

#Generate Permutation Order List
    MyArray = np.arange(szInvArr)
    permut = itertools.permutations(MyArray)
    arrayOfOrders = np.empty((0,szInvArr))

    for p in permut:
        arrayOfOrders = np.append(arrayOfOrders,np.atleast_2d(p),axis=0)


#Apply Order List to I_array
    permNum = perm(szInvArr,szInvArr,exact=True)
    orderInvaderArr = []
    orderInvaderArr = [['a' for i in range(szInvArr)] for j in range(permNum)]

    for count, order in enumerate(arrayOfOrders):
        for count1, i in enumerate(order):
            orderInvaderArr[count][count1] =  I_list[int(i)]
            pass


    totEff = np.zeros([permNum,1])

# go through order to find the best
    P_s = []
    for i, order in enumerate(orderInvaderArr):
        totEff[i], P_ss = plotTrial(Def, I_dict, order, G_circ, alph, r, axisSz=[-10,10,-5,10], T=dist,plotFlag=0,effMethod = eMethod)
        P_s.append(P_ss)

    bestIdx = np.where(totEff == max(totEff))
    P_star = P_s[ int(bestIdx[0])]
    #print(int(bestIdx[0]))
    bestOrder = orderInvaderArr[int(bestIdx[0])]
    #totEff,P_star2 = plotTrial(Def, I_dict, bestOrder, G_circ, alph, r, axisSz=[-10,10,-5,10], plotFlag=0)
    totEff = np.transpose(np.array(totEff))
    sortedEff = np.sort(totEff)
    return bestOrder,P_star,sortedEff


NumeralCounter = 0

def meanNormal(data, ):
    sz = np.shape(data)
    norm_data = np.empty(np.shape(data))
    meanVect = np.mean(data,axis=0)
    print(meanVect)
    for i in range(0,sz[0]):
        norm_data[i,:] = np.subtract(data[i,:],meanVect)


    return norm_data, meanVect

#Feature Scaling
def featureScaling(data,scale_factor):
    sz = np.shape(data)
    scaled_data = np.empty(np.shape(data))
    for i in range(0,sz[0]):
        scaled_data[i,:] = np.divide(data[i,:],scale_factor+1e-8)

    return scaled_data



########################## PCA function

def PCA_Func(dataIn, k):

    sz = np.shape(dataIn)
    nD = sz[1]

  #convert data to matrix

    data = np.mat(dataIn)

  #compute covariance

    Sig = np.var(data)
    Sig = np.zeros([nD,nD])

    for i in range(0,sz[0]):
        Sig = Sig + np.matmul(np.transpose(data[i,:]), data[i,:])
    Sig = np.divide(Sig, sz[0]+1e-8)

  #Compute Eigen Vector
    u, s, vh = np.linalg.svd(Sig,full_matrices=True)

    U=u[:,0:k]


    return U





def computeBestOrderEnum(I_array, r, alph, Def, G_circ, dist=0, eMethod = 0 ):
    I_dict, I_list = createInvDict(I_array)
    szInvArr = len(I_dict)

#Generate Permutation Order List
    MyArray = np.arange(szInvArr)
    permut = itertools.permutations(MyArray)
    arrayOfOrders = np.empty((0,szInvArr))

    for p in permut:
        arrayOfOrders = np.append(arrayOfOrders,np.atleast_2d(p),axis=0)


#Apply Order List to I_array
    permNum = perm(szInvArr,szInvArr,exact=True)
    orderInvaderArr = []
    orderInvaderArr = [['a' for i in range(szInvArr)] for j in range(permNum)]

    for count, order in enumerate(arrayOfOrders):
        for count1, i in enumerate(order):
            orderInvaderArr[count][count1] =  I_list[int(i)]
            pass


    totEff = np.zeros([permNum,1])
   #print(orderInvaderArr)
# go through order to find the best
    P_s = []
    numInTar = []
    flightScore = []
    for i, order in enumerate(orderInvaderArr):
        totEff[i], P_ss = plotTrial(Def, I_dict, order, G_circ, alph, r, axisSz=[-10,10,-5,10], T=dist,plotFlag=0,effMethod = eMethod)
        numInTar.append(numInTarget( P_ss,G_circ))
        flightScore.append(flightPathScore(P_ss, Def))
        P_s.append(P_ss)

    bestNum  = min(numInTar)
    P_s_best = []
    flightScore_best = []
    totEff_best = []
    bestOrderList = []
    numBest = []
    for i, num in enumerate(numInTar):
        if num == bestNum :
            P_s_best.append(P_s[i])
            flightScore_best.append(flightScore[i])
            totEff_best.append(totEff[i])
            bestOrderList.append(orderInvaderArr[i])
            numBest.append(num)
    #minFlPa = np.argmin(flightScore_best)
    minFlPa = np.argmax(totEff_best)


    #fig, ax = plt.subplots(figsize=(20,4), ncols=1)
   # ax.plot(numInTar)
    #ax.axis('auto')



    #fig1 = plt.figure(figsize=(10,10))
    #ax1 = fig1.add_subplot(111, projection='3d')
    #ax1.scatter(numInTar,flightScore,  totEff, marker='.')
    #ax1.set_xlabel('numInTar')
    #ax1.set_ylabel('FLScore')
    #ax1.set_zlabel('Eff')
    #plt.show()




    sz = np.shape(flightScore_best)
    dataB = np.hstack((np.array(flightScore_best).reshape([sz[0],1]), np.array(totEff_best).reshape([sz[0],1])))
        #Mean Normalize Data
    dataSetA, meanVect = meanNormal(dataB)
        #Feature Scale by std of feature
    dataSet = featureScaling(dataSetA, np.std(dataSetA,axis=0))
    #ax.scatter(dataSet[:,0],dataSet[:,1], s=6)
    #ax.scatter(flightScore_best,totEff_best, s=6)
    #ax.axis('auto')
    #plt.show()


    U = PCA_Func(dataSet, 1)
    z = np.matmul(np.transpose(U), np.transpose(np.mat(dataSet) ))
    z =np.array(z)
    bestIdx = np.argmax(z)
    print('U:', U);
    #print("Max of PCA ", bestOrderList[bestIdx])
#     fig, ax = plt.subplots(figsize=(6,6), ncols=1)
#     ax.scatter(z, np.zeros(np.shape(z)), s=2)
#     #ax.plot(z)
#     ax.axis('auto')
#     plt.show()





    print('z-value', z[0,bestIdx])

    P_star = P_s_best[ int(bestIdx)]
    #print(int(bestIdx[0]))
    bestOrder = orderInvaderArr[int(bestIdx)]
    #totEff,P_star2 = plotTrial(Def, I_dict, bestOrder, G_circ, alph, r, axisSz=[-10,10,-5,10], plotFlag=0)
    totEff = np.transpose(np.array(totEff))
    sortedEff = np.sort(totEff)
    return bestOrderList[bestIdx],P_star, meanVect, np.std(dataSetA,axis=0)

def chooseCluster(r, alph, Def, G_circ, C1, C2, dist=0, effMethod=0):
    I_array = [C1, C2]
    I_dict, I_list = createInvDict(I_array)
    bestOrder,P_star,_,_ = computeBestOrderEnum(I_array, r, alph, Def, G_circ, dist=dist, eMethod = effMethod)
    ##removelater
#     totEff0,P_star2 = plotTrial(Def, I_dict, ['I0','I1' ], G_circ, alph, r, axisSz=[-10,10,-5,10], plotFlag=0)
#     print('Eff first:',  totEff0)
#     totEff1,P_star2 = plotTrial(Def, I_dict, ['I1','I0' ], G_circ, alph, r, axisSz=[-10,10,-5,10], plotFlag=0)
#     print('Eff second:',  totEff1)
#     totEff1,P_star2 = plotTrial(Def, I_dict, bestOrder, G_circ, alph, r, axisSz=[-10,10,-5,10], plotFlag=0)
#     print('Eff best:',  totEff1)

    bestIdx = I_list.index(bestOrder[0])
    return bestIdx

def orderByCluster(I_dict, I_list, r, alph, Def, G_circ, dist=0, eMethod=0):
    #I_dict, I_list = createInvDict(I_array)
    # find inv that are the furthest apart
    l = itertools.combinations(I_list,r=2)
    pairs = list(l)
    #print(list(l))
    Distances = np.zeros([len(pairs),1])
    for i, p in enumerate(pairs):
        a = I_dict[p[0]]
        b = I_dict[p[1]]
        Distances[i] = np.linalg.norm(np.array(a)-np.array(b))

    #print(Distances)
    farIdx = np.where(Distances == max(Distances))
    shpe = np.shape(farIdx)
    if(shpe[1] > 1):
        farIdx = farIdx[0]

    FurthestPair = pairs[int(farIdx[0])]
    #print(FurthestPair)
    startPos = np.zeros([2,3])
    startPos[0,:-1] = np.array(I_dict[FurthestPair[0]])
    startPos[1,:-1] = np.array(I_dict[FurthestPair[1]])
    startPos[0,2] = 0 #np.exp(-np.linalg.norm(np.array(I_dict[FurthestPair[0]])-Def))
    startPos[1,2] = 0 #np.exp(-np.linalg.norm(np.array(I_dict[FurthestPair[1]])-Def))

    #print(pairs[int(farIdx[0])])

    #print(farIdx)
    #print(Distances)
    Iarr = []
    Iarr3 = []
    for I in I_list:
        Iarr.append(I_dict[I])
    for inv in Iarr:
        Inv = np.array(inv)
        #inv =[inv[0], inv[1], np.exp(-np.linalg.norm(Inv-Def)) ]
        inv =[inv[0], inv[1], 0]
        Iarr3.append(inv)

    #print(Iarr3)
    kmeans = KMeans(n_clusters=2, init=startPos,n_init=1).fit(Iarr3)
    #kmeans = KMeans(n_clusters=2, init='random').fit(Iarr3)
    #print(kmeans.labels_)
#print(kmeans.cluster_centers_)

    C1 = kmeans.cluster_centers_[0,:]
    C2 = kmeans.cluster_centers_[1,:]

    #print("Clust Score", kmeans.cluster_centers_)


    #plt.scatter( C1[0], C1[1],c='k',marker='*')
    #plt.scatter( C2[0], C2[1],c='k',marker='*')

    #plotTrial(Def, I_dict, bestOrder, G_circ, alph, r)

    bestClust = chooseCluster(r, alph, Def, G_circ, C1[:-1], C2[:-1], dist=dist, effMethod=eMethod)

    invInClust = []

    for i, invad in enumerate(I_list):
        if kmeans.labels_[i] == bestClust:
            invInClust.append(invad)

    invInClust2 = []
    for i, invad in enumerate(I_list):
        if kmeans.labels_[i] != bestClust:
            invInClust2.append(invad)

    return invInClust, invInClust2, kmeans.labels_



#creating some smaller func for order inserstion
#easycaseI - Single Invader in Clust
def ecI(Def, alph, r, Invader, I_dict, I_in_q, Pred_List, distT, Currpos, eMethod=0, G_circ=[0,-8,8] ):
    indA = Pred_List.index('a')

    Pred_List[indA] = Invader
    I_in_q.remove(Invader);
    totEff, P_star = plotTrial(Currpos, I_dict, [Invader], G_circ, alph, r,  plotFlag=0, effMethod=eMethod)
    distT = distT+ ((P_star[0,0] - Currpos[0])**2 + (P_star[0,1] - Currpos[1])**2)**(1/2)
    Ps = P_star[0]
    Currpos = Ps
    return I_in_q, Pred_List, distT, Currpos

def findOrderAlgo(predictedOrder, G_circ, alph, r, CurrPos, distance_trav, I_in_q, I_dict, overFlowState = False, printClust=False, effMethod=0 ):
    iter = 0
    sz = len(I_in_q)
    clusters = []
    numSolved = 0
    Def = CurrPos
    dist = distance_trav
    while numSolved < sz:

        if(len(I_in_q) > 2):

            invInClust,invNotInClust2,_ = orderByCluster(I_dict, I_in_q, r, alph, CurrPos, G_circ, dist = distance_trav, eMethod=effMethod)
            if (len(invNotInClust2) /  len(invInClust)) >= 3:
                #pass
                temp = invInClust
                invInClust = invNotInClust2
                invNotInClust2 = temp

        else:
            invInClust = I_in_q


        if(printClust):
            print(invInClust)

            clusters.insert(len(clusters),invInClust[0:(len(invInClust)+1)])
            #print(len(clusters))
            if(len(invInClust)==1):
                clusters[len(clusters)-1] = invInClust[0:(len(invInClust)+1)]

            #clusters.append(invInClust)

        if(len(invInClust) == 1):
            I_in_q, predictedOrder, distance_trav, CurrPos = ecI(Def, alph, r,invInClust[0], I_dict, I_in_q, predictedOrder, distance_trav, CurrPos, eMethod=effMethod)
            numSolved = numSolved + 1
        elif(len(invInClust) == 3):
            numSolved = numSolved + 3
            Iarr = []
            notIarr = []

            for I in invInClust:
                Iarr.append(I_dict[I])

            for I in invNotInClust2:
                notIarr.append(I_dict[I])

            C1 = np.array(notIarr).mean(0)
            #print(C1)
            print("not Cluster " ,invNotInClust2)
            print("Cluster " ,invInClust)
            Iarr_mod = np.vstack((C1,Iarr))
            #bestof3, p_s, _ = computeBestOrderEnum(Iarr, r, alph, CurrPos, G_circ, dist=distance_trav, eMethod =effMethod )
            bestof3, p_s, _, _ = computeBestOrderEnum(Iarr_mod, r, alph, CurrPos, G_circ, dist=distance_trav, eMethod =effMethod )
            bestof3.remove('I0')
            corrBest=bestof3.copy()
            #print('before:', corrBest)
            #print("best before convert",bestof3)
            corrBest[bestof3.index("I1") ]= invInClust[0]
            corrBest[bestof3.index("I2") ]= invInClust[1]
            corrBest[bestof3.index("I3") ]= invInClust[2]
            #print("best after convert",corrBest)



            #print('after:', corrBest)


            I_in_q, predictedOrder, distance_trav, CurrPos = ecI(Def, alph, r,corrBest[0], I_dict, I_in_q, predictedOrder, distance_trav, CurrPos,eMethod=effMethod)
            I_in_q, predictedOrder, distance_trav, CurrPos = ecI(Def, alph, r,corrBest[1], I_dict, I_in_q, predictedOrder, distance_trav, CurrPos,eMethod=effMethod)
            I_in_q, predictedOrder, distance_trav, CurrPos = ecI(Def, alph, r,corrBest[2], I_dict, I_in_q, predictedOrder, distance_trav, CurrPos,eMethod=effMethod)


        elif(len(invInClust) == 2):
            numSolved = numSolved + 2
            Temp0, Temp1,_  = orderByCluster(I_dict, invInClust, r, alph, CurrPos, G_circ, eMethod=effMethod)
            I_in_q, predictedOrder, distance_trav, CurrPos = ecI(Def, alph, r,Temp0[0], I_dict, I_in_q, predictedOrder, distance_trav, CurrPos,eMethod=effMethod)
            I_in_q, predictedOrder, distance_trav, CurrPos = ecI(Def, alph, r,Temp1[0], I_dict, I_in_q, predictedOrder, distance_trav, CurrPos,eMethod=effMethod)


        elif(len(invInClust) > 3):

            Clust = invInClust
            numSolved = numSolved + len(invInClust)

            predictedOrder,  I_in_q2, CurrPos, distance_trav = findOrderAlgo(predictedOrder, G_circ, alph, r, CurrPos, distance_trav, invInClust,I_dict, printClust=False,overFlowState = True, effMethod=effMethod)

            invInClust = Clust
            I_q = []
            for ele in I_in_q:
                if ele not in predictedOrder:
                    I_q.append(ele)

            I_in_q = I_q








        iter = iter + 1
        if (iter > 10):

            break
    #print(clusters)

    szInvArr = len(clusters)
    predictedOrder2 = predictedOrder
#Generate Permutation Order List
    if(printClust):
        MyArray = np.arange(szInvArr)
        permut = itertools.permutations(MyArray)
        arrayOfOrders = np.empty((0,szInvArr))
        modOrder = []
        for p in permut:
            arrayOfOrders = np.append(arrayOfOrders,np.atleast_2d(p),axis=0)

        permNum = perm(szInvArr,szInvArr,exact=True)

        modOrder = []
        modOrder = [['a' for i in range(szInvArr)] for j in range(permNum)]


        for count, order in enumerate(arrayOfOrders):
            for count1, i in enumerate(order):
                modOrder[count][count1] =  clusters[int(i)]
                pass
        flat_list = [item for sublist in modOrder for item in sublist]
        modOrder= [item for sublist in flat_list for item in sublist]
        modOrder = np.reshape(modOrder, (permNum,len(predictedOrder)))

        totEff = np.zeros([permNum,1])

    # go through order to find the best
        P_s = []
        for i, order in enumerate(modOrder):
            totEff[i], P_ss = plotTrial(Def, I_dict, order.tolist(), G_circ, alph, r, axisSz=[-10,10,-5,10], T=dist,plotFlag=0, effMethod=effMethod)
            P_s.append(P_ss)

        bestIdx = np.where(totEff == max(totEff))
        P_star = P_s[ int(bestIdx[0])]
        #print(int(bestIdx[0]))
        bestOrder = modOrder[int(bestIdx[0])]
        predictedOrder = bestOrder.tolist()

        #print("new best:",bestOrder )


    return predictedOrder2, I_in_q, CurrPos, distance_trav



def computeNormalizeScore(dataVec, meanVec, scalingVec):

    u = np.array([[-1/(np.sqrt(2))],[1/(np.sqrt(2))]])

    out = np.zeros([2,1])
    out[0] = (dataVec[0] -  meanVec[0]) / scalingVec[0]
    out[1] = (dataVec[1] -  meanVec[1]) / scalingVec[1]
    u = np.mat(u.transpose())
    out = np.mat(out)
    print(u)
    return np.matmul(u,out)

computeNormalizeScore([2,3], [1.2,3.4], [1,3])
