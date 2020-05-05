
import itertools
import numpy as np
from scipy.spatial import ConvexHull
from random import randint
import matplotlib.pyplot as plt

def SampleSelection_v1(setOfPoints,nSamples,returnIndicies=False,nTrials=10):
    """Randomly selecting samples to use for SF analysis. Checks for repetition in the sample space. """
    nPoints = len(setOfPoints)
    maxDist=0
    for trial in range(nTrials):
        indicies = []
        sampleSet = []
        while len(indicies) < nSamples:
            x = randint(0,nPoints-1)
            if x in indicies:
                continue
            if x in setOfPoints:
                continue
            indicies.append(x)
            sampleSet.append(setOfPoints[x])
        dist = TotalAverageDistance(sampleSet)
        if dist >= maxDist:
            maxDist=dist
            bestPoints = sampleSet.copy()
            bestIndicies = indicies.copy()


    if returnIndicies:
        return bestIndicies
    return bestPoints

def TotalAverageDistance(setOfPoints):
    dist = 0
    for i in range(len(setOfPoints)):
        for j in range(i+1,len(setOfPoints)):
            dist+=np.linalg.norm(setOfPoints[i]-setOfPoints[j])
    return dist
def arreqclose_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if elem.size == myarr.size and np.allclose(elem, myarr,atol=1E-6)), False)

def SampleSelection_v2(setOfPoints,nSamples,returnIndicies=False, nTrials=10, debug=False):
    """Using Convex Hull to select boundary points. Filling the rest by performing random selections  """
    nPoints = setOfPoints.shape[0]
    hull = ConvexHull(setOfPoints)
    indicies = hull.vertices.tolist()
    boundaryPoints = [];removeIndicies=[]
    for idx in indicies:
        if not arreqclose_in_list(setOfPoints[idx],boundaryPoints):
            boundaryPoints.append(setOfPoints[idx])
        else:
            removeIndicies.append(idx)
    for idx in removeIndicies:
        indicies.remove(idx)
    if debug:print("Finished Calculating Convex Hull of the set. Number of boundary points " + str(len(boundaryPoints)))

    if len(indicies) >= nSamples: #Perform prunning operation
        #Removing the entry that lowers the entropy the least
        while len(indicies) != nSamples:
            worstDist=0
            for i in range(len(boundaryPoints)):
                dist = TotalAverageDistance(boundaryPoints.copy().pop(i))
                if dist > worstDist:
                    worstDist=dist
                    idx = i
            boundaryPoints.pop(idx)
            indicies.pop(idx)
        if returnIndicies:
            return indicies
        return boundaryPoints
    else:

        maxDist = 0
        for trial in range(nTrials):
            if debug:print("Begining sampling trial " + str(trial))

            points = boundaryPoints.copy()
            idx = indicies.copy()
            while len(points) < nSamples:
                x = randint(0,nPoints-1)
                if x in idx:
                    continue
                if arreqclose_in_list(setOfPoints[x],points):
                    continue
                idx = np.append(idx,x)
                points.append(setOfPoints[x])
            dist = TotalAverageDistance(points)
            if dist >= maxDist:
                maxDist=dist
                bestPoints = points.copy()
                bestIndicies = idx.copy()
            if debug: print(maxDist,len(bestPoints),len(bestIndicies))
        if returnIndicies:
            return bestIndicies
        return bestPoints

def SampleSelection_v3(setOfPoints,nSamples,returnIndicies=False, nTrials=10, debug=False):
    """Separating into clusters. Using Convex Hull to select boundary points. Filling the rest by performing random selections  """
    # from sklearn.mixture import GaussianMixture
    # model = GaussianMixture(n_components=4)
    # model.fit(setOfPoints)
    # yhat =model.predict(setOfPoints)
    nPoints = setOfPoints.shape[0]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data = scaler.fit_transform(setOfPoints)
    from sklearn.cluster import DBSCAN
    model = DBSCAN(eps=0.1, min_samples=10)
    yhat = model.fit_predict(data)
    clusters=np.unique(yhat)


    Gindicies = [];GboundaryPoints=[]
    for cluster in clusters:
        row_ix = np.where(yhat==cluster)
        clusterPoints = np.squeeze(setOfPoints[row_ix,:])
        if np.unique(clusterPoints,axis=0).shape[0] < 3:
            GboundaryPoints.append(setOfPoints[row_ix[0][0]])
            Gindicies.append(row_ix[0][0])
            continue
        hull = ConvexHull(clusterPoints)
        indicies = hull.vertices.tolist()
        boundaryPoints = [];removeIndicies=[]
        for idx in indicies:
            if not arreqclose_in_list(setOfPoints[row_ix[0][idx]],GboundaryPoints):
                GboundaryPoints.append(setOfPoints[row_ix[0][idx]])
            else:
                removeIndicies.append(idx)
        for idx in removeIndicies:
            indicies.remove(idx)
        for idx in indicies:
            Gindicies.append(row_ix[0][idx])
        if debug:print("Finished Calculating Convex Hull of the set. Number of boundary points " + str(len(GboundaryPoints)))

    if len(Gindicies) >= nSamples: #Perform prunning operation
        #Removing the entry that lowers the entropy the least
        while len(Gindicies) != nSamples:
            worstDist=0
            for i in range(len(GboundaryPoints)):
                dist = TotalAverageDistance(GboundaryPoints.copy().pop(i))
                if dist > worstDist:
                    worstDist=dist
                    idx = i
            GboundaryPoints.pop(idx)
            Gindicies.pop(idx)
        if returnIndicies:
            return Gindicies
        return GboundaryPoints
    else:

        maxDist = 0
        for trial in range(nTrials):
            if debug:print("Begining sampling trial " + str(trial))

            points = GboundaryPoints.copy()
            idx = Gindicies.copy()
            while len(points) < nSamples:
                x = randint(0,nPoints-1)
                if x in idx:
                    continue
                if arreqclose_in_list(setOfPoints[x],points):
                    continue
                idx = np.append(idx,x)
                points.append(setOfPoints[x])
            dist = TotalAverageDistance(points)
            if dist >= maxDist:
                maxDist=dist
                bestPoints = points.copy()
                bestIndicies = idx.copy()
            if debug: print(maxDist,len(bestPoints),len(bestIndicies))
        if returnIndicies:
            return bestIndicies
        return bestPoints

# def SampleSelection_v3(setOfPoints,nSamples,returnIndicies=False, nTrials=1):
#     """Using Convex Hull to select boundary points. Calculating Lattice points for remaining stuff.  """
#     maxSize = np.max(np.stack(setOfPoints),axis=0)
#     minSize = np.min(np.stack(setOfPoints),axis=0)
#
#     #Calculating number of lattice points in each dimension.
#     lpN = int(np.ceil(nSamples**(1/setOfPoints.shape[1])))
#
#     def split(min,max,N):
#         size = (max-min)/(N-1.0)
#         values = [min]
#         for i in range(1,N):
#             values.append(min+size*i)
#         return values
#     listOfLaticeDivisions = []
#     for dim in range(len(maxSize)):
#         listOfLaticeDivisions.append(split(minSize[dim],maxSize[dim],lpN))
#     for i in range(maxSize.shape[0]-1):
#         if i == 0:
#             iterator = itertools.product(listOfLaticeDivisions[0],listOfLaticeDivisions[1])
#         else:
#             print(dir(iterator))
#             iterator = itertools.product(iterator,listOfLaticeDivisions[i+1])
#     for latticePoint in iterator:
#         print (latticePoint)
#
#     # Trimming the lattice points if the number is too large
#
#     return []


def CreateTestSet(nPoints,nDim):
    set = np.random.rand(nPoints,nDim)
    return set

if __name__ == "__main__":
    set = CreateTestSet(400,3)
    points1 = SampleSelection_v1(set,50)
    print(TotalAverageDistance(points1))
    points2 = SampleSelection_v2(set,50)
    print(TotalAverageDistance(points2))
    points3 = SampleSelection_v3(set,50)
    print(TotalAverageDistance(points3))
    # points4 = SampleSelection_v4(set,50)
    # print(TotalAverageDistance(points4))
    #
    # plt.scatter(np.stack(set)[:,0],np.stack(set)[:,1],color='b')
    # plt.scatter(np.stack(points1)[:,0],np.stack(points1)[:,1],color='r')
    # plt.show()
    # plt.scatter(np.stack(set)[:,0],np.stack(set)[:,1],color='b')
    # plt.scatter(np.stack(points2)[:,0],np.stack(points2)[:,1],color='r')
    # plt.show()
    # plt.scatter(np.stack(set)[:,0],np.stack(set)[:,1],color='b')
    # plt.scatter(np.stack(points4)[:,0],np.stack(points4)[:,1],color='r')
    # plt.show()
