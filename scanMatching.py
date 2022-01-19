import numpy as np
import matplotlib.pyplot as plt
from transformations import transform2Odom
from ndt import ndtCartesian

def makePositive(hessian):
    #print("Before splitting the hessian matrix : ", hessian)
    eigValues, eigVectors = np.linalg.eigh(hessian)
    #print("The Eigen values and eigen vectos of the matrix are : ", eigValues)
    #newSigma = np.matrix([[-sigma[0], 0.0, 0.0], [0.0, sigma[1], 0.0], [0.0, 0.0, sigma[2]]])
    sumOfNegativeEig = 0.0
    minPosEig = 32456.0
    for i in range(len(eigValues)):
        if eigValues[i] < 0:
            sumOfNegativeEig += eigValues[i]
        if eigValues[i] > 0:
            if minPosEig > eigValues[i]:
                minPosEig = eigValues[i]
    #print("The minimum positive eigen value among the values is ", minPosEig)
    sumOfNegativeEig *= 2
    t = (sumOfNegativeEig**2)*100 + 1
    for i in range(len(eigValues)):
        if eigValues[i] < 0.0:
            eigValues[i] = minPosEig*((sumOfNegativeEig - eigValues[i])**2)/t
    #print("The new eigen values are ", eigValues)
    diagonalMat = np.matrix([[eigValues[0], 0.0, 0.0], [0.0, eigValues[1], 0.0], [0.0, 0.0, eigValues[2]]])
    eigHessian = eigVectors @ diagonalMat @ np.linalg.inv(eigVectors)
    #print("The old hessian is ", hessian)
    #print("After the joining of the new hessian : ", eigHessian)
    #eigenValues = np.linalg.eigvals(hessian)
    #print("The sigmas of this matrix are ", sigma)
    #print("The eigen values of hessian are : ", eigenValues)
    #neweigenvalues = np.linalg.eigvals(newHessian)
    #print("The eigen values of the new hessian is : ", neweigenvalues)
    return eigHessian

def nearestNeighbour(cellID, cellCenters):
    """
    Computes the nearest neighbour of the cell under consideration and obtain a new covariance matrix
    :param cellIDs:
    :param cellCenterDict:
    :return:
    """
    neighbours = []
    temp1 = [cellCenters[cellID][0] + 1.0, cellCenters[cellID][1]]
    temp2 = [cellCenters[cellID][0], cellCenters[cellID][1] + 1.0]
    temp3 = [cellCenters[cellID][0] - 1.0, cellCenters[cellID][1]]
    temp4 = [cellCenters[cellID][0], cellCenters[cellID][1] - 1.0]
    temp5 = [cellCenters[cellID][0] - 1.0, cellCenters[cellID][1] - 1.0]
    temp6 = [cellCenters[cellID][0] + 1.0, cellCenters[cellID][1] + 1.0]
    temp7 = [cellCenters[cellID][0] - 1.0, cellCenters[cellID][1] + 1.0]
    temp8 = [cellCenters[cellID][0] + 1.0, cellCenters[cellID][1] - 1.0]
    #print("The cellID and the cell centers are : ", cellID)
    #print("The cell centers are : ", cellCenters.shape)
    #print("The cell center for this given id is ", cellCenters[cellID][0])
    #print("The temp value is : ", temp)
    for i in range(len(cellCenters)):
        if temp1[0] == cellCenters[i][0] and temp1[1] == cellCenters[i][1]:
            neighbours.append(i)
        if temp2[0] == cellCenters[i][0] and temp2[1] == cellCenters[i][1]:
            neighbours.append(i)
        if temp3[0] == cellCenters[i][0] and temp3[1] == cellCenters[i][1]:
            neighbours.append(i)
        if temp4[0] == cellCenters[i][0] and temp4[1] == cellCenters[i][1]:
            neighbours.append(i)
        if temp5[0] == cellCenters[i][0] and temp5[1] == cellCenters[i][1]:
            neighbours.append(i)
        if temp6[0] == cellCenters[i][0] and temp6[1] == cellCenters[i][1]:
            neighbours.append(i)
        if temp7[0] == cellCenters[i][0] and temp7[1] == cellCenters[i][1]:
            neighbours.append(i)
        if temp8[0] == cellCenters[i][0] and temp8[1] == cellCenters[i][1]:
            neighbours.append(i)

    neighbours = np.array(neighbours)
    return neighbours

def updateGradient(point, inverseCov, mean, pose, iterator):
    #jT = jacobian.T
    """
    To compute the jacobian matrix..
    :param point: the 2D point
    :param inverseCov: The inverse covariance matrix of the cluster
    :param mean: mean of the cluster
    :param pose: pose of the robot
    :return: the point's jacobian..
    """
    poseX = pose[0]
    poseY = pose[1]
    orient = pose[2]
    j1 = np.array([[1, 0]])
    j2 = np.array([[0, 1]])
    j3 = np.array([[-poseX * np.sin(orient) - poseY * np.cos(orient),
                   poseX * np.cos(orient) - poseY * np.sin(orient)]])
    jT = np.array([j1, j2, j3])
    grad = []
    normalised = np.matrix(point - mean)
    """print("During this iteration the normalized point is ", normalised)
    firstFactor = np.dot(-1*normalised, inverseCov)
    secondFactor = np.dot(firstFactor, normalised.T)
    print("The first factor in this iteration is ", firstFactor)
    print("Th second factor i this iteration is ", secondFactor)
    print("The exponential power in this iteration is ", np.exp(secondFactor / 2))
    print("The final formula from the full exponential is ",np.exp(np.dot((np.dot(-normalised, inverseCov)), normalised.T) / 2))
    #inverseCov = np.matrix(np.linalg.inv(covMat))
    print("The inverse covariance matrix is ", inverseCov)"""
    for rows in jT:
        rowtranspose = np.matrix(rows).T
        gradient = np.dot(np.dot(normalised, inverseCov), rows.T) * \
                   np.exp(np.dot((np.dot(-normalised, inverseCov)), normalised.T) / 2)
        #print("The gradient value is ", gradient[0,0])
        grad.append(gradient[0,0])

    grad = np.array([grad])
    gradTranspose = grad.T
    #print("The gradient transposed is ", gradTranspose)
    return gradTranspose

def updateHessian(point, invCovmat, mean, pose, iterator):
    poseX = pose[0]
    poseY = pose[1]
    orient = pose[2]
    normalised = np.matrix(point - mean)
    j1 = np.array([[1, 0]]).T
    j2 = np.array([[0, 1]]).T
    j3 = np.array([[-poseX * np.sin(orient) - poseY * np.cos(orient),
                    poseX * np.cos(orient) - poseY * np.sin(orient)]]).T
    secDerivative = np.array([[-poseX*np.cos(orient) + poseY*np.sin(orient), -poseX*np.sin(orient) - poseY*np.cos(orient)]]).T
    h11 = np.exp(-normalised @ invCovmat @ (normalised.T)) @ [(normalised @ invCovmat @ j1) @ (-normalised @ invCovmat @ j1) + ((j1.T) @ invCovmat @ j1)]
    #print("The hessian 1 is ", h11)
    h12 = np.exp(-normalised @ invCovmat @ (normalised.T)) @ [(normalised @ invCovmat @ j1) @ (-normalised @ invCovmat @ j2) + ((j2.T) @ invCovmat @ j1)]
    h13 = np.exp(-normalised @ invCovmat @ (normalised.T)) @ [(normalised @ invCovmat @ j1) @ (-normalised @ invCovmat @ j3) + ((j3.T) @ invCovmat @ j1)]
    h21 = np.exp(-normalised @ invCovmat @ (normalised.T)) @ [(normalised @ invCovmat @ j2) @ (-normalised @ invCovmat @ j1) + ((j1.T) @ invCovmat @ j2)]
    h22 = np.exp(-normalised @ invCovmat @ (normalised.T)) @ [(normalised @ invCovmat @ j2) @ (-normalised @ invCovmat @ j2) + ((j2.T) @ invCovmat @ j2)]
    h23 = np.exp(-normalised @ invCovmat @ (normalised.T)) @ [(normalised @ invCovmat @ j2) @ (-normalised @ invCovmat @ j3) + ((j3.T) @ invCovmat @ j2)]
    h31 = np.exp(-normalised @ invCovmat @ (normalised.T)) @ [(normalised @ invCovmat @ j3) @ (-normalised @ invCovmat @ j1) + ((j1.T) @ invCovmat @ j3)]
    h32 = np.exp(-normalised @ invCovmat @ (normalised.T)) @ [(normalised @ invCovmat @ j3) @ (-normalised @ invCovmat @ j2) + ((j2.T) @ invCovmat @ j3)]
    h33 = np.exp(-normalised @ invCovmat @ (normalised.T)) @ [(normalised @ invCovmat @ j3) @ (-normalised @ invCovmat @ j3) + (normalised @ invCovmat @ secDerivative) + (j3.T) @ invCovmat @ j1]
    hessian = np.matrix([[h11[0, 0], h12[0, 0], h13[0, 0]], [h21[0, 0], h22[0, 0], h23[0, 0]], [h31[0, 0], h32[0, 0], h33[0, 0]]])

    #print(f" During {iterator} values of H12, H21, H23, H32, H13, H31 are : {h12}, {h21}, {h32}, {h23}, {h13}, {h31}")

    #print("The determinant of the hessian matrix is ", np.linalg.det(hessian))
    return hessian

def findScore(pose, point, invCov, mean):
    """
    computed the score of the point being evaluated...
    :param point: The individual point
    :param covMat: The covariance matrix from the loop up
    :param mean: mean of the points in the cell
    :return: the score for that point
    """
    #print("The pose in findScore function", pose)
    #print("The shape of the pose variable is ", pose.shape)
    point = np.matrix([point])
    #print("The point in the findscore function is : ", point.shape)
    rotMat = np.matrix([[np.cos(pose[2, 0]), np.sin(pose[2, 0])], [np.sin(pose[2, 0]), np.cos(pose[2, 0])]])
    #print("The rotational matrix in find score function is ", rotMat)
    transMat = np.matrix([[pose[0, 0]], [pose[1, 0]]])
    point = point @ rotMat + transMat.T
    #print("The transformed point with the corrected pose is : ", point)
    normalised = point - mean
    #print("The normalised point with the corrected pose is : ", normalised)
    factor = (-normalised) @ invCov @ (normalised.T)
    print("The factor is ", factor[0,0])
    exponent = np.exp(float(-factor[0,0]) / 2)
    print("The value of the exponent in the findscore function is :  ", exponent)
    return exponent

def isPositiveDefinite(hessian):
    #print("The eigen values of this hessian matrix is : ", np.linalg.eigvals(hessian))
    minEigenValue = min(np.linalg.eigvals(hessian))
    return np.all(np.linalg.eigvals(hessian) > 0), minEigenValue

def scanMatching(newPoints, covarianceDict, meanDict, currentOdomPose, cellIDs, trueOdomPose, cellCenters):
    """
    The function which performs the matching with the previous scans (covariance matrix and mean for individual cells)
    and the current scan (points in the current cell)
    :param newTranformedPoints: The new set of points from the scan
    :param fileName:  The structure which stores the covariance matrix and the mean of the points...
    :return: The final estimated pose and the score of the points
    score = 0
    ## To transform the new set of scanned points into the coordinate from of the odom frame..
    newtransformedPoints = transform2Odom(newPoints, OdomdataCurrrent, OdomdataPrevious)
    jT = np.array([[1, 0, newtransformedPoints[0]*(-np.sin(OdomdataCurrrent.rot) - newtransformedPoints[1]*np.cos(OdomdataCurrrent.rot))],
                   [0, 1, newtransformedPoints[0]*(-np.sin(OdomdataCurrrent.rot)) - newtransformedPoints[1]*np.cos(OdomdataCurrrent.rot)]])
    """
    #print("The covariance dictionary inside the scan matching function is ", covarianceDict)
    rotMat = np.array([[np.cos(currentOdomPose[2]), -np.sin(currentOdomPose[2])], [np.sin(currentOdomPose[2]), np.cos(currentOdomPose[2])]])
    transMat = np.array([[currentOdomPose[0]], [currentOdomPose[1]]])
    tempPoint = []
    guessedPoints = (rotMat @ (newPoints.T)) + transMat
    guessedPoints = guessedPoints.T
    #print("The rotational matrix is ", rotMat)
    print("The transformed points of scene 2 are ", guessedPoints)
    insideX = False
    insideY = False
    score = 0.0
    requiredCellID = 0
    pose = np.matrix([currentOdomPose]).T
    oldPose = np.matrix([currentOdomPose]).T
    #print("The first element of the 5th cell Id is : ", cellIDs[5][1][0])
    for iter in range(10):
        previousScore = score
        previousPose = pose
        gradient = np.array([[0.0], [0.0], [0.0]])
        hessian = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        for p in range(len(guessedPoints)):
            for i in range(len(cellIDs)):
                insideX = cellIDs[i][0][0] <= guessedPoints[p][0] < cellIDs[i][1][0]
                insideY = cellIDs[i][0][1] <= guessedPoints[p][1] < cellIDs[i][1][1]
                if insideY and insideX:
                    requiredCellIDIndex = i
                    break
                else:
                    continue
            #print("The cell id is ", requiredCellIDIndex)
            #print("The cells are ", cellIDs[requiredCellIDIndex])
            #print("The point is ", point)
            #print("The covariance matrix at this id is ", covarianceDict[str(requiredCellIDIndex + 1)])
            covMat = covarianceDict[str(requiredCellIDIndex + 1)]
            mean = meanDict[str(requiredCellIDIndex + 1)]
            if np.linalg.det(covMat) == 0:
                tempCount = 0
                neighbours = nearestNeighbour(requiredCellIDIndex, cellCenters)
                #covMat, newcellID = nearestNeighbour(requiredCellIDIndex, cellCenters)
                #mean = meanDict[newcellID]
                #print("The neighbours are ", neighbours)
                for neighbour in neighbours:
                    #print(covarianceDict[str(neighbour + 1)])
                    tempcov = covarianceDict[str(neighbour + 1)]
                    tempMean = meanDict[str(neighbour + 1)]
                    covMat += covarianceDict[str(neighbour + 1)]
                    mean += meanDict[str(neighbour + 1)]
                    if np.linalg.det(tempcov) != 0.0:
                        tempCount += 1
                covMat /= tempCount
                mean /= tempCount
                #print("The covariance and the mean matrix for the neighbours is : ", covMat, mean)

            invCov = np.linalg.inv(covMat)
            gradient += updateGradient(guessedPoints[p], invCov, mean, currentOdomPose, iter + 1)
            hess = updateHessian(guessedPoints[p], invCov, mean, currentOdomPose, iter + 1)
            hessian = hessian + hess
            score += findScore(previousPose, newPoints[p], invCov, mean)
        #print("The score is : ", score)
        if score < previousScore:
            score = previousScore
            pose = previousPose
            break
        hessianStatus, minEig = isPositiveDefinite(hessian)
        #print(f"After the end of the iteration {iter + 1} : ")
        #print("The hessian status is ", hessianStatus)
        if hessianStatus == False:
            hessian = makePositive(hessian)
            #print("The new hessian matrix is : ", hessian)
            #print("The eigen values of the new matrix is : ", np.linalg.eigvals(hessian))
            hessianStatus, minEig = isPositiveDefinite(hessian)
        print("The hessian is positive definite : ", hessianStatus)
        invHessian = np.linalg.inv(hessian)
        delP = invHessian @ (-gradient)
        print("The delta pose after calculation is ", delP)
        # print("The shape of the delP variable is ", delP.shape)
        # print("The shape of the pose variable is ", pose.shape)
        pose += delP
        xi = pose[0]
        yi = pose[1]
        plt.plot(xi, yi, 'bo')

    x1 = pose[0]
    y1 = pose[1]
    #print(pose)
    x2 = trueOdomPose[0]
    y2 = trueOdomPose[1]
    x3 = oldPose[0]
    y3 = oldPose[1]
    plt.plot(x1, y1, 'bo')
    plt.plot(x2, y2, 'go')
    plt.plot(x3, y3, 'ro')
    plt.show()

if __name__ == '__main__':
    detectionScene1 = np.array([[-1.0, 4.0], [-0.9, 3.8], [-1.1, 4.1], [2, 3], [2.1, 3.1], [1.9, 2.9], [1, 2], [1.1, 2.1], [0.9, 1.9]])
    truePoseScene1 = [2.0, 1.0, 0.0]
    robotPoseScene2 = [2.4, 1.6, 0.0]
    truePoseScene2 = [3.0, 2.0, 0.0]
    detectionScene2 = np.array([[0.0, 1.0], [-0.1, 1.1], [0.1, 1.2], [1, 2], [0.9, 1.9], [1.1, 2.1], [-2.0, 2.0], [-1.9, 2.1], [-2.1, 2.1]])
    weight = np.ones(len(detectionScene1))
    covarianceDictionary, meanDictionary, cellIDs, cellCenters = ndtCartesian(detectionScene1, weight, truePoseScene1)
    scanMatching(detectionScene2, covarianceDictionary, meanDictionary, robotPoseScene2, cellIDs, truePoseScene2, cellCenters)
    #print("The length of the dictionaries are ", len(covarianceDictionary), len(meanDictionary))
