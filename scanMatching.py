import numpy as np
import matplotlib.pyplot as plt
from transformations import transform2Odom

def updateGradient(point, inverseCov, mean, pose):
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

def updateHessian(point, invCovmat, mean, pose):
    poseX = pose[0]
    poseY = pose[1]
    orient = pose[2]
    normalised = np.matrix(point - mean)
    j1 = np.array([[1, 0]]).T
    j2 = np.array([[0, 1]]).T
    j3 = np.array([[-poseX * np.sin(orient) - poseY * np.cos(orient),
                    poseX * np.cos(orient) - poseY * np.sin(orient)]]).T
    secDerivative = np.array([[-poseX*np.cos(orient) + poseY*np.cos(orient), -poseX*np.sin(orient) - poseY*np.cos(orient)]]).T
    h11 = np.exp((-normalised @ invCovmat) @ (normalised.T))@[((normalised @ invCovmat) @ j1) @ (-normalised @ invCovmat @ j1) + (j1.T) @ invCovmat @ j1]
    #print("The hessian 1 is ", h11)
    h12 = np.exp(-normalised @ invCovmat @ (normalised.T)) @ [(normalised @ invCovmat @ j1) @ (-normalised @ invCovmat @ j2) + ((j2.T) @ invCovmat @ j1)]
    h13 = np.exp(-normalised @ invCovmat @ (normalised.T)) @ [(normalised @ invCovmat @ j1) @ (-normalised @ invCovmat @ j3) + ((j3.T) @ invCovmat @ j1)]
    h21 = np.exp(-normalised @ invCovmat @ (normalised.T)) @ [(normalised @ invCovmat @ j2) @ (-normalised @ invCovmat @ j1) + ((j1.T) @ invCovmat @ j2)]
    h22 = np.exp(-normalised @ invCovmat @ (normalised.T)) @ [(normalised @ invCovmat @ j2) @ (-normalised @ invCovmat @ j2) + ((j2.T) @ invCovmat @ j2)]
    h23 = np.exp(-normalised @ invCovmat @ (normalised.T)) @ [(normalised @ invCovmat @ j2) @ (-normalised @ invCovmat @ j3) + ((j3.T) @ invCovmat @ j2)]
    h31 = np.exp(-normalised @ invCovmat @ (normalised.T)) @ [(normalised @ invCovmat @ j3) @ (-normalised @ invCovmat @ j1) + ((j1.T) @ invCovmat @ j3)]
    h32 = np.exp(-normalised @ invCovmat @ (normalised.T)) @ [(normalised @ invCovmat @ j3) @ (-normalised @ invCovmat @ j2) + ((j2.T) @ invCovmat @ j3)]
    h33 = np.exp(-normalised @ invCovmat @ (normalised.T)) @ [(normalised @ invCovmat @ j3) @ (-normalised @ invCovmat @ j3) + (normalised @ invCovmat @ secDerivative) + (j3.T) @ invCovmat @ j1]
    hessian = np.matrix([[h11[0, 0], h12[0, 0], h12[0, 0]], [h21[0, 0], h22[0, 0], h23[0, 0]], [h31[0, 0], h32[0, 0], h33[0, 0]]])
    return hessian

def findscore(point, covMat, mean):
    """
    computed the score of the point being evaluated...
    :param point: The individual point
    :param covMat: The covariance matrix from the loop up
    :param mean: mean of the points in the cell
    :return: the score for that point
    """
    factor = np.dot(np.dot(point - mean, np.linalg.inv(covMat)), (point - mean).T)
    exponent = mp.exp(-factor / 2)
    return exponent


def scanMatching(newPoints, fileName, cellSizes, x_mrow, y_mcolm):
    pass
    """
    The function whih performs the matching with the previous scans (covariance matrix and mean for individual cells)
    and the current scan (points in the current cell)
    :param newTranformedPoints: The new set of points from the scan
    :param fileName:  The structure which stores the covariance matrix and the mean of the points...
    :return: The final estimated pose and the score of the points
    
    score = 0
    ## To transform the new set of scanned points into the coordinate from of the odom frame..
    newtransformedPoints = transform2Odom(newPoints, OdomdataCurrrent, OdomdataPrevious)
    jT = np.array([[1, 0, newtransformedPoints[0]*(-np.sin(OdomdataCurrrent.rot) - newtransformedPoints[1]*np.cos(OdomdataCurrrent.rot))],
                   [0, 1, newtransformedPoints[0]*(-np.sin(OdomdataCurrrent.rot)) - newtransformedPoints[1]*np.cos(OdomdataCurrrent.rot)]])
    gradient = 0
    hessian = 0
    cellID = []
    points = []
    temp = []

    for i in range(len(x_mrow)):
        for j in range(len(y_mcolm)):
            for point in newtransformedPoints:
                insideX = x_mrow[i] <= point[0] < x_mrow[i + 1]
                insideY = y_mcolm[j] <= point[1] < y_mcolm[j + 1]
                if insideX and insideY:
                    cellID.append([x_mrow[i], y_mcolm[j], x_mrow[i + 1], y_mcolm[j + 1]])
                    temp.append(point)


    while converged:
        score = 0
        gradient = 0
        hessian = 0
        for point in newtransformedPoints:
            cellId = findCellId(point)
            covMat, mean = cellParameters(cellId)
            score += findscore(point, covMat, mean)
            gradient += updateGradient(point, jacobian, covMat, mean)
            hessian += updateHessian(point, covMat, mean, )
        delP = np.dot(np.linalg.inv(h), g)
        pold = p
        pnew = p + delP
        count += 1
        if count == 10:
            converged = True
        else:
            converged = False


    for i in range(1, cellSizes):
        length1 = 0
        length2 = 0
        while length1 + i < len(x_mrow):
            length2 = 0
            while length2 + i <len(y_mcolm):
                cellId = np.array([[x_mrow[length1], y_mcolm[length2]], [x_mrow[length1 + i], y_mcolm[length2 + i]]])
                covMat, mean = loopUp(cellID)
                inverseCov = np.linalg.inv(covMat)
                nomralisePoint = newTranformedPoints - mean
                factor = -(np.dot(nomralisePoint, inverseCov), nomralisePoint.T)
                score += np.exp(factor / 2)  ## Need to do the optimizarion technique
    """

if __name__ == '__main__':
    points = np.array([[1.2, 2.3], [4.6, 2.4], [5.6, 2.5]])
    pose = [1, 2, 45]
    covMat = np.array([[1.2, 2.3], [2.3, 1.1]])
    invCov = np.linalg.inv(covMat)
    #print("inverse covmat is ", invCov)
    mean = np.mean(points, axis=0)
    #print("Mean is ", mean)
    normalised = points - mean
    print("The normalised point is ", normalised)
    j1 = np.array([[1, 0]])
    j2 = np.array([[0, 1]])
    j3 = np.array([[-pose[0] * np.sin(pose[2]) - pose[1] * np.cos(pose[2]),
                   pose[0] * np.cos(pose[2]) - pose[1] * np.sin(pose[2])]])
    jT = np.array([j1, j2, j3])
    #for rows in jT:
    #    print(rows.T)
    firstPoint = np.array([normalised[0]])
    #print(firstPoint.shape)
    #print(invCov.shape)
    #print("Firstpoint transpose is ", firstPoint.T)
    firstfactor = np.dot(-firstPoint, invCov)
    firstfactor2 = -firstPoint @ invCov
    #print("The first factor outside the loop is ", firstfactor)
    #print("The first factor with the @ symbol is : ", firstfactor2)
    """print("The first factor outside the loop is ", firstfactor)
    secondfactor = np.dot(firstfactor, firstPoint.T)
    print("the second factor outside the loop is ", secondfactor)
    #print("First point multiplied", np.dot(), (firstPoint.T)))
    print("The exponential value outside the loop is ", np.exp(secondfactor / 2))
    firstGradientFirstFactor = np.dot(np.dot(firstPoint,invCov), j1.T)
    print("The first factor of the first gradient is ", firstGradientFirstFactor)
    firstGradientSecondFactor = firstGradientFirstFactor*np.exp(secondfactor / 2)
    print("The first gradient outside the loop is ", firstGradientSecondFactor)"""
    gradient = np.array([[0.0, 0.0, 0.0]])
    gradient = gradient.T
    hessian = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    i = 0
    for point in points:
        i += 1
        hess = updateHessian(point, invCov, mean, pose)
        #grad = updateGradient(point, invCov, mean, pose)
        print(f"The iteration {i} is ", hess)
        hessian += hess


    print(hessian)
