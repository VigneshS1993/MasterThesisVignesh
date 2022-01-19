"""
This script provides ways to find the normal distribution transform for both cartesian and polar coordinate points
The points are either in the form of (x, y) and (r, theta). Interconversion is also possible and considered.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm
import dataCollection
import rawDataSynthesisFINAL
from time import sleep
from sklearn.mixture import GaussianMixture
from otherLibraries import errorEllipse

import transformations


def createValues(radius, thetaResol=np.pi/3):
    """
    Used to create random values for the point generation in the form of (r, theta)
    :param radius: the maximum radius limit
    :param thetaResol: the resolution of the theta
    :return: returns the equally spaced array of radius and theta.
    """
    r = np.arange(0, radius, 0.5)
    theta = np.arange(0, 2*np.pi, thetaResol)
    return r, theta

def plot_countour(localPoints, pdf, xlowerEdge, ylowerEdge, xupperEdge, yupperEdge):
    # define grid.
    x = localPoints[:,0]
    y = localPoints[:,1]
    xi = np.linspace(xlowerEdge, xupperEdge, 100)
    yi = np.linspace(yLowerEdge, yupperEdge, 100)
    #yi = np.array([-0.83291747, -0.33291747, 0.16708253, 0.66708253])
    #xi = np.array([-0.29778666, 0.20221334, 0.70221334, 1.20221334])
    ## grid the data.
    zi = griddata((x, y), pdf, (xi[None,:], yi[:,None]), method='nearest')
    levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    #print(f"{xi[None,:]}")
    #print(f"{yi[:,None]}")
    # contour the gridded data, plotting dots at the randomly spaced data points.
    #CS = plt.contour(xi,yi,zi,len(levels),linewidths=0.5,colors='k', levels=levels)
    #CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
    CS = plt.contourf(xi,yi,zi,len(levels),cmap=cm.Greys_r, levels=levels)
    plt.colorbar() # draw colorbar
    # plot data points.
    # plt.scatter(x, y, marker='o', c='b', s=5)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title('griddata test (%d points)' % npts)
    plt.show()

def createCartisianGrids(xmax, ymax, xmin, ymin, cellSize):
    """
    Function that creates a grid of points for the set of x, y points
    :return: x, y grid points
    """
    xarray = np.arange(xmin, xmax, cellSize)
    xarray = xarray.round(decimals=2)
    yarray = np.arange(ymin, ymax, cellSize)
    yarray = yarray.round(decimals=2)
    x_m, y_m = np.meshgrid(xarray, yarray)
    #print("The x vector and y vectors are ", x_m, y_m)
    #plt.plot(x_m, y_m)
    #plt.grid()
    return x_m, y_m

def createMeshGrid(r, theta):
    """ Function to create mesh grid for radius and theta"""

    radius, theta = np.meshgrid(r, theta)
    return radius, theta

def createMGPolar2Cartesian(points):
    """ create cartesian meshgrid for polar to cartesian points"""

    x, y = polar2Cartesian(points)
    x, y = np.meshgrid(x, y)
    return x, y


def polar2Cartesian(points):
    """ can be used to convert the r, theta, elevation into x, y and perform the normal distribution in cartesian coordinates"""

    x = [points[i, 0] * np.sin(points[i, 1]) for i in range(len(points))]
    y = [points[1, 0] * np.cos(points[i, 1]) for i in range(len(points))]
    x = np.array(x)
    y = np.array(y)
    return x, y

def check4Eigen(covMat):
    """
    Function to check if the largest eigen value is atleast 0.001 times the smallest, if not make them like that..
    :param covMat: the covariance matrix to change
    :return: the transformed covMat
    """
    eigenValues, eigenVectors = np.linalg.eigh(covMat)
    u, sigma, ut = np.linalg.svd(covMat)
    #print("u, sigma and ut are : ", u, sigma, ut)
    if sigma[0] / sigma[1] < 0.001:
        sigma[0] = sigma[1] * 0.001
    sigma = np.array([[sigma[0], 0], [0, sigma[1]]])
    covmat = np.dot(u, np.dot(sigma, ut))

    return covmat

def computeCovariance(points, weight):
    """
    Computes the covariance matrix for a set of points considered.
    :param points: The set of local points within a grid cell for which we need to compute the cov matrix
    :return: the dXd covariance matrix
    """
    #mean = np.mean(points, axis=0)
    #print("The statistical mean is ", mean)
    mean = computeMean(points, weight)
    covMat = np.zeros([mean.shape[0], mean.shape[0]])
    for point in points:
        pointTranspose = (point - mean).reshape(2, 1)
        #print("The point transpose is ", pointTranspose)
        pointNorm = (point - mean).reshape(1, 2)
        #print("The normalized point is ", pointNorm)
        covMat += np.matmul(pointTranspose, pointNorm)
    covMat /= len(points)
    #print("Before checking eigen values, covariance matrix is ", covMat)
    covMat = check4Eigen(covMat)
    #print("After checking eigen values, covariance matrix is ", covMat)
    return covMat

def scaleCovMat(covMat, factor):
    """
    Used to scale the existing covariance matrix with a given scalar factor
    :param covMat: the covariance matrix itself
    :param factor: the scalar factor
    :return:
    """
    u, s, ut = np.linald.svd(covMat)
    sNew = factor*np.array([s[0], 0], [0, s[1]])
    newCov = np.matMul(u, sNew, ut)
    return newCov

def pdf(covMat, points):
    """ This is a trial version and did not give any good results..
    performs the multivariance normal distribution in the form of the pdf
    :param covMat: The covariance matrix
    :param points: the array of the points considered
    :return: the continuous pdf values
    """
    mean = np.mean(points, axis=0)
    Sigma_det = np.linalg.det(covMat)
    if Sigma_det > 0:
        n = mean.shape[0]
        Sigma_inv = np.linalg.inv(covMat)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        fac = np.zeros([n, n])
        #print("The point difference is ", points - mean)
        pointsTranspose = (points - mean).transpose()
        pointsNorm = (points - mean)
        temp = np.matmul((pointsNorm), Sigma_inv)
        fac = np.matmul(temp, pointsTranspose)
    return (np.exp(-fac / 2) / n)

def multivariateGaussian(x, dim, covariance, mean):
    """
    function for computing the multivariate gaussian surface
    """
    x_normal = np.matrix(x - mean)
    #print("The normalized x matrix is ", x_normal)
    mean = np.matrix(mean)
    #print("The mean inside the multi variate gaussian function is ", mean)
    normaliser = (1. / (np.sqrt((2*np.pi)**dim*np.linalg.det(covariance))))
    #print("The value of the normaliser is ", normaliser)
    coavriance = np.matrix(covariance)
    x_norm_trans = x_normal.transpose()
    factor = (np.exp(-(np.matmul(np.matmul(x_normal, np.linalg.inv(covariance)), x_norm_trans)) / 2))
    #factor = (np.exp(-(np.linalg.solve(covariance, x_normal).T.dot(x_normal)) / 2))
    return (normaliser*factor)

def probabDenFun(mean, covariance, points, dim):
    """
    The final probabilits density function.
    :param mean: mean of the points
    :param covariance: covariance matrix
    :param points: the set of points
    :param dim: dimensions of the matrix
    :return: the pdf
    """
    (n, m) = (points.shape[0], points.shape[0])
    # print("The shapes n, m are ", n, m)
    x, y = np.meshgrid(points[:, 0], points[:, 1])
    # print("After mesh grid formation the x, y are ", x, y)
    # mean = np.matrix()
    # print("The mean inside the probabDenFun is ", mean)
    pdf = np.zeros((n, m))
    dim = 2
    for i in range(n):
        for j in range(m):
            pdf[i, j] = multivariateGaussian([x[i, j], y[j, i]], dim, covariance, mean)
            # print("The individual pdf values are :", pdf[i, j])

    return pdf

def cellPDF(radius, theta, points):
    """
    function to compute the individual cell parameters..
    :param radius: radius array
    :param theta: theta array
    :param points: set of points
    :return: the final cell probability density function.
    """
    for i in range(len(radius) - 1):
        for j in range(len(theta) - 1):
            count = 0
            localPoints = []
            for point in points:
                inbetweenR = radius[i] <= point[0] < radius[i + 1]
                inbetweenTh = theta[j] <= point[1] < theta[j + 1]
                if inbetweenR and inbetweenTh:
                    count += 1
                    localPoints.append(point[0], point[1])
            localPoints = np.array(localPoints)
            if count >= 2:
                covMat = computeCovariance(localPoints)


def ndtPolar(points, radius):
    radius, theta = createValues(radius)
    radius, theta = createMeshGrid(radius, theta)
    covMat = computeCovariance(points)
    mean = np.mean(points, axis=0)
    dim = 2
    pdfValues = probabDenFun(mean, covMat, points, dim)
    r_p, t_p = np.meshgrid(points[:, 0], points[:, 1])
    levels = [0.3, 0.4, 0.5]
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax = plt.subplot(111, polar=True)
    ax.plot(points[:, 0], points[:, 1], 'o', markersize=5)
    ax.contourf(r_p, t_p, pdfValues)  # , levels=levels)
    x, y = createMGPolar2Cartesian(points)
    #print("The x, y in cartesian coordinates are ", x, y)

def gauss(x, y, covMat, mean):
    X = np.column_stack((x, y))
    #print("The X matrix is ", X)
    normaliser = (1. / (np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covMat))))
    multiplied = np.dot((X-mean[None,...]).dot(np.linalg.inv(covMat)), (X-mean[None,...]).T)
    print("The multiplied variable in the pdf expression is : ", multiplied)
    #return np.diag(np.exp(-1 * (multiplied)))

    pdf = normaliser*(np.diag(np.exp(-1 * (multiplied / 2))))
    #pdf = normaliser*(np.exp(-1 * (multiplied)))
    return pdf

def computeMean(points, weight):
    meanX = 0.0
    meanY = 0.0
    #print("The points in the compute mean function is ", points)
    for i in range(len(points)):
        meanX += points[i, 0]*weight[i]
        meanY += points[i, 1]*weight[i]
    mean = np.array([meanX, meanX])
    mean /= len(points)
    #print("Computed mean is ", mean)
    return mean
def individualCellParameters(points, weight, x_min, y_min, x_max, y_max):
    cellPoints = []
    cellLimits = []
    count = 0
    #print("The points are ", points)
    #print("The xmin and xmax are", x_min, x_max)
    #print("The ymin and ymax are", y_min, y_max)
    for point in points:
        inbetweenX = x_min <= point[0] < x_max
        inbetweenY = y_min <= point[1] < y_max
        if inbetweenY and inbetweenX:
            cellPoints.append([point[0], point[1]])
            count += 1
    covMat = np.zeros([2, 2])
    if count >= 1:
        cellPoints = np.array(cellPoints)
        covMat = computeCovariance(points, weight)
        cellLimits = np.array([[x_min, y_min], [x_max, y_max]])
        xCell = np.linspace(x_min, x_max, 50)
        yCell = np.linspace(y_min, y_max, 50)
    else:
        covMat = np.array([[0.0, 0.0], [0.0, 0.0]])
        cellPoints = np.array([[0.0, 0.0], [0.0, 0.0]])
        xCell = 0.0
        yCell = 0.0
    return covMat, xCell, yCell, count, cellPoints

def identicalPointsRemoval(cellPoints):
    weight = []
    count1 = 0
    count2 = 0
    count = 1
    """for i in range(len(cellPoints)):
        print(f"The {i+1}th point is {cellPoints[i]}")"""

    while count1 < len(cellPoints):
        count2 = count1 + 1
        while count2 < len(cellPoints):
            if (cellPoints[count1][0] == cellPoints[count2][0] and cellPoints[count1][1] == cellPoints[count2][1]):
                #print("Identical points found!!!!")
                count += 1
                cellPoints.pop(count2)
            count2 += 1
        weight.append(count)
        count1 += 1
        count = 1

    cellPoints = np.array(cellPoints)
    #print("The final list of weights and points are", weight, cellPoints)
    return cellPoints, weight

def ndtCartesian(points, weight, robotFrame):
    #x, y = polar2Cartesian(points)
    rotMat = np.array([[np.cos(robotFrame[2]), -np.sin(robotFrame[2])], [np.sin(robotFrame[2]), np.cos(robotFrame[2])]])
    transMat = np.array([[robotFrame[0]], [robotFrame[1]]])
    points = rotMat @ points.T + transMat
    points = points.T
    print("The transformed points of scene 1 are : ", points)
    x, y = (points[:, 0], points[:, 1])
    gmm = GaussianMixture(n_components=2)
    #print("The x and y are ", x, y)
    xmin = -1.0
    ymin = -1.0
    xmax = 6.0
    ymax = 6.0
    #ymin = sround(np.amin(y), 1)
    #xmax = round(np.amax(x), 1)
    #ymax = round(np.amax(y), 1)
    #print("The max and mins are ", xmin, xmax, ymin, ymax)
    #cellSizes = np.array([0.25, 0.5])#, 0.2, 0.25, 0.3]) only two cell sizes
    cellSizes = np.array([0.5])
    x_m, y_m = createCartisianGrids(xmax, ymax, xmin, ymin, 1.0)
    x_mrow = x_m[0]
    y_mcol = y_m[:, 0]
    #print("The x_m and y_m are", x_mrow, y_mcol)
    covarianceDictionary = {}
    meanDictionary = {}
    covDict = {}
    meanDict = {}
    cellCenterDict = {}
    cellCenters = []
    cellIDs = []
    #print(f"The lengths of x and y are {len(x_mrow)} and {len(y_mcol)}")
    cellNumber = 0
    cellTypeCount = 0
    for i in range(1, len(cellSizes) + 1):
        #print(f"The cellSize array {i}")
        length1 = 0
        length2 = 0
        cellTypeCount = 0
        #print(f"Computing for cell size of {i + 1}")
        while length1 + i < len(x_mrow):
            length2 = 0
            #print(f"X while loop {length1}")
            while length2 + i < len(y_mcol):
                cellNumber += 1
                cellTypeCount += 1
                covMat, xCell, yCell, count, cellPoints = individualCellParameters(points, weight,
                                                                                   x_mrow[length1], y_mcol[length2],
                                                                                   x_mrow[length1 + i], y_mcol[length2 + i])
                #print("The collected cell points are ", cellPoints)
                mean = computeMean(cellPoints, weight)
                #print("After computation the covariance matrix is ", covMat)
                covarianceDictionary[str(cellNumber)] = covMat
                meanDictionary[str(cellNumber)] = mean
                cellID = [[x_mrow[length1], y_mcol[length2]], [x_mrow[length1 + i], y_mcol[length2 + i]]]
                cellIDs.append(cellID)
                cellCenter = [(x_mrow[length1] + x_mrow[length1 + i]) / 2, (y_mcol[length2 + i] + y_mcol[length2]) / 2]
                #cellCenter = np.array([cellCenter])
                cellCenters.append(cellCenter)
                dictID = str(cellCenter[0]) + '-' + str(cellCenter[1])
                covDict[dictID] = covMat
                meanDict[dictID] = mean
                #print(f"Y while loop {length2}")
                #print(f"The cell parameters for the cell {x_mrow[length1]}, {y_mcol[length2]} and {x_mrow[length1 + i]}, {y_mcol[length2 + i]}")
                """if count != None and count >= 2:
                    print("The weight is : ", weight)
                    dim = 2
                    #pdfValues = probabDenFun(mean, covMat, cellPoints, dim)  # This function can also be used for usage in interpolation we need to have the pdfValues[0]
                    #pdfValues = pdfValues[0] / sum(pdfValues[0])
                    #print("The normalized pdf values computed by other function is : ", pdfValues)
                    ellipse = errorEllipse(mean, covMat, sigma=3)
                    ax = plt.gca()
                    ax.add_patch(ellipse)
                    plt.plot()
                    ## Using gmm model"""
                    #gmm.fit(cellPoints)
                    #x = cellPoints[:, 0]
                    #y = cellPoints[:, 1]
                    #x_p, y_p = np.meshgrid(x, y)
                    #x_matrix, y_matrix = np.meshgrid(xCell, yCell)
                    #XY = np.array([x_matrix.ravel(), y_matrix.ravel()]).T
                    #xi = x_matrix[0]
                    #yi = y_matrix[:, 0]
                    #Z = gmm.score_samples(XY)
                    #print("The mean from gaussian mixture model are : ", gmm.means_)
                    #print("The covariances from the gaussian mixture model is : ", gmm.covariances_)
                    #print("My computed mean is : ", mean)
                    #print("My covariance matrix is : ", covMat)
                    #levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                    #print("The maximum value of Z before reshaping is ", max(Z))
                    #Z = Z.reshape((50, 50))
                    #print(Z)
                    #print(Z.shape)
                    #plt.contour(x_matrix, y_matrix, Z, levels=levels, cmap=cm.binary)"""
                    #Using gmm model is done
                    #pdfValues = gauss(x, y, covMat, mean)
                    # print("The different cell points are ", cellPoints)
                    # pdfValues = multivariateGaussian(cellPoints, 2, covMat, mean)
                    #print("The pdfValues by gauss function is  ", pdfValues)
                    #zi = griddata((x, y), pdfValues, (xi[None, :], yi[:, None]), method='cubic')
                    #print("The zi's are ", zi)
                    #plt.contourf(xCell, yCell, zi, cmap=cm.binary)
                length2 += i
            #print("The length 2 is ", length2)
            length1 += i
        #print("The length 1 is ", length1)
        #print(f" For cell type {i} the number of cells is {cellTypeCount}")
    #print("The length of the covariance dictionary is ", len(covarianceDictionary))
    #print("The length of the mean dictionary is ", len(meanDictionary))
    #print("The total number of cells crossed is : ", cellNumber)
    cellIDs = np.array(cellIDs)
    cellCenters = np.array(cellCenters)

    """for i in range(len(x_mrow) - 1):
        for j in range(len(y_mcol) - 1):
            covMat, xCell, yCell, count, cellPoints = individualCellParameters(points, weight, x_mrow[i], y_mcol[j], x_mrow[i + 1], y_mcol[j + 1])
            if count!= None and count >= 2:
                mean = computeMean(cellPoints, weight)
                dim = 2
                #pdfValues = probabDenFun(mean, covMat, cellPoints, dim)  # This function can also be used for usage in interpolation we need to have the pdfValues[0]
                x = cellPoints[:, 0]
                y = cellPoints[:, 1]
                x_p, y_p = np.meshgrid(x, y)
                x_matrix, y_matrix = np.meshgrid(xCell, yCell)
                xi = x_matrix[0]
                yi = y_matrix[:, 0]
                pdfValues = gauss(x, y, covMat, mean)
                #print("The different cell points are ", cellPoints)
                #pdfValues = multivariateGaussian(cellPoints, 2, covMat, mean)
                #print("The pdfValues are ", pdfValues)
                zi = griddata((x, y), pdfValues, (xi[None, :], yi[:, None]), method='cubic')
                #print("The zi's are ", zi)
                plt.contourf(xCell, yCell, zi, cmap=cm.binary)"""
    #print("The covariance, mean dictionary and cellIds are", covarianceDictionary, meanDictionary, cellIDs)
    #print("The 22 key value in the covariance dictionary is ", covarianceDictionary['22'])
    #return covDict, meanDict, cellIDs, cellCenterDict
    return covarianceDictionary, meanDictionary, cellIDs, cellCenters

def ndtPolar(points, weight):
    range, azimuth = (points[:, 0], points[:, 1])
    rangeMin = 0.0
    azimuthMin = 0.0
    rangeMax = round(np.amax(range), 1)
    azimuthMax = 2*np.pi
    r, theta = createMeshGrid(range, azimuth)
    r_mrow = r[0]
    a_mcol = theta[:, 0]
    for i in range(len(r_mrow) - 1):
        for j in range(len(a_mcol) - 1):
            covMat, rCell, aCell, count, cellPoints = individualCellParameters(points, weight, r_mrow[i], a_mcol[j], r_mrow[i + 1], a_mcol[j + 1])
            if count != None and count >= 3:
                mean = computeMean(cellPoints, weight)
                dim = 2
                rangeVals = cellPoints[:, 0]
                azimuthVals = cellPoints[:, 1]
                r_p, azimuth_p = np.meshgrid(rangeVals, azimuthVals)
                r_matrix, a_matrix = np.meshgrid(rCell, aCell)
                ri = r_matrix[0]
                ti = a_matrix[:, 0]
                pdfValues = gauss(rangeVals, azimuthVals, covMat, mean)
                # pdfValues = multivariateGaussian(cellPoints, 2, covMat, mean)
                print("The pdfValues are ", pdfValues)
                zi = griddata((rangeVals, azimuthVals), pdfValues, (ri[None, :], ti[:, None]), method='cubic')
                ax = plt.subplot(111, polar=True)
                #ax.plot(r_matrix, a_matrix, 'o')
                ax.plot(points[:, 1], points[:, 0], 'o', markersize=10)
                plt.contourf(rCell, aCell, zi, cmap=cm.Greys_r)


def dataConcatenation():
    count = 0
    validCount = 0
    x = np.array([])
    y = np.array([])
    z = np.array([])
    rangeVal = np.array([])
    azimuthVal = np.array([])
    elevationVal = np.array([])
    #while validCount <= 2:
    #    while count <= 3:
    #print("We are in ndt data concatenation file..")
    points = np.array([])
    pointsXY = np.array([])
    count = 0
    while count == 0:
        objects = dataCollection.serialData()
        #if len(objects):
        #    print("The data is from port : ", objects[1])
        if objects:
            for object in objects:
                if object:
                    points = np.column_stack((object[0]["x"], object[0]["y"], object[0]["z"]))
                    #print("The points before transformation are : ", points)
                    transformedX, transformedY = transformations.transform(points, object[1])
                    x = np.append(x, transformedX)
                    y = np.append(y, transformedY)
                    print("The data is from port : ", object[1])
                    count += 1
            pointsXY = np.column_stack([x, y])
        #print("The points in XY coordinate system is ", pointsXY)
        #pointsRA = np.column_stack([rangeVal, azimuthVal])
        validCount += 1
        #print("The column stacked set of points are ", np.column_stack([x, y]))
        #pointsXY = np.column_stack([x, y])
        #print("The points in data concatenation function is ", pointsXY)
        pointsXY = list(pointsXY) ## converitng into list for checking the duplicate removal algorithm
    if len(pointsXY) > 0:
        return pointsXY
    else:
        return None

if __name__ == '__main__':
    """
    The main function for testing the values..
    """
    configFile = r'D:\Master Thesis\Config_files_for_testing\Optimal\xwr68xx_AOP_profile_2021_12_16T21_57_41_913_exported.cfg'
    configPorts = ['COM11', 'COM13']
    #rawDataSynthesisFINAL.sensorConfiguration(configFile, configPorts)
    #sleep(5.0)
    #pointsXY = dataConcatenation()    ## should be data concatenation and transform next to next..



    #print("The final set of points are ", pointsXY)
    #radius = 4.0
    # print("The weight matrix is ", weight)
    #print("The final set of points before computation is : ", pointsXY)
    #ndtCartesian(pointsXY, weight)
    #pointsXY, weights = identicalPointsRemoval(pointsXY)
    pointsXY = np.array([[0.559, 1.088], [0.308, 1.247], [0.494, 1.165], [-1.022, 2.12],  [0.564, 1.13],
                         [0.412, 1.143], [0.589, 1.059], [0.432, 1.044], [0.389,  0.92]])
    print("The points are ", pointsXY)
    weight = np.ones([len(pointsXY)])
    plt.plot(pointsXY[:, 0], pointsXY[:, 1], 'o', c='red')
    ndtCartesian(pointsXY, weight)
    plt.show()
