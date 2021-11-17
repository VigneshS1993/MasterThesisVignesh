"""
This script provides ways to find the normal distribution transform for both cartesian and polar coordinate points
The points are either in the form of (x, y) and (r, theta). Interconversion is also possible and considered.
"""


import numpy as np
import matplotlib.pyplot as plt

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

def computeCovariance(points):
    """
    Computes the covariance matrix for a set of points considered.
    :param points: The set of local points within a grid cell for which we need to compute the cov matrix
    :return: the dXd covariance matrix
    """
    mean = np.mean(points, axis=0)
    #print("The mean is ", mean)
    covMat = np.zeros([mean.shape[0], mean.shape[0]])
    #covMat = np.array([])
    for point in points:
        pointTranspose = (point - mean).reshape(2, 1)
        #print("The point transpose is ", pointTranspose)
        pointNorm = (point - mean).reshape(1, 2)
        #print("The normalized point is ", pointNorm)
        covMat += np.matmul(pointTranspose, pointNorm)
    covMat /= len(points)
    return covMat

def scapeCovMat(covMat, factor):
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
        #print("The point Normal is {}, The inverse of covMat is {}, the point transpose is {}".format(pointNorm, Sigma_inv, pointTranspose))
        #print("The size of the above variables are {}, {}, {}".format(pointNorm.shape, Sigma_inv.shape, pointTranspose.shape))
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
    factor = (np.exp(-(np.matmul(np.matmul(x_normal, np.linalg.inv(covariance)), x_norm_trans))))
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

if __name__ == '__main__':
    """
    The main function for testing the values..
    """
    points = np.array([[0.4, 0.53], [0.8, 1.05], [0.5, 1.2], [1.0, 2.0]])
    radius = 2.3
    radius, theta = createValues(radius)
    radius, theta = createMeshGrid(radius, theta)
    covMat = computeCovariance(points)
    mean = np.mean(points, axis=0)
    dim = 2
    pdfValues = probabDenFun(mean, covMat, points, dim)
    #print("The pdf values are ", pdfValues)
    #print("The shape of pdf array is ", pdfValues.shape)
    #r = radius[0]
    #t = theta[:, 0]
    #print(r)
    #print(t)
    x_p, y_p = np.meshgrid(points[:, 0], points[:, 1])
    #print(x_p, y_p)
    levels = [0.3, 0.4, 0.5]
    plt.plot(points[:, 0], points[:, 1], 'o', markersize=10)
    plt.contour(x_p, y_p, pdfValues, levels=levels)
    x, y = createMGPolar2Cartesian(points)
    print("The x, y in cartesian coordinates are ", x, y)
    plt.show()