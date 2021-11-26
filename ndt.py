"""
This script provides ways to find the normal distribution transform for both cartesian and polar coordinate points
The points are either in the form of (x, y) and (r, theta). Interconversion is also possible and considered.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm
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
    print(f"{xi[None,:]}")
    print(f"{yi[:,None]}")
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

def createCartisianGrids(xmax, ymax, xmin, ymin):
    """
    Function that creates a grid of points for the set of x, y points
    :return: x, y grid points
    """
    xarray = np.arange(xmin - 2, xmax + 2, 1)
    yarray = np.arange(ymin - 2, ymax + 2, 1)
    x_m, y_m = np.meshgrid(xarray, yarray)
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

def computeCovariance(points, weight):
    """
    Computes the covariance matrix for a set of points considered.
    :param points: The set of local points within a grid cell for which we need to compute the cov matrix
    :return: the dXd covariance matrix
    """
    mean = np.mean(points, axis=0)
    print("The statistical mean is ", mean)
    mean = computeMean(points, weight)
    covMat = np.zeros([mean.shape[0], mean.shape[0]])
    for point in points:
        pointTranspose = (point - mean).reshape(2, 1)
        #print("The point transpose is ", pointTranspose)
        pointNorm = (point - mean).reshape(1, 2)
        #print("The normalized point is ", pointNorm)
        covMat += np.matmul(pointTranspose, pointNorm)
    covMat /= len(points)
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
    print("The x, y in cartesian coordinates are ", x, y)

def gauss(x, y, covMat, mean):
    X = np.column_stack((x, y))
    print("The X matrix is ", X)
    normaliser = (1. / (np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covMat))))
    multiplied = np.dot((X-mean[None,...]).dot(np.linalg.inv(covMat)), (X-mean[None,...]).T)
    #return np.diag(np.exp(-1 * (multiplied)))
    return normaliser*(np.diag(np.exp(-1 * (multiplied))))

def computeMean(points, weight):
    meanX = 0.0
    meanY = 0.0
    for i in range(len(points)):
        meanX += points[i,0]*weight[i]
        meanY += points[i, 1]*weight[i]
    mean = np.array([meanX, meanX])
    mean /= len(points)
    print("Computed mean is ", mean)
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
    if count >= 3:
        cellPoints = np.array(cellPoints)
        covMat = computeCovariance(points, weight)
        cellLimits = np.array([[x_min, y_min], [x_max, y_max]])
        xCell = np.linspace(x_min, x_max, 1000)
        yCell = np.linspace(y_min, y_max, 1000)
        return covMat, xCell, yCell, count, cellPoints
    else:
        return None, None, None, None, None

def ndtCartesian(points, weight):
    #x, y = polar2Cartesian(points)
    x, y = (points[:, 0], points[:, 1])
    #print("The x and y are ", x, y)
    xmin = round(np.amin(x), 1)
    ymin = round(np.amin(y), 1)
    xmax = round(np.amax(x), 1)
    ymax = round(np.amax(y), 1)
    #print("The max and mins are ", xmin, xmax, ymin, ymax)
    x_m, y_m = createCartisianGrids(xmax, ymax, xmin, ymin)
    x_mrow = x_m[0]
    y_mcol = y_m[:, 0]
    #print("The x_m and y_m for the grid are", x_mrow, y_mcol)
    for i in range(len(x_mrow) - 1):
        for j in range(len(y_mcol) - 1):
            covMat, xCell, yCell, count, cellPoints = individualCellParameters(points, weight, x_mrow[i], y_mcol[j], x_mrow[i + 1], y_mcol[j + 1])
            if count!= None and count >= 3:
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
                #pdfValues = multivariateGaussian(cellPoints, 2, covMat, mean)
                print("The pdfValues are ", pdfValues)
                zi = griddata((x, y), pdfValues, (xi[None, :], yi[:, None]), method='cubic')
                #print("The zi's are ", zi)
                plt.contourf(xCell, yCell, zi, cmap=cm.Greys_r)
                #print("The meshgrid points are ", x_p, y_p)
                #print("The x_m, y_m are", x_m, y_m)
                plt.plot(x, y, 'o', markersize=10, c='red')
    for i in range(len(x_m)):
        for j in range(len(y_m)):
            plt.plot(x_m[i], y_m[j], 'o', markersize=5, c='magenta')

if __name__ == '__main__':
    """
    The main function for testing the values..
    """
    #points = np.array([[0.4, 0.53], [0.8, 1.05], [0.5, 1.2], [1.0, 2.0], [0.1, 0.2], [0.2, 0.2], [0.15, 0.15], [0.115, 0.1145]])
    points = np.array([[0.1, 0.9], [0.5, 0.5], [0.75, 0.9], [0.8, 0.2], [2.0, 0.5], [2.5, 0.8], [2.9, 0.9],
                       [2.1, 2.4], [2.2, 2.5], [2.5, 2.2], [2.8, 2.8], [2.4, 2.4], [4.1, 4.1], [4.2, 4.2],
                       [4.3, 4.3], [4.4, 4.4], [4.6, 4.9]])
    radius = 2.3
    #ndtPolar(points, radius)
    #plt.show()
    weight = np.ones([len(points)])
    #print("The weight matrix is ", weight)
    ndtCartesian(points, weight)
    plt.show()
