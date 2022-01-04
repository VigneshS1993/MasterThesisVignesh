import cv2 as oCV
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

def errorEllipse(mean, covMat, sigma):
    """
    getting the parameters of an ellipse from the given set of points and covarinace matrix
    :param pointsXY: The set of points, x and y
    :param mean: mean of the points
    :param covMat: The covariance matrix of the points
    :return: the ellipse object
    """
    # get the eigen values form the covmatrix
    eigValues, eigVectors = np.linalg.eigh(covMat)
    #Sort the eigen values in descending order to compute the major axis values
    order = eigValues.argsort()[::-1]
    eigenValues, eigenVectors = eigValues[order], eigVectors[:, order]

    # compute the angle of the major axis with respect to the x axis

    vx, vy = eigenVectors[:, 0][0], eigenVectors[:, 0][1]
    #print("Vx, and vy are ", vx, vy)
    theta = np.arctan2(vy, vx)
    print("The eigValues are ", eigValues)
    # width and height of the ellipse
    width, height = 2 * sigma * np.sqrt(eigValues)

    return Ellipse(xy=mean, width=width, height=height, angle=np.degrees(theta), fc='black')

if __name__ == '__main__':
    x = np.random.normal(0.5, 0.1, 100)
    y = np.random.normal(0.5, 0.1, 100)
    xyPoints = np.array([[x[i], y[i]] for i in range(len(x))])
    mean = np.mean(xyPoints, axis=0)
    print("The mean is ", mean)
    covMat = np.cov(xyPoints.T)
    print("The covariance matrix is ", covMat)
    ellipse = errorEllipse(mean, covMat, sigma=2)
    #plt.figure()
    ax = plt.gca()
    ax.add_patch(ellipse)
    plt.plot()
    plt.plot(x, y, 'o', markersize=3, c='r')
    plt.show()




