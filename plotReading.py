import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm
import dataCollection
import rawDataSynthesisFINAL
from time import sleep
import transformations
import ndt
from matplotlib.animation import FuncAnimation
from otherLibraries import errorEllipse

def animate():
    print("Inside the animate function")
    pointsXY = ndt.dataConcatenation()
    pointsXY = np.array(pointsXY)
    plt.cla()
    plt.plot(pointsXY[:, 0], pointsXY[:, 1])
    plt.legend(loc='upper left')
    plt.tight_layout()

if __name__ == '__main__':
    configFile = r'D:\Master Thesis\Config_files_for_testing\Optimal\xwr68xx_AOP_profile_2021_12_16T21_57_41_913_exported.cfg'
    configFile = r'D:\Master Thesis\Config_files_for_testing\Optimal\xwr68xx_AOP_profile_2021_12_27T11_31_31_570_27_12_2021.cfg'
    configPorts = ['COM11', 'COM13']
    #configPorts = ['COM9']
    count = 0
    #rawDataSynthesisFINAL.sensorConfiguration(configFile, configPorts)
    #print("Configuration done..")
    x_min = -2.0
    x_max = 2.0
    y_min = -0.5
    y_max = 3.5
    while True:
        pointsXY = ndt.dataConcatenation()
        #print("The points are in the main function ", pointsXY)
        pointsXY, weights = ndt.identicalPointsRemoval(pointsXY)
        pointsXY = np.array(pointsXY)
        #print("The points are in the main function as a numpy array : ", pointsXY)
        max1 = round(max(pointsXY[:, 0]), 2)
        max2 = round(max(pointsXY[:, 1]), 2)
        min1 = round(min(pointsXY[:, 0]), 2)
        min2 = round(min(pointsXY[:, 1]), 2)
        x_mrow = np.linspace(x_min, x_max, 51)
        x_mrow = x_mrow.round(decimals=2)
        y_mcol = np.linspace(y_min, y_max, 51)
        y_mcol = y_mcol.round(decimals=2)
        weights = np.ones(len(pointsXY))
        #print("The x rows and y columns are ", x_mrow, y_mcol)
        ndt.ndtCartesian(pointsXY, weights)
        plt.xlim([-1.5, 1.5])
        plt.ylim([-0.5, 2.5])
        plt.grid(color='black', linestyle='--',linewidth=0.5)
        plt.xticks(x_mrow)
        plt.yticks(y_mcol)
        plt.plot(pointsXY[:, 0], pointsXY[:, 1], 'o', markersize=1, c='red')
        plt.pause(0.01)
        plt.show()
        count += 1
    #print("After the ani function..")


