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
    configPorts = ['COM11', 'COM13']
    count = 0
    rawDataSynthesisFINAL.sensorConfiguration(configFile, configPorts)
    #print("Before the ani function..")
    #ani = FuncAnimation(plt.gcf(), animate, interval=2000)
    x_min = -2.0
    x_max = 2.0
    y_min = -0.5
    y_max = 3.5
    while count < 10:
        pointsXY = ndt.dataConcatenation()
        pointsXY, weights = ndt.identicalPointsRemoval(pointsXY)
        pointsXY = np.array(pointsXY)
        #print("The points are in the main function ", pointsXY)
        max1 = round(max(pointsXY[:, 0]), 2)
        max2 = round(max(pointsXY[:, 1]), 2)
        min1 = round(min(pointsXY[:, 0]), 2)
        min2 = round(min(pointsXY[:, 1]), 2)
        """x_max = max1 if max1 < x_max else x_max
        x_min = min1 if min1 > x_max else x_min
        y_max = max2 if max2 < y_max else y_max
        x_min = min2 if min2 > y_max else y_min
        #plt.cla()"""
        x_mrow = np.linspace(x_min, x_max, 51)
        x_mrow = x_mrow.round(decimals=2)
        y_mcol = np.linspace(y_min, y_max, 51)
        y_mcol = y_mcol.round(decimals=2)
        weights = np.ones(len(pointsXY))
        #print("The x rows and y columns are ", x_mrow, y_mcol)
        ndt.ndtCartesian(pointsXY, weights)
        plt.xlim([-2.0, 2.0])
        plt.ylim([-0.5, 3.5])
        plt.grid(color='black', linestyle='--',linewidth=0.5)
        plt.xticks(x_mrow)
        plt.yticks(y_mcol)
        plt.plot(pointsXY[:, 0], pointsXY[:, 1], 'o', markersize=1, c='red')
        plt.pause(0.01)
        plt.show()
        count += 1
    #print("After the ani function..")


