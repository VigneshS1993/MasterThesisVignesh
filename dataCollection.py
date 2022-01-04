"""This program will help us to parallelize the data coming from the radars and fuse them"""
import concurrent.futures
import threading
import multiprocessing as mp

import numpy as np
from serial import Serial as Se
from sys import platform
from serialDataParsing import parser_one_mmw_demo_output_packet
import time
import rawDataSynthesis
import rawDataSynthesisFINAL

def keyPress(key):
    if key == keyboard.Key.esc:
        return True
    else:
        return False

def transformation(x1, y1, z1, x2, y2, z2):
    t1 = np.matrix()

def readData(port, q):
    with Se(port, baudrate=921600, timeout=3) as ser:
        #ser.open()
        line = ser.readline()
        length = len(line)
        q.put(line)
        print(f"Length of the line is {length} and the data from the port {port} line ", line)
        ser.close()

def readSerialData(port):
    with Se(port, baudrate=921600) as serialPort:
        byteCount = serialPort.inWaiting()
        sData = serialPort.read(byteCount)
        if len(sData) == 0:
            sData = None
    serialPort.close()
    return sData

def readDataSerially(ports):
    fused = []
    byteCounts = []
    for port in ports:
        with Se(port, baudrate=921600) as serialPort:
            byteCount = serialPort.inWaiting()
            sData = serialPort.read(byteCount)
            fused.append(sData)
            byteCounts.append(byteCount)
    return fused, byteCounts
#    with serial.Serial(port, 115200, timeout=3) as ser:

#if  __name__ == '__main__':
def serialData():
    if platform == 'win32':
        #dataPorts = ['COM12', 'COM10']
        dataPorts = ['COM10', 'COM12']
        #dataPorts = ['COM14']
        configPorts = ['COM11', 'COM13']
    elif platform == 'linux':
        dataPorts = ['/usb/..', '/usb/..']
        configPorts = ['/usb/..', '/usb/..']
    configFiles = [r'D:\Master Thesis\Config_files_for_testing\Optimal\xwr68xx_AOP_profile_2021_12_01T11_52_53_572.cfg', r'D:\Master Thesis\Config_files_for_testing\Optimal\xwr68xx_AOP_profile_2021_12_01T11_52_53_572.cfg']
    configFiles = [r'D:\Master Thesis\Config_files_for_testing\Optimal\xwr68xx_AOP_profile_2021_12_01T16_44_38_461.cfg', r'D:\Master Thesis\Config_files_for_testing\Optimal\xwr68xx_AOP_profile_2021_12_01T16_44_38_461.cfg']
    configFiles = [r'D:\Master Thesis\Config_files_for_testing\Optimal\xwr68xx_AOP_profile_2021_12_03T20_14_13_130.cfg', r'D:\Master Thesis\Config_files_for_testing\Optimal\xwr68xx_AOP_profile_2021_12_03T20_14_13_130.cfg']
    configFiles = [r'D:\Master Thesis\Config_files_for_testing\Optimal\xwr68xx_AOP_profile_2021_12_03T20_31_57_911.cfg', r'D:\Master Thesis\Config_files_for_testing\Optimal\xwr68xx_AOP_profile_2021_12_03T20_31_57_911.cfg']
    configFiles = [r'D:\Master Thesis\Config_files_for_testing\Optimal\xwr68xx_AOP_profile_2021_12_03T20_45_31_915.cfg', r'D:\Master Thesis\Config_files_for_testing\Optimal\xwr68xx_AOP_profile_2021_12_03T20_45_31_915.cfg']
    configFiles = [r'D:\Master Thesis\Config_files_for_testing\Optimal\xwr68xx_AOP_profile_2021_12_06T15_54_48_642.cfg', r'D:\Master Thesis\Config_files_for_testing\Optimal\xwr68xx_AOP_profile_2021_12_06T15_54_48_642.cfg']
    configFiles = [r'D:\Master Thesis\Config_files_for_testing\Optimal\xwr68xx_AOP_profile_2021_12_14T16_11_26_053.cfg', r'D:\Master Thesis\Config_files_for_testing\Optimal\xwr68xx_AOP_profile_2021_12_14T16_11_26_053.cfg']

    count = 0
    try:
        #print("Inside the try block..")
        start = time.perf_counter()
        numDetectedObj = 0
        dataFrame, byteCounts = readDataSerially(dataPorts)
        #print("The dataFrame is ", dataFrame)
        objects = []
        if dataFrame:
            #print("The fused data is ", dataFrame)
            for i in range(len(dataFrame)):
                if len(dataFrame[i]) > 0:
                    #print("Inside the serial data function..")
                    configParameters = rawDataSynthesisFINAL.parseConfigFile(configFiles[i])
                    detObj, dataOK = rawDataSynthesisFINAL.readAndParseData(dataFrame[i], configParameters)
                    if dataOK:
                        portNumber = i + 1
                        #print("The port number : ", portNumber)
                        count += 1
                        print("The detected objects are ", detObj)
                        #print("The count of the run is : ", count)
                        objectsTuple = (detObj, portNumber)
                        objects.append(objectsTuple)
        stop = time.perf_counter()
        #print("Time taken for the run is ", (stop - start))
        #print("The total number of detected objects are ", numDetectedObj)
        #print(f"The dataFrame and byteCounts are  {dataFrame}, {byteCounts}")
        return objects
    except KeyboardInterrupt:
        exit()