import numpy as np
import matplotlib.pyplot as plt
import rawDataSynthesisFINAL
import dataCollection
from serial import Serial as Se
from datetime import datetime
from serialDataParsing import parser_one_mmw_demo_output_packet

def readserialData(port):
    with Se(port, baudrate=921600) as serialPort:
        #byteCount = serialPort.inWaiting()
        byteCount = 1024
        sData = serialPort.read(byteCount)
        serialPort.close()
    return sData, byteCount

if __name__ == '__main__':
    print("We are in get data file")
    configFile = r'D:\Master Thesis\Config_files_for_testing\Optimal\profile_2022_01_13T12_35_27_476_13_01_2022_1.cfg'
    configPorts = ['COM11']#, 'COM13']
    dataPorts = ['COM10']#, 'COM12']
    rawDataSynthesisFINAL.sensorConfiguration(configFile, configPorts)
    fileName = r'D:\Master Thesis\RadarTestData\radarData.txt'
    while True:
        objects = []
        dataFrame, numBytes = readserialData(dataPorts[0])
        if numBytes > 0:
            result, headerStartIndex, totalPacketNumBytes, numDetObj, numTlv, subFrameNumber, detectedX_array, detectedY_array, detectedZ_array, detectedV_array, detectedRange_array, detectedAzimuth_array, detectedElevAngle_array, detectedSNR_array, detectedNoise_array = parser_one_mmw_demo_output_packet(dataFrame, numBytes)
            #configParameters = rawDataSynthesisFINAL.parseConfigFile(configFile)
            #detObj, dataOK = rawDataSynthesisFINAL.readAndParseData(dataFrame, configParameters)
            #if dataOK:
            #    print(detObj)
            print("The detected object values are : ")
            print(detectedX_array, detectedY_array, detectedZ_array, detectedSNR_array, detectedNoise_array)