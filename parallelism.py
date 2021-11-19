"""This program will help us to parallelize the data coming from the radars and fuse them"""

import threading
import multiprocessing as mp
from serial import Serial as Se
from sys import platform
from serialDataParsing import parser_one_mmw_demo_output_packet

def keyPress(key):
    if key == keyboard.Key.esc:
        return True
    else:
        return False

def readData(port, q):
    with Se(port, baudrate=921600, timeout=3) as ser:
        #ser.open()
        line = ser.readline()
        length = len(line)
        q.put(line)
        print(f"Length of the line is {length} and the data from the port {port} line ", line)
        #result, headerStartIndex, totalPacketNumBytes, numDetObj, numTlv, subFrameNumber, detectedX_array, detectedY_array, detectedZ_array, detectedV_array, detectedRange_array, detectedAzimuth_array, detectedElevAngle_array, detectedSNR_array, detectedNoise_array = parser_one_mmw_demo_output_packet(line, len(line))
        #print("The number of objects detected in this port are", numDetObj)
        ser.close()
#def configuration(port):
#    with serial.Serial(port, 115200, timeout=3) as ser:

if  __name__ == '__main__':
    if platform == 'win32':
        dataPorts = ['COM6', 'COM9']
        configPorts = ['COM7', 'COM8']
    elif platform == 'linux':
        dataPorts = ['/usb/..', '/usb/..']
        configPorts = ['/usb/..', '/usb/..']

    try:
        while True:
            q1 = mp.Queue()
            q2 = mp.Queue()
            p1 = mp.Process(target=readData, args=('COM9',q1))
            p2 = mp.Process(target=readData, args=('COM6',q2))
            p1.start()
            p2.start()
            p1.join()
            p2.join()
            serialData = []
            ## Not able to acheive the q.empty logic for getting all the values from the queue
            serialData.append(q1.get())
            serialData.append(q2.get())
            print("The appended serial data is ", serialData)

    except KeyboardInterrupt:
        exit()

