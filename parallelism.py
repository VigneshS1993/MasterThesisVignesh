"""This program will help us to parallelize the data coming from the radars and fuse them"""

import threading
import multiprocessing
import serial


def portCommunication(portName):
    with serial.Serial(portName, 921600, timeout=3) as ser:
        line =ser.readline()
        print(line)

if  __name__ == '__main__':
    portname =