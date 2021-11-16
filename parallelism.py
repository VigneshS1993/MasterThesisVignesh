"""This program will help us to parallelize the data coming from the radars and fuse them"""

import threading
import multiprocessing
import serial
from sys import platform


def portCommunication(port):
    with serial.Serial(portName, 921600, timeout=3) as ser:
        line =ser.readline()
        print(line)
def configuration(port):
    with serial.Serial(port, 115200, timeout=3) as ser:

if  __name__ == '__main__':
    if platform == 'win32':
        dataPorts = ['COM6', 'COM9']
        configPorts = ['COM7', 'COM6']
    elif platform == 'linux':
        dataPorts = ['/usb/..', '/usb/..']
        configPorts = ['/usb/..', '/usb/..']

