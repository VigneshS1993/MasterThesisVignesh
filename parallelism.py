"""This program will help us to parallelize the data coming from the radars and fuse them"""

import threading
import multiprocessing as mp
from serial import Serial as Se
from sys import platform
from serialDataParsing import parser_one_mmw_demo_output_packet


def readData(port):
    with Se(port, baudrate=921600, timeout=3) as ser:
        #ser.open()
        line = ser.readline()
        #q.put(line)
        print(line)
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

    q = mp.Queue()
    p1 = mp.Process(target=readData, args=('COM9',))
    p2 = mp.Process(target=readData, args=('COM6',))
    p1.start()
    p2.start()

    p1.join()
    p2.join()

    #serialData = []

    """while not q.empty() == False:
        data = q.get()
        serialData.append(data)

    print(serialData)"""

