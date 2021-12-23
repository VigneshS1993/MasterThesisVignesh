import serial
import time
import numpy as np
import math as mt
import struct
import math
import binascii
import codecs
def sensorConfiguration(fileName, configPorts):
    for port in configPorts:
        try:
            confPort = serial.Serial(port, 115200)
        except serial.SerialException as se:
            print("Error occurred and it is : ")
            print(str(se))
            return
        try:
            """with open(fileName) as configFile:
                commands = [command.rstrip('\r\n') for command in configFile]
                for command in commands:
                    confPort.write((command + '\n').encode())
                    time.sleep(0.05)
                    #print(f"{command} successfully written into the sensor..")"""

            comands = [line.rstrip('\r\n') for line in open(fileName)]
            for command in comands:
                confPort.write((command + '\n').encode())
                time.sleep(0.01)
                #confPort.write(('sensorStop\n').encode())      ## Need to send this to ensure the sensor starts sending the data.."""
            confPort.close()
            print("Successfully sent all the commands as CLI..")
        except FileNotFoundError as fe:
            print("No file exists..")

def parseConfigFile(configFileName):
    configParameters = {}
    config = [line.rstrip('\n') for line in open(configFileName)]
    for i in config:
        splitWords = i.split(" ")
        # Hard code number of antennas
        numRxAnt = 4
        numTxAnt = 3

        # Get the information about the profile configurations
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1
            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2
            digOutSampleRate = int(splitWords[11])

            # Get the information about the frame configuration

        elif "frameCfg" in splitWords[0]:
            chirpStartIdx = int(splitWords[1])
            chirpEndIdx = int(splitWords[2])
            numLoops = int(splitWords[3])
            numFrames = int(splitWords[4])
            framePeriodicity = float(splitWords[5])

    #Combine the read data to obtain config parameters

    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (
                2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (
                2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (
                2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)

    return configParameters

def powerOf2(number):
    """if number & (number<<1) == 0 and number!= 0:
        return True
    else:
        return False"""
    while (number > 1):
        if number % 2 == 0:
            number /= 2
        else:
            break
    if number == 1:
        return True
    else:
        return False

def readAndParseData(bufferData, configParameters):
    MMDEMO_UART_MSG_DETECTED_POINTS = 1
    MMDEMO_UART_MSG_SIDE_INFO = 7
    word = [1, 2**8, 2**16, 2**24]

    #Initialize variables
    dataOK = 0 # check if the data has been read correctly
    byteBuffer = 0
    frameNumber = 0
    numDetectedObj = 0
    detObj = {"noObjects": numDetectedObj, "x": [], "y": [], "z": [], "range": [], "azimuth": [], "elevation": [], "velocity" : [], "snr": [], "noise": []} #
    byteBuffer = np.frombuffer(bufferData, dtype = 'uint8')
    magicWord = byteBuffer[0:8]
    if powerOf2(len(byteBuffer)) and np.all(magicWord == [2, 1, 4, 3, 6, 5, 8, 7]):    ## Experimental and a multiple of 2 for the length in bytes
        try:
            #print("The total length of the buffer is ", len(byteBuffer))
            idX = 0
            magicNumber = byteBuffer[0:8]
            idX += 8
            version = format(np.matmul(byteBuffer[idX:idX+4], word), 'x')
            idX += 4
            totalPacketLen = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            platform = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
            idX += 4
            frameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            timeCpuCycles = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            numDetectedObj = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            numTLVs = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            subFrameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            if numDetectedObj > 0:
                for tlv in range(numTLVs):
                    if len(byteBuffer[idX:idX+4]) > 0:
                        tlv_type = np.matmul(byteBuffer[idX:idX+4], word)
                        idX += 4
                        tlv_length = np.matmul(byteBuffer[idX:idX+4], word)
                        idX += 4
                        if tlv_type == MMDEMO_UART_MSG_DETECTED_POINTS:
                            x = []
                            y = []
                            z = []
                            computedAzimuth = []
                            computedRange = []
                            computedElevation = []
                            velocity = []
                            snr = []
                            noise = []
                            for objectNum in range(numDetectedObj):
                                if len(byteBuffer[idX:idX+4]) < 4:
                                    break
                                #print("Before unpacking x")
                                #print(byteBuffer[idX:idX + 4])
                                xi = struct.unpack('<f', codecs.decode(binascii.hexlify(byteBuffer[idX:idX+4]), 'hex'))[0]
                                #xi = round(xi, 4)
                                #print("Unpacked x and the value is ", xi)
                                idX += 4
                                x.append(xi)
                                #print(byteBuffer[idX:idX + 4])
                                if len(byteBuffer[idX:idX+4]) < 4:
                                    x.pop()
                                    break
                                # convert byte4 to byte7 to float y value
                                yi = struct.unpack('<f', codecs.decode(binascii.hexlify(byteBuffer[idX:idX+4]), 'hex'))[0]
                                #yi = round(yi, 4)
                                #print("Unpacked y")
                                idX += 4
                                y.append(yi)
                                #print(byteBuffer[idX:idX + 4])
                                if len(byteBuffer[idX:idX+4]) < 4:
                                    x.pop()
                                    y.pop()
                                    break
                                # convert byte8 to byte11 to float z value
                                zi = struct.unpack('<f', codecs.decode(binascii.hexlify(byteBuffer[idX:idX+4]), 'hex'))[0]
                                #zi = round(zi, 4)
                                #print("Unpacked z")
                                idX += 4
                                z.append(zi)
                                #print(byteBuffer[idX:idX + 4])
                                if len(byteBuffer[idX:idX+4]) < 4:
                                    x.pop()
                                    y.pop()
                                    z.pop()
                                    break
                                # convert byte12 to byte15 to float v value
                                vi = struct.unpack('<f', codecs.decode(binascii.hexlify(byteBuffer[idX:idX+4]), 'hex'))[0]
                                #vi = round(vi, 4)
                                #print("Unpacked velocity")
                                idX += 4
                                velocity.append(vi)
                                tempR = mt.sqrt((xi * xi) + (yi * yi) + (zi * zi))
                                #tempR = round(tempR, 4)
                                #tempR = mt.sqrt((tempx[0] * tempx[0]) + (tempy[0] * tempy[0]) + (tempz[0] * tempz[0]))
                                computedRange.append(tempR)
                                if yi == 0.0:
                                    if xi >= 0.0:
                                        tempCA = 90.00
                                        #tempCA = round(tempCA, 4)
                                        computedAzimuth.append(tempCA)
                                    else:
                                        tempCA = -90.00
                                        computedAzimuth.append(tempCA)
                                else:
                                    #tempCA = mt.atan(tempx[0]/tempy[0])*180/np.pi
                                    tempCA = mt.atan(xi / yi) * 180 / np.pi
                                    #tempCA = round(tempCA, 4)
                                    computedAzimuth.append(tempCA)
                                if xi == 0 and yi == 0:
                                    if zi >= 0.0:
                                        tempCE = 90.00
                                        computedElevation.append(tempCE)
                                    else:
                                        tempCE = -90.00
                                        computedElevation.append(tempCE)
                                else:
                                    #tempCE = mt.atan(tempz[0]/mt.sqrt((tempx[0] * tempx[0]) + (tempy[0] * tempy[0]) + (tempz[0] * tempz[0])))*180/np.pi
                                    tempCE = mt.atan(zi / mt.sqrt((xi * xi) + (yi * yi) + (zi * zi))) * 180 / np.pi
                                    #tempCE = round(tempCE, 4)
                                    computedElevation.append(tempCE)
                            else:
                                idX += tlv_length
                        if tlv_type == 7:
                            for obj in range(numDetectedObj):
                                tempSNR = struct.unpack('<f', codecs.decode(binascii.hexlify(byteBuffer[idX:idX+2]), 'hex'))[0]
                                idx += 2
                                snr.append(tempSNR)
                                tempnoise = struct.unpack('<f', codecs.decode(binascii.hexlify(byteBuffer[idX:idX+2]), 'hex'))[0]
                                idx += 2
                                noise.append(tempnoise)

                        else:
                            for obj in range(numDetectedObj):
                                snr.append(0)
                                noise.append(0)

            detObj = {"noObjects": numDetectedObj, "x": x, "y": y, "z": z, "range": computedRange,
                      "azimuth": computedAzimuth, "elevation": computedElevation, "velocity": velocity, "snr": snr, "noise": noise} #
            dataOK = 1

        except Exception as e:
            print("Error while reading/parsing incoming radar data stream! Error:")
            print(e)
            print("ByteBuffer:")
            print(byteBuffer)
            print("The length of the buffer is : ", len(byteBuffer))
            magicNumber = None
            version = None
            totalPacketLen = None
            platform = None
            frameNumber = None
            timeCpuCycles = None
            numDetectedObj = None
            numTLVs = None
            subFrameNumber = None

    return detObj, dataOK

