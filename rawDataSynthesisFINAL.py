import serial
import time
import numpy as np
import math as mt
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


def readAndParseData(bufferData, configParameters):

    MMDEMO_UART_MSG_DETECTED_POINTS = 1
    MMDEMO_UART_MSG_SIDE_INFO = 7
    word = [1, 2**8, 2**16, 2**24]

    #Initialize variables
    dataOK = 0 # check if the data has been read correctly
    byteBuffer = 0
    frameNumber = 0
    numDetectedObj = 0
    detObj = {"noObjects": numDetectedObj, "x": [], "y": [], "z": [], "range": [], "azimuth": [], "elevation": [], "velocity" : []}#, "snr": [], "noise": []}
    byteBuffer = np.frombuffer(bufferData, dtype = 'uint8')
    magicWord = byteBuffer[0:8]
    if len(byteBuffer) > (2**5) and np.all(magicWord == [2, 1, 4, 3, 6, 5, 8, 7]):
        try:
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
                            for objectNum in range(numDetectedObj):
                                tempx = byteBuffer[idX:idX+4].view(dtype=np.float32)
                                if len(tempx) == 1:
                                    x.append(tempx[0])
                                else:
                                    break
                                idX += 4
                                tempy = byteBuffer[idX:idX+4].view(dtype=np.float32)
                                if len(tempy) == 1:
                                    y.append(tempy[0])
                                else:
                                    break
                                idX += 4
                                tempz = byteBuffer[idX:idX+4].view(dtype=np.float32)
                                if len(tempz) == 1:
                                    z.append(tempz[0])
                                else:
                                    break
                                idX += 4
                                tempv = byteBuffer[idX:idX+4].view(dtype=np.float32)
                                if len(tempv) == 1:
                                    velocity.append(tempv[0])
                                idX += 4
                                tempR = mt.sqrt((tempx[0] * tempx[0]) + (tempy[0] * tempy[0]) + (tempz[0] * tempz[0]))
                                computedRange.append(tempR)
                                if tempy == 0.0:
                                    if tempx >= 0.0:
                                        tempCA = 90.00
                                        computedAzimuth.append(tempCA)
                                    else:
                                        tempCA = -90.00
                                        computedAzimuth.append(tempCA)
                                else:
                                    tempCA = mt.atan(tempx/tempy)*180/np.pi
                                    computedAzimuth.append(tempCA)
                                if tempx == 0 and tempy == 0:
                                    if tempz >= 0.0:
                                        tempCE = 90.00
                                        computedElevation.append(tempCE)
                                    else:
                                        tempCE = -90.00
                                        computedElevation.append(tempCE)
                                else:
                                    tempCE = mt.atan(tempz/mt.sqrt((tempx * tempx) + (tempy * tempy) + (tempz * tempz)))*180/np.pi
                                    computedElevation.append(tempCE)
                            else:
                                idX += tlv_length
            detObj = {"noObjects": numDetectedObj, "x": x, "y": y, "z": z, "range": computedRange,
                      "azimuth": computedAzimuth, "elevation": computedElevation, "velocity" : velocity}#, "snr": snr, "noise": noise}
            dataOK = 1

        except Exception as e:
            print("Error while reading/parsing incoming radar data stream! Error:")
            print(e)
            print("ByteBuffer:")
            print(byteBuffer)
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

