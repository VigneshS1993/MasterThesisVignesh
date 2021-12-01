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
    detObj = {"number_of_objects": numDetectedObj, "x": [], "y": [], "z": [], "range": [], "azimuth": [], "elevation": []}

    byteBuffer = np.frombuffer(bufferData, dtype = 'uint8')
    magicWord = byteBuffer[0:8]
    if len(byteBuffer) > (2**5) and np.all(magicWord == [2, 1, 4, 3, 6, 5, 8, 7]):
        try:
            #Initialize the pointer index
            idX = 0
            #Read the header
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
            #print(magicNumber, version, totalPacketLen, platform, frameNumber, timeCpuCycles, numDetectedObj, numTLVs, subFrameNumber)
            if numDetectedObj > 0:
                    # Read the tlv messages
                for tlv in range(numTLVs):
                    tly_type = np.matmul(byteBuffer[idX:idX+4]. word)
                    idX += 4
                    tlv_length = np.matmul(byteBuffer[idX:idX+4], word)
                    idX += 4
                    # Read data depending on the tlv messages
                    if tlv_type == MMDEMO_UART_MSG_DETECTED_POINTS:
                        x = np.zeros(numDetectedObj, dtype=np.float32)
                        y = np.zeros(numDetectedObj, dtype=np.float32)
                        z = np.zeros(numDetectedObj, dtype=np.float32)
                        computedAzimuth = np.zeros(numDetectedObj, dtype=np.float32)
                        computedRange = np.zeros(numDetectedObj, dtype=np.float32)
                        computedElevation = np.zeros(numDetectedObj, dtype=np.float32)
                        velocity = np.zeros(numDetectedObj, dtype=np.float32)
                        for objectNum in range(numDetectedObj):
                            x[objectNum] = byteBuffer[idx:idX+4].view(dtype=float32)
                            idX += 4
                            y[objectNum] = byteBuffer[idX:idX+4].view(dtype=float32)
                            idX += 4
                            z[objectNum] = byteBuffer[idX:idX+4].view(dtype=float32)
                            idx += 4
                            velocity[objectNum] = byteBuffer[idX+4].view(dtype=float32)
                            computedRange[objectNum] = mt.sqrt((x*x) + (y*y) + (z*z))
                            ## azimuth computation from x and y
                            if y == 0.0:
                                if x >= 0.0:
                                    computedAzimuth[objectNum] = 90
                                else:
                                    computedAzimuth[objectNum] = -90
                            else:
                                computedAzimuth[objectNum] = math.atan(x/y)*180/np.pi

                            ## calculating elevation angel from x, y, z
                            if x == 0 and y == 0:
                                if z >= 0.0:
                                    computedElevation[objectNum] = 90
                                else:
                                    computedElevation[objectNum] = -90
                            else:
                                computedElevation[objectNum] = mt.atan(z/mt.sqrt((x*x) + (y*y) + (z*z)))*180/np.pi


                    elif tlv_type == MMDEMO_UART_MSG_SIDE_INFO:
                        snr = np.zeros(numDetectedObj, dtype=np.int16)
                        noise = np.zeros(numDetectedObj, dtype=np.int16)

                        for objectNum in range(numDetectedObj):
                            snr[objectNum] = byteBuffer[idX:idX+2].view(dtype=np.int16)
                            idX += 2
                            noise[objectNum] = byteBuffer[idX:idX+2].view(dtype=np.int16)
                            idX += 2

                    else:
                        idX += tlv_mength
                detObj = {"noObjects": numDetectedObj, "x": x, "y": y, "z": z, "range": computedRange,
                              "azimuth": computedAzimuth, "elevation": computedElevation}
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

