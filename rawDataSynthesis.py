""" This program will help us to decode the raw data received from the serial ports (radars) """
import serial
import time
import numpy as np

byteBuffer = np.zeros(2**15,dtype = 'uint8')
byteBufferLength = 0

def parseConfigFile(configFileName):
    configParameters = {}
    config = [line.rstrip('\n') for line in open(configFileName)]
    for i in config:

        splitWords = i.split(" ")
        # Split lines

        #Hard code number of antennas, change if other configuration is used
        numRxAnt = 4
        numTxAnt = 3

        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[3])
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
            framePeriodicity = int(splitWords[5])

    # Combine the read data to obtain the configuration parameters
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / ((2 * startFreq * 1e9) * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)

    return configParameters

def readAndParseData(serialData, configParameters):
    global byteBuffer, byteBufferLength
    byteBufferLength = 0

    #Constants

    OBJ_STRUCT_SIZE_BYTES = 12
    BYTE_VEC_ACC_MAX_SIZE = 2**15
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1
    MMWDEMO_UART_MSG_RANGE_PROFILE = 2
    maxBufferSize = 2**5
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

    #Initialize variables

    magicOK = 0 # Checks if magic number has been read
    dataOK = 0 # checks if the data has been read correctly
    frameNumber = 0
    detObj = {}

    byteVec = np.frombuffer(serialData, dtype = 'uint8')
    print(f"The byte vector is {byteVec}")
    byteCount = len(byteVec)

    print(f"The value of byteBufferLength, byteCount and maxBufferSize are {byteBufferLength}, {byteCount}, {maxBufferSize}")

    if (byteBufferLength + byteCount) > maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength = byteBufferLength + byteCount
        print(f"The new byteBufferLength is {byteBufferLength}")

        if byteBufferLength > 16:
            # check for possible locations of the magic word
            possibleLocs = np.where(byteBuffer == magicWord[0])[0]
            print(f"The magic word is found in {possibleLocs}")

            # Confirm that is the beginning of the magic word and store the index in startIdx
            startIdx = []
            for loc in possibleLocs:
                check = byteBuffer[loc:loc+8]
                if np.all(check == magicWord):
                    startIdx.append(loc)

            if startIdx:
                # Remove the data before the first start index
                if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                    byteBuffer[:byteBufferLength - startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                    byteBuffer[startIdx[0]:byteBufferLength] = np.zeros(len(byteBufferLength - startIdx[0]), dtype= 'uint8')
                    byteBufferLength = byteBufferLength - startIdx[0]

                # Check that there have been no errors with the byte buffer length
                if byteBufferLength < 0:
                    byteBufferLength = 0

                # Word array to convert 4 bytes to a 32 bit number
                word = [1, 2**8, 2**16, 2**24]

                # Read the total packet length
                totalPacketLen = np.matmul(byteBuffer[12:12+4], word)

                # Check that all the packet has been read

                #if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                if byteBufferLength != 0:
                    magicOK = 1
                else:
                    print(f"The byteBufferLength, the total packet length are : {byteBufferLength}, {totalPacketLen}")
                    print("Magic word not found !! ")

        print(f"The value of magic ok variable is {magicOK}")

        if magicOK:
            #word array to convert 4 bytes to a 32 bit number
            word = [1, 2**8, 2**16, 2**24]

            #Initialize the pointer index
            idx = 0
            # Reading the header
            magicNumber = byteBuffer[idx:idx+8]
            idx += 8
            version = format(np.matmul(byteBuffer[idx:idx+4], word), 'x')
            idx += 8
            totalPacketLen = np.matmul(byteBuffer[idx:idx+4], word)
            idx += 8
            platform = format(np.matmul(byteBuffer[idx:idx+4], word), 'x')
            idx += 8
            frameNumber = np.matmul(byteBuffer[idx:idx+4], word)
            idx += 8
            timeCpuCycles = np.matmul(byteBuffer[idx:idx+4], word)
            idx += 8
            numDetectedObj = np.matmul(byteBuffer[idx:idx+4], word)
            idx += 8
            numTLVs = np.matmul(byteBuffer[idx:idx+4], word)
            idx += 8

            print(f"The number of objects detected in this case is {numDetectedObj} and the number of TLVs is {numTLVs}")

            #Read the tlv messages
            for tlv in range(numTLVs):                 #### A bit ambiguous, going by the online github code !!!
                # Check the header of the tlv message
                tlv_type = np.matmul(byteBuffer[idx:idx+4], word)
                idx += 4
                tlv_length = np.matmul(byteBuffer[idx:idx+4], word)
                idx += 4

                # Read the data depending on the tlv message
                if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:
                    # word array to convert 4 bytes into 16 bit number
                    word = [1, 2**8]
                    tlv_numObj = np.matmul(byteBuffer[idx:idx+2], word)
                    idx += 2
                    tlv_xyzQFormat = np.matmul(byteBuffer[idx:idx+2], word)
                    idx += 2

                    #Initialize the arrays

                    rangeIdx = np.zeros(tlv_numObj, dtype = 'int16')
                    dopplerIdx = np.zeros(tlv_numObj, dtype = 'int16')
                    peakVal = np.zeros(tlv_numObj, dtype = 'int16')
                    x = np.zeros(tlv_numObj, dtype = 'int16')
                    y = np.zeros(tlv_numObj, dtype = 'int16')
                    z = np.zeros(tlv_numObj, dtype = 'int16')
                    for object in range(tlv_numObj):

                        # Reading the data for each object
                        rangeIdx[object] = np.matmul(byteBuffer[idx:idx+2], word)
                        idx += 2
                        dopplerIdx[object] = np.matmul(byteBuffer[idx:idx+2], word)
                        idx += 2
                        peakVal[object] = np.matmul(byteBuffer[idx:idx+2], word)
                        idx += 2
                        x[object] = np.matmul(byteBuffer[idx:idx+2], word)
                        idx += 2
                        y[object] = np.matmul(byteBuffer[idx:idx+2], word)
                        idx += 2
                        z[object] = np.matmul(byteBuffer[idx:idx+2], word)
                        idx +=2
                    rangeVal = rangeIdx * configParameters["rangeIdxToMeters"]
                    dopplerIdx[dopplerIdx > (configParameters["numDopplerBins"]/2 - 1)] = doppler[dopplerIdx > (configParameters["numDopplerBins"]/2 - 1)] - 65535
                    dopplerVal = dopplerIdx * configParameters["dopplerResolutionMps"]
                    x = x / tlv_xyzQFormat
                    y = y / tlv_xyzQFormat
                    z = z / tlv_xyzQFormat

                    ## Store the data in the detObj dictionary

                    detObj = {"numObj": tlv_numObj, "rangeIdx": rangeIdx, "range": rangeVal, "dopplerIdx": dopplerIdx,\
                              "doppler": dopplerVal, "peakVal": peakVal, "x": x, "y": y, "z": z}
                    dataOK = 1
                    print(f"The detected objects are {detObj}")

                    # Remove already processed data

                    if idx > 0 and byteBufferLength > idx:
                        shiftSize = totalPacketLen

                        byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
                        byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),dtype = 'uint8')
                        byteBufferLength = byteBufferLength - shiftSize

                        # Check that there are no errors with the buffer length

                        if byteBufferLength < 0:
                            byteBufferLength = 0

    else:
        print("The sum of the buffer length and the byteBuffer size is not > max buffer size")
        print("So nothing is computed !!!")

    return dataOK, frameNumber, detObj










