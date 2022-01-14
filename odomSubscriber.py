import rospy
import time
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty

def storeData(odomData):
    fileName = r'D:\Master Thesis\RadarTestData\radarData.txt'
    with open(fileName, 'a+') as file:
        current_time = now.strftime("%H:%M:%S")
        file.write("x : ",odomData.pose.position.x)
        file.write('\n')
        file.write("y : ",odomData.pose.position.y)
        file.write('\n')
        file.write("theta : ", odomData.pose.orientation)
        file.close()
    dataFrame, portCount = readserialData(dataPorts[0])
    if len(dataFrame) > 0:
        configParameters = rawDataSynthesisFINAL.parseConfigFile(configFile)
        detObj, dataOK = rawDataSynthesisFINAL.readAndParseData(dataFrame, configParameters)
        if dataOK:
            now = datetime.now()
            with open(fileName, 'a+') as file:
                current_time = now.strftime("%H:%M:%S")
                file.write(current_time)
                file.write('\n')
                file.write(dataPorts[0])
                file.write('\n')
                for key, value in detObj.items():
                    file.write('%s : %s\n' % (key, value))
                file.close()
    dataFrame, portCount = readserialData(dataPorts[1])
    if len(dataFrame) > 0:
        configParameters = rawDataSynthesisFINAL.parseConfigFile(configFile)
        detObj, dataOK = rawDataSynthesisFINAL.readAndParseData(dataFrame, configParameters)
        if dataOK:
            now = datetime.now()
            with open(fileName, 'a+') as file:
                current_time = now.strftime("%H:%M:%S")
                file.write('\n')
                file.write(current_time)
                file.write('\n')
                file.write(dataPorts[1])
                file.write('\n')
                for key, value in detObj.items():
                    file.write('%s : %s\n' % (key, value))
                file.close()

if __name__ == '__main__':
    configFile = r'D:\Master Thesis\Config_files_for_testing\Optimal\profile_2022_01_13T12_35_27_476_13_01_2022_1.cfg'
    configPorts = ['COM11', 'COM13']
    dataPorts = ['COM10', 'COM12']
    rawDataSynthesisFINAL.sensorConfiguration(configFile, configPorts)
    rospy.init_node('dataCollector')
    rospy.Subscriber("odom", Odometry, storeData)
    rospy.spin()
