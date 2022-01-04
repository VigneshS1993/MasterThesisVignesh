import numpy as np

def transform2Odom(cellPoints, OdomdataCurrrent, OdomdataPrevious):
    position = np.array([OdomData.pose])
    r1 = np.array([[np.cos(Odomdata.rot*np.pi/180), -np.sin(Odomdata.rot*np.pi/180), 0], [np.sin(Odomdata.rot*np.pi/180), np.cos(Odomdata.rot*np.pi/180), 0], [0, 0, 1]])
    translation = OdomdataCurrent.pose - OdomdataPrevious.pose
    poseGuess = np.dot(r1, cellPoints.T) + translation
    return poseGuess

def transform(cellPoints, port):
    #print(cellPoints1, cellPoints2)
    l1 = np.array([0.5, 0.5])
    l2 = np.array([0.5, 0.5])
    dx1 = l1[0]
    dy1 = l1[1]
    dx2 = l2[0]
    dy2 = l2[1]
    theta1 = 45
    theta2 = -45
    ## For two 3D transformations  for here rotation about z
    r1 = [[np.cos((theta1*np.pi/180)), -np.sin(theta1*np.pi/180), 0] ,[np.sin(theta1*np.pi/180), np.cos(theta1*np.pi/180), 0], [0, 0, 1]]
    r2 = [[np.cos((theta2*np.pi/180)), -np.sin(theta2*np.pi/180), 0] ,[np.sin(theta2*np.pi/180), np.cos(theta2*np.pi/180), 0], [0, 0, 1]]
    #print(r1, r2)
    t1 = [[dx1], [dy1], [0]]
    t2 = [[dx2], [dy2], [0]]
    #print(t1, t2)
    ones1 = np.ones(len(cellPoints[:, 0]))
    ones2 = np.ones(len(cellPoints[:, 0]))
    p1 = np.array([cellPoints[:, 0], cellPoints[:, 1], cellPoints[:, 2], ones1])
    p2 = np.array([cellPoints[:, 0], cellPoints[:, 1], cellPoints[:, 2], ones2])
    #print("The points are ", p1, p2)
    hT1 = np.concatenate((r1, t1), axis=1)
    hT2 = np.concatenate((r2, t2), axis=1)
    #print("Before homogenization", hT1, hT2)
    hT2 = np.concatenate((hT2, [[0, 0, 0, 1]]), axis=0)
    hT1 = np.concatenate((hT1, [[0, 0, 0, 1]]), axis=0)
    hT1 = np.matrix(hT1)
    hT2 = np.matrix(hT2)
    #print("After homogenization", hT1, hT2)
    radar1 = np.matmul(hT1, p1)
    radar1 = radar1.round(decimals=3)
    radar2 = np.matmul(hT2, p2)
    radar2 = radar2.round(decimals=3)
    #print("The final radar points are ", radar1, radar2)
    x1 = radar1[0]
    y1 = radar1[1]
    x2 = radar2[0]
    y2 = radar2[1]
    """#t1 = np.matrix([[1, 0, 0, dx1], [0, 1, 0, dy1], [0, 0, 1, 0], [0, 0, 0, 1]])
    #t2 = np.matrix([[1, 0, 0, dx2], [0, 1, 0, dy2], [0, 0, 1, 0], [0, 0, 0, 1]])
    t1 = np.matrix([[-dx1], [-dy1], [0]])
    t2 = np.matrix([[dx2], [-dy2], [0]])
    r1 = np.matrix([[np.cos(theta1 * np.pi / 180), -np.sin(theta1 * np.pi / 180), 0], [np.sin(theta1 * np.pi / 180), np.cos(theta1 * np.pi / 180), 0], [0, 0, 1]])
    r2 = np.matrix([[np.cos(theta2 * np.pi / 180), -np.sin(theta2 * np.pi / 180), 0], [np.sin(theta2 * np.pi / 180), np.cos(theta2 * np.pi / 180), 0], [0, 0, 1]])
    #r1 = np.matrix([[np.cos(theta1 * np.pi / 180), -np.sin(theta1 * np.pi / 180), 0, 0],
    #                [np.sin(theta1 * np.pi / 180), np.cos(theta1 * np.pi / 180), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    #r2 = np.matrix([[np.cos(theta2 * np.pi / 180), -np.sin(theta2 * np.pi / 180), 0, 0],
    #                [np.sin(theta2 * np.pi / 180), np.cos(theta2 * np.pi / 180), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    #ones1 = np.ones(len(cellPoints1[:, 0]))
    #ones2 = np.ones(len(cellPoints2[:, 0]))
    #print(r1, r2)
    #print(cellPoints1)
    #p1 = np.array([cellPoints1[:,0], cellPoints1[:,1], cellPoints1[:,2], ones1])
    #p2 = np.array([cellPoints2[:,0], cellPoints2[:,1], cellPoints2[:,2], ones2])
    #print(p1, p2)
    p1 = np.matrix([cellPoints1[:, 0], cellPoints1[:, 1], cellPoints1[:, 2]])
    p2 = np.matrix([cellPoints2[:, 0], cellPoints2[:, 1], cellPoints2[:, 2]])
    #radar1 = np.matmul(t1, p1)
    #radar2 = np.matmul(t2, p2)
    #print("Before rotation ", radar1, radar2)
    #radar1 = np.matmul(r1, p1)
    #radar2 = np.matmul(r2, p2)
    radar1 = np.matmul(r1, p1)
    radar2 = np.matmul(r2, p2)
    print("After translation", radar1, radar2)
    radar1 += t1
    radar2 += t2
    print("After rotation", radar1, radar2)
    return radar1, radar2"""
    points1 = np.column_stack((x1, y1))
    points2 = np.column_stack((x2, y2))
    if port == 1:   ## Port 1 is COM10 - right side in my PC
        return x1, y1
    else:           ## Port 2 is COM12 - left side in my PC
        return x2, y2

if __name__ == '__main__':
    cellPoints1 = np.column_stack(([1, 2, 3, 1], [1, 2, 3, 1], [1, 2, 3, 1]))
    cellPoints2 = np.column_stack(([2, 3, 4, 6], [2, 3, 4, 6], [2, 3, 4, 6]))
    theta1 = 0
    theta2 = 0
    l1 = np.array([0.45, 0.0])
    l2 = np.array([0.45, 0.0])
    #transform(cellPoints1, cellPoints2, theta1, theta2, l1, l2)