import numpy as np

def transform(x1, y1, z1, x2, y2, z2):
    dx = 0.5
    dy = 0.5
    r1 = np.matrix([[np.cos((-45*np.pi/180)), -np.sin(-45*np.pi/180), 0, 0]
                       , [np.sin(-45*np.pi/180), np.cos(-45*np.pi/180), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    r2 = np.matrix([[np.cos((45*np.pi/180)), -np.sin(45*np.pi/180), 0, 0]
                       , [np.sin(45*np.pi/180), np.cos(45*np.pi/180), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    t1 = np.matrix([[1, 0, 0, 0]
                       , [0, 1, 0, 0], [0, 0, 1, 0], [dx, dy, 0, 1]])
    t2 = np.matrix([[1, 0, 0, 0]
                       , [0, 1, 0, 0], [0, 0, 1, 0], [-dx, dy, 0, 1]])

    p1 = np.matrix([x1[:], y1[:], z1[:], [1, 1, 1]])
    p2 = np.matrix([x2[:], y2[:], z2[:], [1, 1, 1]])
    print(p2)
    #p1 = np.matrix(p1)
    #p2 = np.matrix(p2)
    #print(x1, x2)

    radar1 = np.matmul(np.matmul(t1, r1), p1)
    radar2 = np.matmul(np.matmul(t2, r2), p2)

    return radar1, radar2

if __name__ == '__main__':
    print(transform([1, 2, 3], [1, 2, 3], [1, 2, 3], [2, 3, 4], [2, 3, 4], [2, 3, 4]))