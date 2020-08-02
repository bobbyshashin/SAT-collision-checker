import numpy as np
from numpy import cos,sin

def constructRotationMatrix(orientation, rotation_convention):
    # roll, pitch, yaw
    r, p, y = orientation

    Rx = np.array([[1.0, 0.0, 0.0],
                  [0.0, cos(r), -sin(r)],
                  [0.0, sin(r), cos(r)]])

    Ry = np.array([[cos(p), 0.0, sin(p)],
                  [0.0, 1.0, 0.0],
                  [-sin(p), 0.0, cos(p)]])

    Rz = np.array([[cos(y), -sin(y), 0.0],
                  [sin(y), cos(y), 0.0],
                  [0.0, 0.0, 1.0]])

    if rotation_convention == "ZYX": # ZYX euler angle
        return Rz @ Ry @ Rx
    elif rotation_convention == "XYZ": # XYZ euler angle (for vrep)
        return Rx @ Ry @ Rz

class Cuboid:
    def __init__(self, origin, orientation, dimension, rotation_convention="ZYX"):

        self.origin = np.array([origin[0], origin[1], origin[2]])

        self.rpy = np.array([orientation[0], orientation[1], orientation[2]])
        self.R = constructRotationMatrix(orientation, rotation_convention)

        # dimension is the length of each edge
        self.dimension = np.array([dimension[0], dimension[1], dimension[2]])

        # 8x3 matrix, each row is the [x, y, z] for one vertex
        self.vertices = (self.R @ constructVertices(self.dimension) + self.origin.reshape(3,1)).transpose()

    def update(self, origin, R):
        self.origin = np.array([origin[0], origin[1], origin[2]])
        self.R = R.copy()
        self.vertices = (self.R @ constructVertices(self.dimension) + self.origin.reshape(3,1)).transpose() # 8x3 matrix, each row is the [x, y, z] for one vertex

def constructCuboid(origin, orientation, dimension):

    cuboid = Cuboid(origin, orientation, dimension)
    return cuboid