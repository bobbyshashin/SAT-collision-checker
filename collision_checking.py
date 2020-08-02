# Import system libraries
import argparse
import os
import sys

# Modify the following lines if you have problems importing the V-REP utilities
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd,'lib'))
sys.path.append(os.path.join(cwd,'utilities'))

import numpy as np
from numpy import cos,sin
# import vrep_utils as vu

joint_cuboid_names = ['arm_base_link_joint_collision_cuboid',
                'shoulder_link_collision_cuboid',
                'elbow_link_collision_cuboid',
                'forearm_link_collision_cuboid',
                'wrist_link_collision_cuboid',
                'gripper_link_collision_cuboid',
                'finger_r_collision_cuboid',
                'finger_l_collision_cuboid']

obstacle_cuboid_names = ['cuboid_0',
                'cuboid_1',
                'cuboid_2',
                'cuboid_3',
                'cuboid_4',
                'cuboid_5']

def constructRotationMatrix(orientation, rotation_convention):
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
    if rotation_convention is "ZYX":
        return Rz @ Ry @ Rx
    elif rotation_convention is "XYZ": # for vrep
        return Rx @ Ry @ Rz

def constructVertices(dimension):
    dx, dy, dz = dimension
    vertices_constructor_matrix = np.array([[ 1,  1,  1],
                                            [ 1,  1, -1],
                                            [ 1, -1,  1],
                                            [-1,  1,  1],
                                            [ 1, -1, -1],
                                            [-1,  1, -1],
                                            [-1, -1,  1],
                                            [-1, -1, -1]]).T

    dim = np.array([dx * 0.5, dy * 0.5, dz * 0.5]).reshape(3, 1)

    return (vertices_constructor_matrix * np.tile(dim, (1, 8)))

class Cuboid:
    def __init__(self, origin, orientation, dimension, rotation_convention="ZYX"):

        self.origin = np.array([origin[0], origin[1], origin[2]])

        self.rpy = np.array([orientation[0], orientation[1], orientation[2]])
        self.R = constructRotationMatrix(orientation, rotation_convention)

        self.dimension = np.array([dimension[0], dimension[1], dimension[2]])

        self.vertices = (self.R @ constructVertices(self.dimension) + self.origin.reshape(3,1)).transpose() # 8x3 matrix, each row is the [x, y, z] for one vertex

    def update(self, origin, R):
        self.origin = np.array([origin[0], origin[1], origin[2]])
        self.R = R.copy()
        self.vertices = (self.R @ constructVertices(self.dimension) + self.origin.reshape(3,1)).transpose() # 8x3 matrix, each row is the [x, y, z] for one vertex

def constructCuboid(origin, orientation, dimension):

    cuboid = Cuboid(origin, orientation, dimension)
    return cuboid


def singleAxisCollisionCheck(axis, c1, c2):
    if c1.vertices.shape[0] != c2.vertices.shape[0]:
        print("Error: num of vertices does not match!")
        return None

    num_vertices = c1.vertices.shape[0]

    v1 = c1.vertices
    v2 = c2.vertices

    max1 = float('-inf')
    min1 = float('inf')
    max2 = float('-inf')
    min2 = float('inf')

    for i in range(num_vertices):

        dist1 = v1[i] @ axis
        dist2 = v2[i] @ axis

        if dist1 > max1: max1 = dist1
        if dist1 < min1: min1 = dist1
        if dist2 > max2: max2 = dist2
        if dist2 < min2: min2 = dist2

    longSpan = max(max1, max2) - min(min1, min2)
    sumSpan = max1 - min1 + max2 - min2

    return longSpan <= sumSpan


def checkOverlap(max1, min1, max2, min2):

    if min1 > max2 or max1 < min2:
        return False
    else:
        return True


def findNormalVectors(c1, c2):

    # columns are x, y, z of the cuboid in world frame
    c1_axes = np.eye(3) @ c1.R
    c2_axes = np.eye(3) @ c2.R

    # each column is an axis to check SAT
    normals = np.empty((3, 0))

    for i in range(3):
        for j in range(3):
            new_axis = np.cross(c1_axes[:, i], c2_axes[:, j]).reshape(3, 1)
            if np.linalg.norm(new_axis) != 0.0:
                normals = np.append(normals, new_axis, axis=1)

    normals = np.append(normals, c1_axes, axis=1)
    normals = np.append(normals, c2_axes, axis=1)

    return normals # 3xn matrix, each column is a normal vector (either of one surface of a cuboid, or of a pair of edges of two cuboids)

def SATcollisionCheck(c1, c2):

    axes_to_check = findNormalVectors(c1, c2)

    collide = True
    for i in range(axes_to_check.shape[1]):
        yes = singleAxisCollisionCheck(axes_to_check[:, i], c1, c2)
        collide = collide and yes
    # print(collide)
    return collide


def vecProjection(vec_ref, vec):
    return (np.dot(vec_ref, vec) / (np.linalg.norm(vec_ref) ** 2)) * vec_ref


class CollisionChecker:
    def __init__(self, link_cuboids, obstacle_cuboids, forward_kinematics_handler):
        self.link_cuboids = link_cuboids
        self.obstacle_cuboids = obstacle_cuboids
        self.fk_handler = forward_kinematics_handler

    def collisionCheckAll(self):

        # Check collision between obstacle and links
        num_links = len(self.link_cuboids)
        num_obstacles = len(self.obstacle_cuboids)

        for i in range(num_links):
            for j in range(num_obstacles):
                collide = SATcollisionCheck(self.link_cuboids[i], self.obstacle_cuboids[j])
                if collide:
                    return True
                    # print(joint_cuboid_names[i] + " and " + obstacle_cuboid_names[j] + ": Collided!")
                    # print(collide)

        # Check collision between different links (arm's self-collision check)

        for i in range(num_links):
            for j in range(2, num_links - i - 1):
                collide = SATcollisionCheck(self.link_cuboids[i], self.link_cuboids[i+j])
                if collide:
                    # print(joint_cuboid_names[i] + " and " + joint_cuboid_names[i+j] + ":")
                    # print(collide)
                    if (i == 0) and (i+j == 2):
                        # ignore when base_arm and elbow link collides
                        pass
                    else:
                        return True

        # Check collisin between links and ground
        for i in range(num_links):
            min_z = np.min(self.link_cuboids[i].vertices[:, 2])
            # print(min_z)
            if min_z <= 0.0:
                return True

        return False

    def checkCollisionSample(self, sample):
        cuboid_poses = self.fk_handler.updateCuboidPoses(sample)
        # self.link_cuboids.clear()

        # print("Len of link cuboids:")
        # print(len(self.link_cuboids))

        # print("Len of cuboid poses:")
        # print(len(cuboid_poses))

        for i in range(len(cuboid_poses)):
            self.link_cuboids[i+1].update(cuboid_poses[i][0:3, 3], cuboid_poses[i][0:3, 0:3])

        # collide = self.collisionCheck()
        # print(collide)
        return self.collisionCheckAll()

    def validEdgeCheck(self, profile1, profile2, num_interval=10):
        # perform interpolation between two samples and check collision on each interpolated waypoint

        p1 = np.array(profile1)
        p2 = np.array(profile2)

        p_diff = (p2 - p1) * 1.0 / num_interval

        for i in range(num_interval):
            if self.checkCollisionSample(p1 + p_diff * i): # collide
                return False
        return True


# Examples
c_ref = Cuboid(origin=[0.0, 0.0, 0.0], orientation=[0.0, 0.0, 0.0], dimension=[3.0, 1.0, 2.0])

c1 = Cuboid(origin=[0.0, 1.0, 0.0], orientation=[0.0, 0.0, 0.0], dimension=[0.8, 0.8, 0.8])
c2 = Cuboid(origin=[1.5, -1.5, 0.0], orientation=[1.0, 0.0, 1.5], dimension=[1.0, 3.0, 3.0])
c3 = Cuboid(origin=[0.0, 0.0, -1.0], orientation=[0.0, 0.0, 0.0], dimension=[2.0, 3.0, 1.0])
c4 = Cuboid(origin=[3.0, 0.0, 0.0], orientation=[0.0, 0.0, 0.0], dimension=[3.0, 1.0, 1.0])
c5 = Cuboid(origin=[-1.0, 0.0, -2.0], orientation=[0.5, 0.0, 0.4], dimension=[2.0, 0.7, 2.0])
c6 = Cuboid(origin=[1.8, 0.5, 1.5], orientation=[-0.2, 0.5, 0.0], dimension=[1.0, 3.0, 1.0])
c7 = Cuboid(origin=[0.0, -1.2, 0.4], orientation=[0.0, 0.785, 0.785], dimension=[1.0, 1.0, 1.0])
c8 = Cuboid(origin=[-0.8, 0.0, -0.5], orientation=[0.0, 0.0, 0.2], dimension=[1.0, 0.5, 0.5])

# c_test = Cuboid(origin=[-1.5, 1.5, 0.0], orientation=[0.0, 0.0, 0], dimension=[1.0, 2.9, 3.0])
# collisionCheck(c2, c_test)

print("======== Collision Checking For 8 Cuboids ========")
print(SATcollisionCheck(c_ref, c1))
print(SATcollisionCheck(c_ref, c2))
print(SATcollisionCheck(c_ref, c3))
print(SATcollisionCheck(c_ref, c4))
print(SATcollisionCheck(c_ref, c5))
print(SATcollisionCheck(c_ref, c6))
print(SATcollisionCheck(c_ref, c7))
print(SATcollisionCheck(c_ref, c8))