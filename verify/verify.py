import trimesh
import numpy as np
import matplotlib.pyplot as plt


K = """
1885.5221881803345 0 360.0
0 1885.5221881803345 640.0
0 0 1
"""

def unreal_vector_to_opencv(vector_unreal: list):
    """ Swaps the axes of a 3D vector so we're in an OpenCV coordinate system. """
    return [vector_unreal[0], vector_unreal[2], vector_unreal[1]]


def unreal_rotator_to_opencv_matrix(rotator_unreal: list):
    """ Converts a 3D unreal rotator to a rotation matrix in OpenCV coordinate system. """

    # Convert to radians
    roll = np.deg2rad(rotator_unreal[1])
    pitch = np.deg2rad(90 - rotator_unreal[2])
    yaw = np.deg2rad(180 + rotator_unreal[0])

    # Produce rotation matrices for each axis
    r_roll = np.array([
        [1, 0, 0, 0],
        [0, np.cos(roll), -np.sin(roll), 0],
        [0, np.sin(roll), np.cos(roll), 0],
        [0, 0, 0, 1]
    ])
    r_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch), 0],
        [0, 1, 0, 0],
        [-np.sin(pitch), 0, np.cos(pitch), 0],
        [0, 0, 0, 1]
    ])
    r_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, 0],
        [np.sin(yaw), np.cos(yaw), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Compute the rotation matrix for all axes
    return r_pitch @ r_yaw @ r_roll


def get_pose(location_unral, rotator_unreal):
    location = unreal_vector_to_opencv(location_unral)
    r = unreal_rotator_to_opencv_matrix(rotator_unreal)
    r[0,3] = location[0]
    r[1,3] = location[1]
    r[2,3] = location[2]
    return r


# Location and rotation in Unreal format:
location = [-100, 0, 160]
rotator = [0, 0, 0]

# Another example:
# location = [1752.130371, -194.492889, 1298.693848]
# rotator = [0.0, -46.941517, 144.097931]
# location[0] -= 1000
# location[1] -= 350
# location[2] -= 155

c2w = get_pose(location, rotator)
# Print pose matrix
np.set_printoptions(suppress = True)
print(c2w)

# Read extrinsic and intrinsic matrices from string
# c2w = np.fromstring(pose, sep=" ")
c2w = c2w.reshape(4, 4)

K = np.fromstring(K, sep=" ")
K = K.reshape(3, 3)

# Load the mesh
mesh = trimesh.load(f'untitled.obj')
vertices = mesh.vertices * 100
width, height = 720, 1280

w2c = np.linalg.inv(c2w)
vertices = w2c[:3,:3] @ vertices.T + w2c[:3,3:]

vertices = K @ vertices
vertices /= vertices[-1:]

vertices = vertices.astype('int')
mask = (vertices[0]>=0) * (vertices[0]<width)*(vertices[1]>=0)*(vertices[1]<height)
vertices = vertices.T[mask]

img = np.zeros((height,width))
img[vertices[:,1],vertices[:,0]] = 1.0
plt.imshow(img)
plt.show()
