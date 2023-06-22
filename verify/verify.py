import trimesh
import numpy as np
import matplotlib.pyplot as plt


K = """
1885.5221881803345 0 360.0
0 1885.5221881803345 640.0
0 0 1
"""


def unreal_to_opencv_matrix(location, rotator):
    roll = np.deg2rad(rotator[1])
    yaw = -np.deg2rad(rotator[2])
    pitch = -np.deg2rad(rotator[0])

    # Create rotation matrices for each rotation
    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(pitch), np.sin(pitch), 0],
                   [0, -np.sin(pitch), np.cos(pitch), 0],
                   [0, 0, 0, 1]])

    Ry = np.array([[np.cos(yaw), 0, -np.sin(yaw), 0],
                   [0, 1, 0, 0],
                   [np.sin(yaw), 0, np.cos(yaw), 0],
                   [0, 0, 0, 1]])

    Rz = np.array([[np.cos(roll), np.sin(roll), 0, 0],
                   [-np.sin(roll), np.cos(roll), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Combine rotation matrices (in reverse order)
    rotation_matrix = Rx @ Ry @ Rz

    x = location[0]
    y = -location[2]  # Flip Y coordinate
    z = -location[1]  # Flip Z coordinate

    # Add translation to the matrix
    rotation_matrix[0,3] = x
    rotation_matrix[1,3] = y
    rotation_matrix[2,3] = z

    # Rotate to acount for different coordinate systems
    deg = -90
    rot_negative_90_deg = np.array([
        [np.cos(np.deg2rad(deg)), 0, -np.sin(np.deg2rad(deg)), 0],
        [0, 1, 0, 0],
        [np.sin(np.deg2rad(deg)), 0, np.cos(np.deg2rad(deg)), 0],
        [0, 0, 0, 1]])
    
    rotation_matrix = rotation_matrix @ rot_negative_90_deg

    return rotation_matrix


# Location and rotation in Unreal format:
location = [-100, 0, 160]
rotator = [0, 0, 0]

# Other examples:
# location = [-100, 0, 160]
# rotator = [-10, -10, -10]

# location = [1752.130371, -194.492889, 1298.693848]
# rotator = [0.0, -46.941517, 144.097931]
# location[0] -= 1000
# location[1] -= 350
# location[2] -= 155

c2w = unreal_to_opencv_matrix(location, rotator)

np.set_printoptions(suppress = True)
print(c2w)

# Read extrinsic and intrinsic matrices from string
# c2w = np.fromstring(pose, sep=" ")
c2w = c2w.reshape(4, 4)

# Convert from OpenCV to Blender because the 
# mesh in in Blender coordinate system
blender2opencv = np.diag([1,-1,-1,1])
c2w = blender2opencv @ c2w

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
