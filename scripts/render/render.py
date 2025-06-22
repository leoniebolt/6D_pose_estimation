import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import os

# Ensure EGL is used for rendering
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Load the 3D model
moro_trimesh = trimesh.load('moro.obj')

# Normalize the scale of the object
moro_trimesh.apply_scale(1.0 / np.max(moro_trimesh.extents))

# Center the object at the origin
moro_trimesh.apply_translation(-moro_trimesh.centroid)

# Rotation: 90-degree rotation around the Y-axis
rotation_matrix = trimesh.transformations.rotation_matrix(
    np.radians(90),  # Convert degrees to radians
    [0, 1, 0]  # Rotate around the Y-axis
)
moro_trimesh.apply_transform(rotation_matrix)


# Convert Trimesh to Pyrender Mesh
mesh = pyrender.Mesh.from_trimesh(moro_trimesh)

# Create a scene and add the mesh
scene = pyrender.Scene()
scene.add(mesh)

# Set up the camera
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

# Adjust the camera position to ensure visibility
camera_pose = np.array([
   [1.0,  0.0,  0.0,  0.0],
   [0.0,  1.0,  0.0,  0.0],
   [0.0,  0.0,  1.0,  1.5],  # Move camera further away
   [0.0,  0.0,  0.0,  1.0],
])

# Add camera to the scene
scene.add(camera, pose=camera_pose)

# Add a light source
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                           innerConeAngle=np.pi/16.0,
                           outerConeAngle=np.pi/6.0)
scene.add(light, pose=camera_pose)

# Render the scene
r = pyrender.OffscreenRenderer(400, 400)
color, depth = r.render(scene)

# Display the result
plt.figure()
plt.subplot(1,2,1)
plt.axis('off')
plt.imshow(color)
plt.subplot(1,2,2)
plt.axis('off')
plt.imshow(depth, cmap=plt.cm.gray_r)
plt.savefig("rendered_object.png", dpi=300, bbox_inches='tight')
plt.show()

