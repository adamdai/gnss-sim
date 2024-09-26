import open3d as o3d
import numpy as np


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction


def ray_geometries(ray_origins, ray_directions):
    # Create a list to hold geometries for visualization
    geometries = []

    # Create a point for the origin
    origin = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)  # Small sphere as origin marker
    origin.paint_uniform_color([1, 0, 0])  # Red color
    origin.translate([0, 0, 0])  # Position at the origin
    geometries.append(origin)

    for i in range(len(ray_origins)):
        p0 = ray_origins[i]
        p1 = p0 + ray_directions[i]  # Endpoint of the ray

        # Create a LineSet for the ray
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([p0, p1])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.paint_uniform_color([0, 1, 0])  # Green color for rays
        geometries.append(line)

    return geometries



def visualize(geometries, bg_color=[0.1, 0.1, 0.1]):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = np.asarray(bg_color)

    for geometry in geometries:
        vis.add_geometry(geometry)

    vis.run()
    vis.destroy_window()


def rays_tensor(origins, directions):
    """Constructs a ray data structure from origins and directions."""
    # Ensure the origins and directions are numpy arrays
    origins = np.asarray(origins)
    directions = np.asarray(directions)
    rays_data = np.hstack((origins, directions))
    return o3d.core.Tensor(rays_data, dtype=o3d.core.Dtype.Float32)


def reflect(ray_direction, normal):
    """Calculate the reflected direction given the ray direction and the normal."""
    # Normalize the incoming ray direction
    direction_normalized = ray_direction / np.linalg.norm(ray_direction)
    # Normalize the normal vector
    normal_normalized = normal / np.linalg.norm(normal)
    
    # Calculate the reflection
    reflected_direction = direction_normalized - 2 * np.dot(direction_normalized, normal_normalized) * normal_normalized
    
    return reflected_direction


def setup_scene(box_dims, box_positions):
    meshes = []
    scene = o3d.t.geometry.RaycastingScene()   

    for i in range(box_dims.shape[0]):
        box_mesh = o3d.geometry.TriangleMesh.create_box(box_dims[i, 0], box_dims[i, 1], box_dims[i, 2])
        box_mesh.translate(box_positions[i])
        meshes.append(box_mesh)
        box_tris = o3d.t.geometry.TriangleMesh.from_legacy(box_mesh)
        scene.add_triangles(box_tris)
    
    return scene, meshes