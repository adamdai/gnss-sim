import open3d as o3d
import numpy as np
from collections import defaultdict


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


# def generate_lineset_from_mesh(mesh):
#     # Get the vertices and triangles of the mesh
#     vertices = np.asarray(mesh.vertices)
#     triangles = np.asarray(mesh.triangles)

#     # Initialize an empty list for edges
#     edges = set()

#     # Iterate through each triangle and extract its edges
#     for tri in triangles:
#         # Each triangle has 3 edges (pairs of vertices)
#         edges.add(tuple(sorted([tri[0], tri[1]])))
#         edges.add(tuple(sorted([tri[1], tri[2]])))
#         edges.add(tuple(sorted([tri[2], tri[0]])))

#     # Convert the set of edges to a numpy array
#     edges = np.array(list(edges))

#     # Create the LineSet
#     lineset = o3d.geometry.LineSet()
#     lineset.points = o3d.utility.Vector3dVector(vertices)
#     lineset.lines = o3d.utility.Vector2iVector(edges)

#     return lineset

def generate_lineset_from_mesh(mesh, exclude_diagonals=False):
    # Get the vertices and triangles of the mesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Initialize an empty list for edges and adjacency list
    edges = set()
    edge_to_triangle = defaultdict(list)

    # Iterate through each triangle and record its edges
    for i, tri in enumerate(triangles):
        # Each triangle has 3 edges (pairs of vertices)
        for edge in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            sorted_edge = tuple(sorted(edge))
            edge_to_triangle[sorted_edge].append(i)

    # Now find edges that appear in only one triangle (i.e., outer edges)
    for edge, adjacent_triangles in edge_to_triangle.items():
        if len(adjacent_triangles) == 1:
            # This is a boundary edge, add it
            edges.add(edge)
        elif len(adjacent_triangles) == 2:
            # This is an internal edge, check if we should exclude diagonals
            if exclude_diagonals:
                # Check if it's a diagonal of a rectangle
                tri1, tri2 = adjacent_triangles
                shared_vertices = set(triangles[tri1]) & set(triangles[tri2])
                if len(shared_vertices) == 2:
                    # These triangles share exactly 2 vertices, so they form a rectangle
                    # Exclude the diagonal (i.e., don't add this edge)
                    continue

            # If not a diagonal or diagonals should not be excluded, add the edge
            edges.add(edge)

    # Convert the set of edges to a numpy array
    edges = np.array(list(edges))

    # Create the LineSet
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(vertices)
    lineset.lines = o3d.utility.Vector2iVector(edges)

    return lineset


def create_ground_plane(size=10, grid_step=1, color=[0.5, 0.5, 0.5]):
    # Create a flat grid for the ground plane
    x_range = np.arange(-size, size + grid_step, grid_step)
    z_range = np.arange(-size, size + grid_step, grid_step)

    vertices = []
    lines = []
    colors = []

    # Generate the grid lines in the XZ plane (ground plane)
    for x in x_range:
        vertices.append([x, 0, -size])
        vertices.append([x, 0, size])
        lines.append([len(vertices) - 2, len(vertices) - 1])
        colors.append(color)  # Apply color to each line

    for z in z_range:
        vertices.append([-size, 0, z])
        vertices.append([size, 0, z])
        lines.append([len(vertices) - 2, len(vertices) - 1])
        colors.append(color)  # Apply color to each line

    # Create a LineSet for the ground grid
    ground_plane = o3d.geometry.LineSet()
    ground_plane.points = o3d.utility.Vector3dVector(vertices)
    ground_plane.lines = o3d.utility.Vector2iVector(lines)
    ground_plane.colors = o3d.utility.Vector3dVector(colors)  # Set the color of each line

    return ground_plane