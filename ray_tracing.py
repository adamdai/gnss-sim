import open3d as o3d
import gnss_lib_py as glp
from datetime import datetime, timezone
import numpy as np

from sv_utils import azel2los_vec, enu2o3d
from o3d_utils import visualize, generate_lineset_from_mesh, create_ground_plane, rays_tensor, ray_geometries


# %% Load the .obj file as a triangle mesh
mesh = o3d.io.read_triangle_mesh("data/NYC.obj")

mesh_edges = generate_lineset_from_mesh(mesh, exclude_diagonals=True)
ground_plane = create_ground_plane(size=500, grid_step=10)

origin = o3d.geometry.PointCloud()
origin.points = o3d.utility.Vector3dVector([[0, 0, 0]])
origin.colors = o3d.utility.Vector3dVector([[1, 0, 0]])


# %% Get satellite ephemeris and az/el info
timestamp_start = datetime(year=2023, month=3, day=14, hour=12, tzinfo=timezone.utc)
timestamp_end = datetime(year=2023, month=3, day=14, hour=13, tzinfo=timezone.utc)
gps_millis = glp.datetime_to_gps_millis(np.array([timestamp_start,timestamp_end]))

sp3_path = glp.load_ephemeris(file_type="sp3", gps_millis=gps_millis, verbose=True)
sp3 = glp.Sp3(sp3_path)

rx_LLA = np.reshape([40.74685, -73.9895, 0], [3, 1])
rx_ecef = np.reshape(glp.geodetic_to_ecef(rx_LLA), [3, 1])

rx_state = glp.NavData()
rx_state["gps_millis"] = gps_millis[0]
rx_state["x_rx_m"] = rx_ecef[0]
rx_state["y_rx_m"] = rx_ecef[1]
rx_state["z_rx_m"] = rx_ecef[2]

# Plot skyplot computes az/els
# sp3 = sp3.where("gps_millis", sp3['gps_millis'][0], "eq")
fig = glp.plot_skyplot(sp3, rx_state)