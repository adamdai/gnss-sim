import open3d as o3d
import gnss_lib_py as glp
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt

from sv_utils import azel2los_vec, enu2o3d
from o3d_utils import visualize, generate_lineset_from_mesh, create_ground_plane, rays_tensor, ray_geometries

# %% Setup / Parameters

MODEL_PATH = "data/NYC.obj"
timestamp = datetime(year=2024, month=3, day=14, hour=0, tzinfo=timezone.utc)
RX_LLA = np.reshape([40.74685, -73.9895, 0], [3, 1])

# %% Load the .obj file as a triangle mesh

mesh = o3d.io.read_triangle_mesh(MODEL_PATH)
mesh_edges = generate_lineset_from_mesh(mesh, exclude_diagonals=True)
ground_plane = create_ground_plane(size=500, grid_step=10)

origin = o3d.geometry.PointCloud()
origin.points = o3d.utility.Vector3dVector([[0, 0, 0]])
origin.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))


# %% Get satellite ephemeris and compute az/el info

gps_millis = glp.datetime_to_gps_millis(timestamp)

sp3_path = glp.load_ephemeris(file_type="sp3", gps_millis=gps_millis, verbose=True)
sp3 = glp.Sp3(sp3_path)

rx_ecef = np.reshape(glp.geodetic_to_ecef(RX_LLA), [3, 1])

rx_state = glp.NavData()
rx_state["gps_millis"] = gps_millis
rx_state["x_rx_m"] = rx_ecef[0]
rx_state["y_rx_m"] = rx_ecef[1]
rx_state["z_rx_m"] = rx_ecef[2]

# Plot skyplot computes az/els
# sp3 = sp3.where("gps_millis", sp3['gps_millis'][0], "eq")
fig = glp.plot_skyplot(sp3, rx_state)

# %% Gather az/el info per epoch

timestamps = []
dop_per_epoch = []
num_LOS_svs_per_epoch = []

for timestamp, delta_t, data in glp.loop_time(sp3, "gps_millis"):
    data_df = data.pandas_df()

    # Get az/els
    all_sv_els = data['el_sv_deg']
    all_sv_azs = data['az_sv_deg']

    above_horizon_svs = all_sv_els > 0 
    above_horizon_sv_els = all_sv_els[above_horizon_svs]
    above_horizon_sv_azs = all_sv_azs[above_horizon_svs]
    data_df = data_df[above_horizon_svs]

    sv_azels_rad = np.radians(np.stack([above_horizon_sv_azs, above_horizon_sv_els], axis=1))

    # Determine LOS/NLOS
    sv_directions = enu2o3d(azel2los_vec(sv_azels_rad))
    ray_origins = np.zeros((sv_directions.shape[0], 3))
    sv_rays = rays_tensor(ray_origins, sv_directions)

    result = scene.cast_rays(sv_rays)
    LOS_svs = result['t_hit'].cpu().numpy() == np.inf

    # Get LOS vectors and compute DOP
    LOS_data_df = data_df[LOS_svs]
    LOS_data = glp.NavData()
    LOS_data.from_pandas_df(LOS_data_df)
    dop_df = glp.get_dop(LOS_data)

    timestamps.append(timestamp)
    dop_per_epoch.append([dop_df["HDOP"], dop_df["VDOP"]])
    num_LOS_svs_per_epoch.append(np.sum(LOS_svs))

# %% Plot DOP and num SVs over time

timestamps = np.array(timestamps)
dop_per_epoch = np.array(dop_per_epoch)
num_LOS_svs_per_epoch = np.array(num_LOS_svs_per_epoch)

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(timestamps, dop_per_epoch[:, 0], label="HDOP")
ax[0].plot(timestamps, dop_per_epoch[:, 1], label="VDOP")
ax[0].set_ylabel("DOP")
ax[0].legend()

ax[1].plot(timestamps, num_LOS_svs_per_epoch, label="Num LOS SVs")
ax[1].set_ylabel("Num LOS SVs")
ax[1].legend()

plt.show()

# %% Animate the rays over time

