import gnss_lib_py as glp
from datetime import datetime, timezone
import numpy as np

def azel2los(az, el):
    x = np.cos(az) * np.sin(el)
    y = np.sin(az) * np.sin(el)
    z = np.cos(el)
    return np.array([x, y, z])

def azel2los_vec(azels):
    x = np.cos(azels[:,0]) * np.sin(azels[:,1])
    y = np.sin(azels[:,0]) * np.sin(azels[:,1])
    z = np.cos(azels[:,1])
    return np.stack((x,y,z)).T

def enu2o3d(enu):
    # ENU: right, in, up
    # o3d: right, up, out
    o3d_pts = enu[:,[0,2,1]]
    o3d_pts[:,2] *= -1
    return o3d_pts

def get_sv_azels(ephem, time, rx_LLA):
    # TODO
    pass