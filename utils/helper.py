from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np


diffx = -989.70
diffy = -58.984
start_location = (39.53745, -122.33879)


mt = Basemap(llcrnrlon=-122.341041, llcrnrlat=39.532678,
             urcrnrlon=-122.337929, urcrnrlat=39.541455,
             projection='merc', lon_0=start_location[1], lat_0=start_location [0], resolution='h')

def to_gps(simx, simy):
    projx, projy = simx+diffx, simy + diffy
    lon, lat = mt(projx, projy, inverse=True)
    return np.array([lon, lat])



def get_norm_factor(angle, hist, edges):
    for i, edge in enumerate(edges[:-1]):
        if (angle > edge) and (angle < edges[i + 1]):
            return hist[i]
    return hist[-1]


def retrieve_gps(vecString):
    split = vecString.split(":")
    return pd.Series(to_gps(float(split[0]), float(split[1])))

