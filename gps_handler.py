import time
import numpy as np
import utm

UTM_path = None
diff_x = 0  # -989.70
diff_y = 0  # -58.984
total_lap_distance = 0.0
pi = 3.14159265358979

# Calculate new way points and initialize distance between way points
# for i in range(len(UTM_path)):
#     delta_x = UTM_path[i][0] - UTM_path[(i+1)%len(UTM_path)][0]
#     delta_y = UTM_path[i][1] - UTM_path[(i+1)%len(UTM_path)][1]
#     distance_to_next_waypoint = np.sqrt(delta_x*delta_x + delta_y*delta_y)
#     if distance_to_next_waypoint == 0.0:
#         print(i, "entry has length == 0.0 to last waypoint")
#     UTM_path[i].append(distance_to_next_waypoint)
#     UTM_path[i].append(total_lap_distance)
#     total_lap_distance += distance_to_next_waypoint


def RadToDeg (rad):
    return (rad / pi * 180.0)


def UTMtoLatLon(utmx, utmy):
    lat, lon = utm.to_latlon(utmx, utmy, 10, 'T')
    return lon, lat


def LatLontoUTM(lon, lat):
    utmx, utmy, zone, zoneletter = utm.from_latlon(lat, lon)
    return [utmx, utmy]


# CTE and distance calculations
def sqr(x):
    return x * x


def dist2(v, w):
    return sqr(v[0] - w[0]) + sqr(v[1] - w[1])


def midPointSegment(p, v, w):
    l2 = dist2(v, w)
    if l2 == 0:
        return v

    t = ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1])*(w[1] - v[1])) / l2
    t = np.max([0, np.min([1, t])])
    midp = [(v[0]+t*(w[0]-v[0])), (v[1]+t*(w[1]-v[1]))]
    return midp


def distToSegmentSquared(p, v, w):
    midp = midPointSegment(p, v, w)
    return dist2(p, midp)


def distToSegment(p, v, w):
    return np.sqrt(distToSegmentSquared(p, v, w))


def closestPathSegmentMidpoint(nk, p):
    kp0 = UTM_path[nk % len(UTM_path)]
    kp1 = UTM_path[(nk+1) % len(UTM_path)]
    return midPointSegment(p, kp0, kp1)


def distToPathSegment(nk, p):
    kp0 = UTM_path[nk % len(UTM_path)]
    kp1 = UTM_path[(nk+1) % len(UTM_path)]
    return distToSegment(p, kp0, kp1)


def sideOfPoint(nk, p):
    kp0 = UTM_path[nk % len(UTM_path)]
    kp1 = UTM_path[(nk+1) % len(UTM_path)]
    d = (p[0]-kp0[0])*(kp1[1]-kp0[1])-(p[1]-kp0[1])*(kp1[0]-kp0[0])
    if d < 0.0:
        return -1
    return 1

# find nearest waypoint and its CTE and distance to next waypoint


def NearestWayPointCTEandDistance(p):
    distance_to_waypoint = 1000.0

    k=0
    for i in range(len(UTM_path)):
        distance = distToPathSegment(i, p)
        if distance < distance_to_waypoint:
            distance_to_waypoint = distance
            k=i

    # get closest midpoint
    midp = closestPathSegmentMidpoint(k, p)

    # calculate CTE and distance to next waypoint
    dist_2_next_waypoint = np.sqrt(dist2(UTM_path[(k+1) % len(UTM_path)], p))
    CTE = sideOfPoint(k, p)*distToPathSegment(k, p)

    # calculate current lap distance
    lap_distance = UTM_path[k][3]
    lap_distance += np.sqrt(dist2(UTM_path[k], midp))

    # return results
    return CTE, dist_2_next_waypoint, lap_distance