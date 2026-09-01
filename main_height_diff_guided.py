"""
Multi-Drone Area Coverage Planner
=================================
Pipeline:
    1. read each drone's launch GPS from its MAVLink telemetry stream
    2. partition the geofence with a Power Diagram, weighted so every drone
       finishes at the same time (real lawnmower time estimate, not area)
    3. generate a boustrophedon (lawnmower) path per cell
    4. stack takeoff/transit altitudes and stagger the launches
    5. export per-drone missions (CSV + optional QGC WPL 110) and ship them
       to the drones over TCP
    6. save a mission plot as PNG (no GUI window)

Usage:  python main_height_diff_guided.py
"""

import csv
import math
import os
import socket
import time

import numpy as np
from pymavlink import mavutil
import matplotlib
matplotlib.use('Agg')  # no GUI — saves to file only
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely import affinity

# ═══════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════
GEOFENCE_FILE  = "oldtest_uas.txt"
OUTPUT_DIR     = '/home/akshit/uav/swarming_guided'

N_DRONES       = 1
TARGET_IP      = ['192.168.1.126']   # one entry per drone
TARGET_PORT    = [9000]              # one entry per drone

ALTITUDE       = 6       # survey altitude (m AGL)
HFOV           = 87.0    # camera horizontal field of view (deg)
BUFFER_WIDTH   = 2.0     # keep-out inset from the cell boundary (m)
CRUISE_SPEED   = 2.0     # survey / lawnmower ground speed (m/s)
TRANSIT_SPEED  = 8.0     # launch -> survey-area cruise speed (m/s)
TRANSIT_ALT    = 15.0    # transit altitude (m AGL), flown before descending to ALTITUDE
ALT_SEPARATION = 5.0     # vertical spacing between drones (m), reported in the plot
LAUNCH_DELAY   = 10.0    # staggered launch: drone i lifts off at i*LAUNCH_DELAY (s)
YAW_RATE       = 45.0    # deg/s, used by the mission time estimate
SETTLING_TIME  = 1.5     # s per waypoint, used by the mission time estimate

BALANCE_ON_LANDING_TIME = True   # True  -> size cells so every drone FINISHES together
                                 #          (later drones get less area)
                                 # False -> balance flight time only; the fleet then
                                 #          finishes staggered by LAUNCH_DELAY

EXPORT_WAYPOINTS = False   # also write QGC .waypoints files next to the CSVs
DEBUG            = True    # print [debug] lines tracing the planner internals

# Drone telemetry: {drone_id: UDP port carrying its MAVLink stream}
DRONE_UDP_PORTS = {
    1: 14553,   # Drone 1 telemetry port
    # 2: 14554,   # Drone 2 telemetry port
}
MAVLINK_TIMEOUT = 10.0   # seconds to wait per drone for a valid GPS fix

R_EARTH = 6_371_000.0

# Populated at runtime from MAVLink telemetry
LAUNCH_LATLON    = (0.0, 0.0)   # coordinate-system origin (drone 1, or fence centroid)
LAUNCH_POSITIONS = {}           # {drone_index: (lat, lon)}

# ═══════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════
_T0 = time.perf_counter()

def log(msg=''):
    print(f'  {msg}')

def warn(msg):
    print(f'  WARNING: {msg}')

def err(msg):
    print(f'  ERROR: {msg}')

def dbg(msg):
    """Verbose trace of the planner internals — silenced by DEBUG = False."""
    if DEBUG:
        print(f'  [debug {time.perf_counter() - _T0:6.2f}s] {msg}')

# ═══════════════════════════════════════════════════
#  MAVLink TELEMETRY GPS READER
# ═══════════════════════════════════════════════════
def read_gps_from_mavlink(drone_id, port, timeout=MAVLINK_TIMEOUT):
    """
    Connect to udp:0.0.0.0:<port> and read the first valid GLOBAL_POSITION_INT
    (MAVLink msg 33), whose lat/lon come in 1e-7 degrees.

    Returns (lat, lon) as floats, or None on timeout / error.
    """
    conn_str = f'udpin:0.0.0.0:{port}'
    log(f'[MAVLink] D{drone_id} — connecting on {conn_str} (timeout={timeout}s) ...')
    try:
        mav = mavutil.mavlink_connection(conn_str, input=True)
        mav.wait_heartbeat(timeout=timeout)
        log(f'[MAVLink] D{drone_id} — heartbeat received (sysid={mav.target_system})')

        msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=timeout)
        if msg is None:
            log(f'[MAVLink] D{drone_id} — timeout waiting for GLOBAL_POSITION_INT '
                f'on port {port}. Falling back to geofence centroid for this drone.')
            return None

        lat, lon = msg.lat / 1e7, msg.lon / 1e7
        log(f'[MAVLink] D{drone_id} GPS -> lat={lat:.7f}  lon={lon:.7f}')
        dbg(f'D{drone_id} raw fix: lat={msg.lat} lon={msg.lon} '
            f'relative_alt={getattr(msg, "relative_alt", 0)/1000.0:.1f}m')
        return (lat, lon)

    except Exception as e:
        log(f'[MAVLink] D{drone_id} — error on port {port}: {e}. '
            f'Falling back to geofence centroid for this drone.')
        return None


def read_all_drone_positions(drone_ports=DRONE_UDP_PORTS, timeout=MAVLINK_TIMEOUT):
    """Read each drone's launch GPS in turn. Returns {drone_id: (lat, lon)} for
    the drones that responded; an empty dict triggers the centroid fallback."""
    positions = {}
    for drone_id, port in sorted(drone_ports.items()):
        fix = read_gps_from_mavlink(drone_id, port, timeout)
        if fix is not None:
            positions[drone_id] = fix
    dbg(f'launch positions resolved for {len(positions)}/{len(drone_ports)} drones')
    return positions


def parse_waypoints(file_path):
    """Read a QGC WPL file and return its waypoints as [(lat, lon), ...]."""
    geofence = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines[1:]:            # skip the 'QGC WPL 110' header
        parts = line.strip().split('\t')
        if len(parts) < 11:
            continue                  # skip malformed lines
        geofence.append((float(parts[8]), float(parts[9])))

    dbg(f'parsed {len(geofence)} geofence vertices from {file_path}')
    return geofence


GEOFENCE = parse_waypoints(GEOFENCE_FILE)

# ═══════════════════════════════════════════════════
#  COORDINATE MATH
# ═══════════════════════════════════════════════════
def ll2xy(lat, lon, rlat, rlon):
    """Lat/lon -> local East/North metres about (rlat, rlon)."""
    c = math.cos(math.radians(rlat))
    return (math.radians(lon - rlon) * R_EARTH * c,
            math.radians(lat - rlat) * R_EARTH)

def xy2ll(x, y, rlat, rlon):
    """Local East/North metres about (rlat, rlon) -> lat/lon."""
    c = math.cos(math.radians(rlat))
    return (rlat + math.degrees(y / R_EARTH),
            rlon + math.degrees(x / (R_EARTH * c)))

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R_EARTH * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def longest_edge_angle(pts):
    """Heading (deg) of the polygon's longest edge — the sweep direction."""
    best, ang = 0, 0
    for i in range(len(pts)):
        dx = pts[(i+1) % len(pts)][0] - pts[i][0]
        dy = pts[(i+1) % len(pts)][1] - pts[i][1]
        d2 = dx*dx + dy*dy
        if d2 > best:
            best, ang = d2, math.degrees(math.atan2(dy, dx))
    return ang

def launch_of(drone_idx, default=None):
    """This drone's launch lat/lon, or `default` (LAUNCH_LATLON) if unknown."""
    if LAUNCH_POSITIONS and drone_idx in LAUNCH_POSITIONS:
        return LAUNCH_POSITIONS[drone_idx]
    return default if default is not None else LAUNCH_LATLON

# ═══════════════════════════════════════════════════
#  LAWNMOWER (BOUSTROPHEDON) PATH
# ═══════════════════════════════════════════════════
def inset_cell(cell, buf):
    """Shrink the cell by `buf`, relaxing the inset rather than giving up on
    small cells. Returns the largest usable polygon."""
    if buf <= 0:
        return cell
    for frac in (1.0, 0.5, 0.25):
        try:
            shrunk = cell.buffer(-buf * frac)
        except Exception:
            shrunk = Polygon()
        if not shrunk.is_empty and shrunk.area >= 1:
            if frac < 1.0:
                warn(f'buffer {buf}m too large for cell (area={cell.area:.1f}m²) '
                     f'— reduced to {buf*frac:.2f}m')
            return shrunk
    warn(f'buffer {buf}m too large for cell (area={cell.area:.1f}m²) '
         f'— flying without inset')
    return cell


def generate_lawnmower(cell, spacing, buf, rlat, rlon):
    """Boustrophedon sweep of one cell, returned as [(lat, lon), ...]."""
    if cell.is_empty or cell.area < 1:
        return []

    inner = inset_cell(cell, buf)
    if isinstance(inner, MultiPolygon):
        inner = max(inner.geoms, key=lambda g: g.area)
    coords = list(inner.exterior.coords)[:-1]
    if len(coords) < 3:
        return []

    # Rotate so the longest edge is horizontal, then sweep in +y
    sweep_ang = longest_edge_angle(coords)
    rot = affinity.rotate(inner, -sweep_ang, origin=(0, 0))
    bx, by, mx, my = rot.bounds
    height = my - by

    # At least one sweep line, and never wider than `spacing` (guarantees coverage
    # of cells thinner than one swath instead of returning an empty path).
    n_lines = max(1, int(math.ceil(height / spacing))) if height > 1e-9 else 1
    step = height / n_lines
    offsets = [by + step * (k + 0.5) for k in range(n_lines)]

    path, reverse = [], False
    for y in offsets:
        hit = LineString([(bx-1, y), (mx+1, y)]).intersection(rot)
        if hit.is_empty:
            continue
        if hit.geom_type == 'LineString':
            segs = [list(hit.coords)]
        elif hit.geom_type == 'MultiLineString':
            segs = [list(p.coords) for p in hit.geoms]
        else:
            continue
        if reverse:                       # alternate direction each pass
            segs.reverse()
            for s in segs:
                s.reverse()
        for s in segs:
            path.extend(s)
        reverse = not reverse

    if not path:
        # Degenerate cell (sliver) — still visit it once at its centroid
        c = rot.centroid
        path = [(c.x, c.y)]
        dbg(f'cell area={cell.area:.1f}m² too thin to sweep — single centroid waypoint')

    ca, sa = math.cos(math.radians(sweep_ang)), math.sin(math.radians(sweep_ang))
    dbg(f'lawnmower: area={cell.area:.0f}m² lines={n_lines} step={step:.1f}m '
        f'heading={sweep_ang:.0f}° wps={len(path)}')
    return [xy2ll(x*ca - y*sa, x*sa + y*ca, rlat, rlon) for x, y in path]


def orient_paths(paths):
    """Flip each path so it starts at whichever end is nearer that drone's pad."""
    for i, path in enumerate(paths):
        if len(path) < 2:
            continue
        lat, lon = launch_of(i + 1)
        if haversine(lat, lon, *path[-1]) < haversine(lat, lon, *path[0]):
            path.reverse()
            dbg(f'D{i+1} path reversed — far end was closer to the pad')
    return paths

# ═══════════════════════════════════════════════════
#  MISSION BUILD + TIME ESTIMATE
# ═══════════════════════════════════════════════════
def plan_altitudes(n):
    """
    Per-drone vertical/timing profile. Each drone takes off and cruises at
    TRANSIT_ALT, then descends to the common survey altitude ALTITUDE over its
    first waypoint. Launches are staggered by LAUNCH_DELAY.
    """
    return [{'transit_alt': TRANSIT_ALT,
             'coverage_alt': ALTITUDE,
             'launch_delay': i * LAUNCH_DELAY,
             'needs_climb': True}
            for i in range(n)]


def build_mission(path_gps, ap):
    """Turn a lat/lon path into [(lat, lon, alt), ...]. When the drone transits
    high, the first waypoint is repeated: once at transit alt, once at survey alt."""
    if not path_gps:
        return []
    mission = []
    if ap['needs_climb']:
        mission.append((*path_gps[0], ap['transit_alt']))
    for lat, lon in path_gps:
        mission.append((lat, lon, ap['coverage_alt']))
    return mission


def analyze_mission(path):
    """Estimate (time_s, distance_m) for a lawnmower path: straight legs at
    CRUISE_SPEED, turn legs slower, plus yaw and per-waypoint settling."""
    if len(path) < 2:
        return 0., 0.
    d_straight, d_turn, t_yaw, prev_h = 0, 0, 0, None
    for i in range(len(path) - 1):
        d = haversine(path[i][0], path[i][1], path[i+1][0], path[i+1][1])
        h = math.degrees(math.atan2(path[i+1][1] - path[i][1],
                                    path[i+1][0] - path[i][0]))
        if prev_h is not None:
            t_yaw += abs((h - prev_h + 180) % 360 - 180) / YAW_RATE
        prev_h = h
        if i % 2 == 0:
            d_straight += d
        else:
            d_turn += d
    t = (d_straight / CRUISE_SPEED + d_turn / (CRUISE_SPEED * 0.6)
         + t_yaw + len(path) * SETTLING_TIME)
    return t, d_straight + d_turn

# ═══════════════════════════════════════════════════
#  POWER DIAGRAM PARTITION
# ═══════════════════════════════════════════════════
def clip_half(poly, pi, pj, wi, wj):
    """Clip `poly` to the side of the weighted bisector belonging to site i."""
    d = pj - pi
    dist = np.linalg.norm(d)
    if dist < 1e-10:
        return poly
    n = d / dist

    # Scale the clipping half-plane to the polygon, NOT to the site distance.
    # With `far` tied to dist, close-together launch pads produced a tiny clip box
    # that truncated the cells and left most of the geofence unassigned.
    bx, by, mx, my = poly.bounds
    ext = max(mx - bx, my - by, 1.0)
    far = 4.0 * ext + dist

    # Clamp the weighted shift to the polygon scale so tiny `dist` can't blow it up.
    off = (wi - wj) / (2. * dist)
    off = max(-ext, min(ext, off))
    mid = (pi + pj) * 0.5 + n * off

    perp = np.array([-n[1], n[0]])
    clip = Polygon([mid + perp*far, mid - perp*far,
                    mid - perp*far - n*far, mid + perp*far - n*far])
    try:
        r = poly.intersection(clip)
        if isinstance(r, MultiPolygon):
            r = max(r.geoms, key=lambda g: g.area)
        return r if isinstance(r, Polygon) and not r.is_empty else Polygon()
    except Exception:
        return Polygon()


def power_cells(sites, w, fence):
    """Power diagram of `sites` with weights `w`, clipped to the geofence."""
    cells = []
    for i in range(len(sites)):
        cell = fence
        for j in range(len(sites)):
            if i == j:
                continue
            cell = clip_half(cell, sites[i], sites[j], w[i], w[j])
            if cell.is_empty:
                break
        cells.append(cell)
    return cells


def seed_points(n, fence):
    """n well-separated points inside the fence (farthest-point sampling)."""
    bx, by, mx, my = fence.bounds
    step = max(mx-bx, my-by) / max(int(np.sqrt(n*8)), 4)
    pts = []
    x = bx + step*0.5
    while x < mx:
        y = by + step*0.5
        while y < my:
            if fence.contains(Point(x, y)):
                pts.append([x, y])
            y += step
        x += step
    pts = np.array(pts) if pts else np.array([[fence.centroid.x, fence.centroid.y]])

    if len(pts) < n:
        # Tiny / narrow fence — spread n points along its longest axis instead,
        # so we never hand back fewer seeds than drones.
        horiz = (mx - bx) >= (my - by)
        out = []
        for k in range(n):
            f = (k + 0.5) / n
            pt = (Point(bx + (mx-bx)*f, (by+my)/2) if horiz
                  else Point((bx+mx)/2, by + (my-by)*f))
            if not fence.contains(pt):
                pt = fence.exterior.interpolate(fence.exterior.project(pt))
            out.append((pt.x, pt.y))
        dbg(f'seed_points: fence too small for a grid — {n} points along its long axis')
        return out

    if len(pts) == n:
        return [tuple(p) for p in pts]

    sel = [0]
    for _ in range(n - 1):
        d = np.min([np.linalg.norm(pts - pts[s], axis=1) for s in sel], axis=0)
        sel.append(int(np.argmax(d)))
    dbg(f'seed_points: picked {n} of {len(pts)} grid candidates')
    return [tuple(pts[s]) for s in sel]


def balance(seeds, fence, rlat, rlon, spacing, buf, iters=80, tol=0.03):
    """
    Iteratively adjust the power-diagram weights until every drone's mission
    takes the same time. Returns (cells, deviation, paths, times, infos).
    """
    n = len(seeds)
    sites = np.array(seeds)
    w = np.zeros(n)
    dd = [np.linalg.norm(sites[i]-sites[j]) for i in range(n) for j in range(i+1, n)]
    md = np.mean(dd) if dd else 100.
    best_cells, best_dev, best_paths = None, np.inf, []
    best_times, best_infos = np.zeros(n), []
    dbg(f'balance: {n} seeds, mean site spacing {md:.1f}m, tol {tol*100:.0f}%')

    for it in range(iters):
        cells = power_cells(sites, w, fence)

        ok, paths = True, []
        for i in range(n):
            if cells[i].is_empty or cells[i].area < 1:
                ok = False
                paths.append([])
                continue
            # Small cells are fine — generate_lawnmower always returns at least one
            # sweep line, so don't throw the whole partition away.
            paths.append(generate_lawnmower(cells[i], spacing, buf, rlat, rlon))
        if not ok:
            dbg(f'iter {it:2d}: degenerate cell — damping weights and retrying')
            w *= 0.6
            w -= w.mean()
            continue

        paths = orient_paths(paths)

        times, infos = np.zeros(n), []
        for i in range(n):
            t_cover, d_cover = analyze_mission(paths[i])
            if paths[i]:
                lat, lon = launch_of(i + 1)
                t_travel = haversine(lat, lon, *paths[i][0]) / TRANSIT_SPEED
            else:
                err(f'Path list for drone {i+1} is empty after lawnmower generation.')
                t_travel = 0
            t_delay = i * LAUNCH_DELAY          # this drone waits on the pad
            times[i] = (t_delay if BALANCE_ON_LANDING_TIME else 0) + t_travel + t_cover
            infos.append({'t_delay': t_delay, 't_travel': t_travel,
                          't_coverage': t_cover, 'dist': d_cover,
                          'n_wp': len(paths[i]), 'area': cells[i].area})

        avg = times.mean()
        dev = np.abs(times - avg).max() / avg if avg > 0 else 0
        if dev < best_dev:
            best_dev, best_cells = dev, list(cells)
            best_paths, best_times, best_infos = paths, times.copy(), infos
        dbg(f'iter {it:2d}: dev={dev*100:5.1f}%  best={best_dev*100:5.1f}%  '
            f'areas={[round(c.area) for c in cells]}  '
            f'times={[round(float(t), 1) for t in times]}')
        if dev < tol:
            dbg(f'balance: converged at iteration {it} (dev {dev*100:.1f}% < '
                f'{tol*100:.0f}%)')
            break

        # Push weight from the slow drones towards the fast ones, with decay
        g = times - avg
        gm = np.abs(g).max()
        if gm > 1e-10:
            w -= md**2 * 0.08 * (0.95**it) * (g / gm)
        w -= w.mean()

    return best_cells or cells, best_dev, best_paths, best_times, best_infos

# ═══════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════
COL  = ['#1D9E75','#534AB7','#D85A30','#D4537E','#378ADD','#639922']
FILL = ['#E1F5EE','#EEEDFE','#FAECE7','#FBEAF0','#E6F1FB','#EAF3DE']

def path_xy(path, rlat, rlon):
    """Path in lat/lon -> ([east...], [north...]) for plotting."""
    pts = [ll2xy(la, lo, rlat, rlon) for la, lo in path]
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_mission(cells, paths, infos, times, fence, rlat, rlon, alt_plans, save_path):
    """Render the partition, the paths and the timing summary to a PNG."""
    n = len(cells)
    fig = plt.figure(figsize=(18, 9))

    # ── map ───────────────────────────────────────────────────────────────
    ax = fig.add_axes([0.04, 0.06, 0.48, 0.86])
    gx, gy = fence.exterior.xy
    ax.plot(gx, gy, 'k-', lw=2.2)

    for i in range(n):
        c = COL[i % len(COL)]
        lat, lon = launch_of(i + 1, default=(rlat, rlon))
        lx, ly = ll2xy(lat, lon, rlat, rlon)
        ax.plot(lx, ly, 's', color=c, ms=12, mec='white', mew=1.5, zorder=10)
        ax.annotate(f'LAUNCH_{i+1}', (lx, ly),
                    fontsize=7, fontweight='bold', ha='center', color=c,
                    va='top', xytext=(0, -10), textcoords='offset points', zorder=10,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=c, alpha=0.85))
        if paths[i]:
            pxs, pys = path_xy(paths[i], rlat, rlon)
            ls = '--' if alt_plans[i]['needs_climb'] else '-'
            ax.annotate('', xy=(pxs[0], pys[0]), xytext=(lx, ly),
                        arrowprops=dict(arrowstyle='->', color=c, lw=1.8, ls=ls, alpha=0.6))

    for i in range(n):
        if cells[i].is_empty:
            continue
        c, f = COL[i % len(COL)], FILL[i % len(FILL)]
        cx, cy = cells[i].exterior.xy
        ax.fill(cx, cy, alpha=0.2, color=f, ec=c, lw=1.5)

        if paths[i]:
            pxs, pys = path_xy(paths[i], rlat, rlon)
            ap = alt_plans[i]
            ax.plot(pxs, pys, '-', color=c, lw=0.6, alpha=0.7)

            lbl = f'D{i+1}\nSTART'
            if ap['needs_climb']:
                lbl += f'\n{ap["transit_alt"]:.0f}m→{ap["coverage_alt"]:.0f}m'
            ax.plot(pxs[0], pys[0], 'o', color=c, ms=9, mec='white', mew=1.5, zorder=8)
            ax.annotate(lbl, (pxs[0], pys[0]), fontsize=6, fontweight='bold', color=c,
                        ha='center', va='bottom', xytext=(0, 8),
                        textcoords='offset points', zorder=9)
            ax.plot(pxs[-1], pys[-1], 'X', color=c, ms=8, mew=2, zorder=8)
            ax.annotate('END', (pxs[-1], pys[-1]), fontsize=5.5, color=c, ha='center',
                        va='bottom', xytext=(0, 6), textcoords='offset points', zorder=9)

            if ap['needs_climb']:   # transit altitude tag, halfway out to the cell
                ax.text(pxs[0]/2, pys[0]/2, f'{ap["transit_alt"]:.0f}m', fontsize=6.5,
                        color=c, ha='center', va='bottom', fontweight='bold', alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.15', fc='yellow', ec=c,
                                  alpha=0.6, lw=0.5))

            mid = len(pxs) // 2     # direction-of-travel arrow
            if mid + 1 < len(pxs):
                ax.annotate('', xy=(pxs[mid+1], pys[mid+1]), xytext=(pxs[mid], pys[mid]),
                            arrowprops=dict(arrowstyle='->', color=c, lw=2.5, alpha=0.4))

        info = infos[i] if i < len(infos) else {}
        cent = cells[i].centroid
        ax.text(cent.x, cent.y,
                f'D{i+1}\n{info.get("area",0):.0f}m²\n{info.get("n_wp",0)} wps',
                fontsize=7, ha='center', va='center', color=c, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=c, alpha=0.75, lw=0.5))

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.1)
    ax.set_title(f'Multi-drone coverage — {n} drones @ {CRUISE_SPEED} m/s survey / '
                 f'{TRANSIT_SPEED} m/s transit', fontsize=13, fontweight='bold')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')

    # ── summary table ─────────────────────────────────────────────────────
    ax2 = fig.add_axes([0.56, 0.5, 0.42, 0.42])
    ax2.axis('off')
    hdr = (f'{"Drone":>6}{"Area":>8}{"WPs":>6}{"Wait":>7}{"Travel":>8}{"Cover":>8}'
           f'{"Total":>8}{"Dist":>8}\n{"─"*60}\n')
    body = ''.join(
        f'  D{i+1:<4}{infos[i].get("area",0):>7.0f}{infos[i].get("n_wp",0):>6}'
        f'{infos[i].get("t_delay",0):>6.0f}s{infos[i].get("t_travel",0):>7.1f}s'
        f'{infos[i].get("t_coverage",0):>7.1f}s'
        f'{times[i]:>7.1f}s{infos[i].get("dist",0):>7.0f}m\n' for i in range(n))
    tots = [t for t in times if t > 0]
    foot = (f'\n{"─"*61}\n  Max: {max(tots):.1f}s ({max(tots)/60:.1f}min)\n'
            f'  Min: {min(tots):.1f}s ({min(tots)/60:.1f}min)\n'
            f'  Dev: {max(tots)-min(tots):.1f}s '
            f'({(max(tots)-min(tots))/np.mean(tots)*100:.1f}%)\n'
            f'\n  Survey: {CRUISE_SPEED} m/s @ {ALTITUDE}m\n'
            f'  Transit: {TRANSIT_SPEED} m/s @ {TRANSIT_ALT}m '
            f'(+{ALT_SEPARATION}m per drone)\n'
            f'  HFOV: {HFOV}° | Buffer: {BUFFER_WIDTH}m\n'
            f'  Launch stagger: {LAUNCH_DELAY:.0f}s per drone\n'
            f'  Area: {fence.area:.0f}m² ({fence.area/10000:.2f}ha)\n') if tots else ''
    ax2.text(0, 1, hdr + body + foot, transform=ax2.transAxes, fontsize=9,
             fontfamily='monospace', va='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='#f8f8f6', ec='#ccc', alpha=0.95))

    # ── time breakdown ────────────────────────────────────────────────────
    if tots:
        ax3 = fig.add_axes([0.58, 0.06, 0.38, 0.36])
        y = np.arange(n)
        dl = [infos[i].get('t_delay', 0) for i in range(n)]
        tr = [infos[i].get('t_travel', 0) for i in range(n)]
        co = [infos[i].get('t_coverage', 0) for i in range(n)]
        cs = [COL[i % len(COL)] for i in range(n)]
        ax3.barh(y, dl, height=0.5, color='#dddddd', ec='#999', lw=0.8,
                 label='Launch wait')
        ax3.barh(y, tr, height=0.5, left=dl, color=[c+'55' for c in cs], ec=cs, lw=0.8,
                 label='Travel')
        ax3.barh(y, co, height=0.5, left=[dl[i]+tr[i] for i in range(n)], color=cs,
                 alpha=0.5, ec=cs, lw=0.8, label='Coverage')
        ax3.axvline(np.mean(tots), color='gray', lw=0.8, ls=':', alpha=0.6, label='Avg')
        ax3.set_yticks(y)
        ax3.set_yticklabels([f'D{i+1}' for i in range(n)], fontweight='bold')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Mission time (real lawnmower estimate)',
                      fontsize=11, fontweight='bold')
        ax3.legend(fontsize=8, loc='lower right')
        ax3.grid(axis='x', alpha=0.15)

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    log(f'Plot → {save_path}')

# ═══════════════════════════════════════════════════
#  EXPORT + TRANSFER
# ═══════════════════════════════════════════════════
def export_missions(missions, out_dir, alt_plans=None):
    """Write one QGC WPL 110 .waypoints file per drone."""
    os.makedirs(out_dir, exist_ok=True)
    exported = []

    for i, mission in enumerate(missions):
        if not mission:
            continue
        drone_idx = i + 1
        fp = os.path.join(out_dir, f'drone_{drone_idx}.waypoints')
        home_lat, home_lon = launch_of(drone_idx)

        ap = alt_plans[i] if alt_plans else None
        takeoff_alt = ap['transit_alt'] if ap else mission[0][2]
        delay = ap['launch_delay'] if ap else 0.0

        with open(fp, 'w') as f:
            f.write('QGC WPL 110\n')
            f.write(f'0\t1\t0\t16\t0\t0\t0\t0\t'
                    f'{home_lat:.6f}\t{home_lon:.6f}\t0\t1\n')
            k = 1
            if delay > 0:
                # MAV_CMD_NAV_DELAY (93): hold on the pad for the staggered launch slot
                f.write(f'{k}\t0\t3\t93\t{delay:.6f}\t-1\t-1\t-1\t0\t0\t0\t1\n')
                k += 1
            # MAV_CMD_NAV_TAKEOFF (22): climb straight to the transit layer
            f.write(f'{k}\t0\t3\t22\t0.000000\t0.000000\t0.000000\t0.000000\t'
                    f'{home_lat:.6f}\t{home_lon:.6f}\t{takeoff_alt:.6f}\t1\n')
            k += 1
            for j, (lat, lon, alt) in enumerate(mission):   # MAV_CMD_NAV_WAYPOINT (16)
                f.write(f'{k+j}\t0\t3\t16\t0.000000\t0.000000\t0.000000\t0.000000\t'
                        f'{lat:.6f}\t{lon:.6f}\t{alt:.6f}\t1\n')
            # MAV_CMD_NAV_RETURN_TO_LAUNCH (20)
            f.write(f'{k+len(mission)}\t0\t3\t20\t0\t0\t0\t0\t0\t0\t0\t1\n')

        a0, a1 = mission[0][2], mission[-1][2]
        alt_s = f'{a0:.0f}→{a1:.0f}m' if a0 != a1 else f'{a1:.0f}m'
        log(f'  {fp}  ({len(mission)} wps, alt={alt_s}, launch +{delay:.0f}s)')
        exported.append(fp)

    return exported


def export_missions_csv(missions, out_dir, alt_plans=None):
    """Write one tab-separated mission CSV per drone (the format the GCS reads)."""
    os.makedirs(out_dir, exist_ok=True)
    exported = []

    for i, mission in enumerate(missions):
        if not mission:
            continue
        drone_idx = i + 1
        fp = os.path.join(out_dir, f'drone_{drone_idx}_mission.csv')
        home_lat, home_lon = launch_of(drone_idx)

        ap = alt_plans[i] if alt_plans else None
        delay = ap['launch_delay'] if ap else 0.0
        takeoff_alt = ap['transit_alt'] if ap else mission[0][2]

        with open(fp, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['drone_id', 'wp_index', 'lat', 'lon', 'alt_m',
                             'wp_type', 'delay_s'])
            writer.writerow([drone_idx, 0, f'{home_lat:.7f}', f'{home_lon:.7f}', 0,
                             'HOME', f'{delay:.1f}'])
            # delay_s on the TAKEOFF row = hold on the pad this long before lifting off
            writer.writerow([drone_idx, 1, f'{home_lat:.7f}', f'{home_lon:.7f}',
                             f'{takeoff_alt:.2f}', 'TAKEOFF', f'{delay:.1f}'])
            for j, (lat, lon, alt) in enumerate(mission):
                # first row is the descent point: reached at transit altitude,
                # the next row repeats it at the survey altitude
                kind = 'TRANSIT' if (ap and ap['needs_climb'] and j == 0) else 'COVERAGE'
                writer.writerow([drone_idx, j + 2, f'{lat:.7f}', f'{lon:.7f}',
                                 f'{alt:.2f}', kind, '0.0'])
            writer.writerow([drone_idx, len(mission) + 2, '', '', '', 'RTL', '0.0'])

        log(f'CSV  → {fp}')
        dbg(f'D{drone_idx} CSV: {len(mission)+3} rows, takeoff {takeoff_alt:.1f}m, '
            f'delay {delay:.0f}s')
        exported.append(fp)

    return exported


def send_file_over_socket(filepath, target_ip, target_port):
    """Ship one mission file to a drone: name length, name, data length, data."""
    filename = os.path.basename(filepath).encode()
    with open(filepath, 'rb') as f:
        data = f.read()
    dbg(f'sending {len(data)}B of {filename.decode()} to {target_ip}:{target_port}')

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((target_ip, target_port))
        s.sendall(len(filename).to_bytes(4, 'big'))
        s.sendall(filename)
        s.sendall(len(data).to_bytes(8, 'big'))
        s.sendall(data)

        ack = s.recv(1)
        if ack == b'\x01':
            log(f'Drone {target_ip} ACK received — file delivered')
        else:
            warn(f'No ACK from {target_ip}:{target_port}')

    log(f'Sent {filepath} → {target_ip}:{target_port}')

# ═══════════════════════════════════════════════════
#  PLANNER STEPS
# ═══════════════════════════════════════════════════
def resolve_origin(geofence_latlon):
    """Coordinate origin: drone 1's pad if we have telemetry, else fence centroid."""
    if LAUNCH_POSITIONS:
        rlat, rlon = LAUNCH_POSITIONS[1]
        log(f'Launch origin (D1): lat={rlat:.7f}  lon={rlon:.7f}')
        for idx, (lat, lon) in sorted(LAUNCH_POSITIONS.items()):
            log(f'D{idx} launch: lat={lat:.7f}  lon={lon:.7f}')
        return rlat, rlon

    rlat = sum(p[0] for p in geofence_latlon) / len(geofence_latlon)
    rlon = sum(p[1] for p in geofence_latlon) / len(geofence_latlon)
    warn('No launch positions provided — using geofence centroid as origin')
    return rlat, rlon


def choose_seeds(fence, rlat, rlon):
    """One partition seed per drone: its own launch pad where that works,
    otherwise well-spread points assigned to the nearest pad."""
    if not (LAUNCH_POSITIONS and len(LAUNCH_POSITIONS) == N_DRONES):
        dbg('no per-drone launch positions — seeding geometrically')
        return seed_points(N_DRONES, fence)

    seeds = []
    for idx in range(1, N_DRONES + 1):
        x, y = ll2xy(*LAUNCH_POSITIONS[idx], rlat, rlon)
        pt = Point(x, y)
        if not fence.contains(pt):
            nearest = fence.exterior.interpolate(fence.exterior.project(pt))
            x, y = nearest.x, nearest.y
            warn(f'D{idx} launch pad outside geofence — snapped to boundary')
        seeds.append((x, y))

    # Launch pads that sit (nearly) on top of each other make the power diagram
    # degenerate — bisector directions become noise and cells collapse, leaving
    # part of the geofence uncovered. Require a sensible minimum separation.
    min_sep = 0.25 * math.sqrt(fence.area / max(N_DRONES, 1))
    pad_sep = min((math.dist(seeds[i], seeds[j])
                   for i in range(N_DRONES) for j in range(i + 1, N_DRONES)),
                  default=float('inf'))
    dbg(f'pad separation {pad_sep:.1f}m vs required {min_sep:.1f}m')

    if pad_sep < min_sep:
        warn(f'launch pads only {pad_sep:.1f}m apart (need ~{min_sep:.1f}m) '
             f'— using spread seeds, assigned to the nearest pad')
        spread = seed_points(N_DRONES, fence)
        # greedy nearest-pad assignment so each drone keeps the cell closest to it
        assigned, taken = [], set()
        for pad in seeds:
            best_k = min((k for k in range(len(spread)) if k not in taken),
                         key=lambda k: math.dist(pad, spread[k]))
            taken.add(best_k)
            assigned.append(spread[best_k])
        seeds = assigned

    return seeds


def build_missions(paths, alt_plans):
    """Attach altitudes to every path; an empty path yields an empty mission."""
    missions = []
    for i in range(N_DRONES):
        mission = build_mission(paths[i], alt_plans[i])
        if not mission:
            warn(f'Mission for drone {i+1} is empty (no waypoints).')
        missions.append(mission)
    return missions


def print_summary(infos, times, alt_plans):
    log('─' * 55)
    for i in range(N_DRONES):
        info = infos[i]
        log(f'D{i+1}: {info["area"]:.0f}m² | {info["n_wp"]} wps | '
            f'wait={info.get("t_delay",0):.0f}s  travel={info["t_travel"]:.1f}s  '
            f'cover={info["t_coverage"]:.1f}s  total={times[i]:.1f}s | '
            f'{alt_plans[i]["transit_alt"]:.0f}m→{ALTITUDE:.0f}m')
    log('─' * 55)

    tots = [t for t in times if t > 0]
    if tots:
        log(f'Range: {min(tots):.1f}–{max(tots):.1f}s '
            f'(Δ{max(tots)-min(tots):.1f}s / '
            f'{(max(tots)-min(tots))/np.mean(tots)*100:.1f}%)')


def transfer_missions(csv_paths):
    """Send each drone its CSV; a missing IP/port entry is reported, not fatal."""
    for i, path in enumerate(csv_paths):
        try:
            send_file_over_socket(path, TARGET_IP[i], TARGET_PORT[i])
            print(f'Sent CSV mission for Drone {i+1} to {TARGET_IP[i]}:{TARGET_PORT[i]}')
        except Exception as e:
            err(f'Failed to send CSV mission for Drone {i+1}: {e}')

# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════
def main():
    global LAUNCH_POSITIONS, LAUNCH_LATLON

    log('MULTI-DRONE COVERAGE PLANNER')
    log(f'{N_DRONES} drones | survey {CRUISE_SPEED} m/s @ {ALTITUDE}m | '
        f'transit {TRANSIT_SPEED} m/s @ {TRANSIT_ALT}m +{ALT_SEPARATION}m/drone')

    # ── launch positions from the telemetry streams ──
    udp_positions = read_all_drone_positions()
    if udp_positions:
        LAUNCH_POSITIONS = udp_positions

    rlat, rlon = resolve_origin(GEOFENCE)
    LAUNCH_LATLON = (rlat, rlon)

    # ── geofence in local XY about that origin ──
    fence = Polygon([ll2xy(la, lo, rlat, rlon) for la, lo in GEOFENCE])
    if not fence.is_valid:
        fence = fence.buffer(0)
        dbg('geofence polygon was self-intersecting — repaired with buffer(0)')

    spacing = 2 * ALTITUDE * math.tan(math.radians(HFOV / 2))
    print()
    log(f'Geofence: {fence.area:.0f}m²')
    log(f'Sweep spacing: {spacing:.1f}m')

    # ── partition + lawnmower paths ──
    seeds = choose_seeds(fence, rlat, rlon)
    log(f'Seeds: {[(round(float(x), 1), round(float(y), 1)) for x, y in seeds]}')

    cells, dev, paths, times, infos = balance(
        seeds, fence, rlat, rlon, spacing, BUFFER_WIDTH)
    log(f'Deviation: {dev*100:.1f}%')

    # ── altitude + launch-stagger profile ──
    alt_plans = plan_altitudes(N_DRONES)
    print()
    log('Launch stagger + transit layers:')
    for i, ap in enumerate(alt_plans):
        log(f'  D{i+1}: t+{ap["launch_delay"]:.0f}s  takeoff+cruise @ '
            f'{ap["transit_alt"]:.0f}m → descend to {ap["coverage_alt"]:.0f}m for survey')

    missions = build_missions(paths, alt_plans)
    print()
    print_summary(infos, times, alt_plans)

    # ── outputs ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_mission(cells, paths, infos, times, fence, rlat, rlon,
                 alt_plans, os.path.join(OUTPUT_DIR, 'mission_plan.png'))

    if EXPORT_WAYPOINTS:
        log(f'Exporting .waypoints to {OUTPUT_DIR}/:')
        export_missions(missions, OUTPUT_DIR, alt_plans)

    csv_paths = export_missions_csv(missions, OUTPUT_DIR, alt_plans)
    transfer_missions(csv_paths)
    dbg(f'planner finished in {time.perf_counter() - _T0:.2f}s')


if __name__ == '__main__':
    main()
