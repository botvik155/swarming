"""
Multi-Drone Area Coverage Planner
===================================
- Partitions geofence via Power Diagram (balanced by real lawnmower time)
- Generates lawnmower path per drone
- Detects transit conflicts → altitude separation
- Exports per-drone .waypoints files (QGC WPL 110)
- Saves plot as PNG (no GUI window)

Usage:  python multi_drone.py
"""

import math
import os
import numpy as np
from pymavlink import mavutil
import matplotlib
matplotlib.use('Agg')  # no GUI — saves to file only
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely import affinity
import csv
import socket


filepath = "oldtest_uas.txt"

TARGET_IP = ['192.168.1.126']
TARGET_PORT = [9000]
N_DRONES       = 1
ALTITUDE       = 6
ALT_SEPARATION = 5.0
HFOV           = 87.0
BUFFER_WIDTH   = 2.0
CRUISE_SPEED   = 2.0
YAW_RATE       = 45.0
SETTLING_TIME  = 1.5
OUTPUT_DIR     = '/home/akshit/uav/swarming_guided'
LAUNCH_LATLON     = (0.0, 0.0)   # populated at runtime from MAVLink telemetry
LAUNCH_POSITIONS  = {}            # populated at runtime from MAVLink telemetry — {drone_index: (lat, lon)}

# ═══════════════════════════════════════════════════
#  MAVLink TELEMETRY GPS READER
#  Drone 1 streams on UDP port 14553
#  Drone 2 streams on UDP port 14554
# ═══════════════════════════════════════════════════
DRONE_UDP_PORTS = {
    1: 14553,   # Drone 1 telemetry port
    # 2: 14554,   # Drone 2 telemetry port
}
MAVLINK_TIMEOUT = 10.0   # seconds to wait per drone for a valid GPS fix

def read_gps_from_mavlink(drone_id, port, timeout=MAVLINK_TIMEOUT):
    """
    Connect to a MAVLink telemetry stream on udp:0.0.0.0:<port> and read
    the first valid GLOBAL_POSITION_INT message (MAVLink msg ID 33).

    MAVLink GLOBAL_POSITION_INT fields used:
        lat  — latitude  in 1e-7 degrees  (int32)
        lon  — longitude in 1e-7 degrees  (int32)

    Returns:
        (lat, lon) as floats, or None on timeout / error.
    """
    connection_str = f'udpin:0.0.0.0:{port}'
    print(f'  [MAVLink] D{drone_id} — connecting on {connection_str} '
          f'(timeout={timeout}s) ...')
    try:
        mav = mavutil.mavlink_connection(connection_str, input=True)
        mav.wait_heartbeat(timeout=timeout)
        print(f'  [MAVLink] D{drone_id} — heartbeat received '
              f'(sysid={mav.target_system})')

        # Wait for GLOBAL_POSITION_INT which carries the live GPS fix
        msg = mav.recv_match(type='GLOBAL_POSITION_INT',
                             blocking=True, timeout=timeout)
        if msg is None:
            print(f'  [MAVLink] D{drone_id} — timeout waiting for '
                  f'GLOBAL_POSITION_INT on port {port}. '
                  f'Falling back to geofence centroid for this drone.')
            return None

        lat = msg.lat / 1e7   # convert from 1e-7 deg to degrees
        lon = msg.lon / 1e7
        print(f'  [MAVLink] D{drone_id} GPS -> lat={lat:.7f}  lon={lon:.7f}')
        return (lat, lon)

    except Exception as e:
        print(f'  [MAVLink] D{drone_id} — error on port {port}: {e}. '
              f'Falling back to geofence centroid for this drone.')
        return None


def read_all_drone_positions(drone_ports=DRONE_UDP_PORTS, timeout=MAVLINK_TIMEOUT):
    """
    Read initial GPS positions for all drones from their individual
    MAVLink telemetry UDP ports sequentially.

    Returns:
        dict  {drone_id (int): (lat, lon)}  — only drones that responded.
        Empty dict triggers geofence-centroid fallback in main().
    """
    positions = {}
    for drone_id, port in sorted(drone_ports.items()):
        result = read_gps_from_mavlink(drone_id, port, timeout)
        if result is not None:
            positions[drone_id] = result
    return positions

def parse_waypoints(file_path):
    geofence = []
    launch = None

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip header (first line)
    for i, line in enumerate(lines[1:]):
        parts = line.strip().split('\t')

        if len(parts) < 11:
            continue  # skip malformed lines

        lat = float(parts[8])
        lon = float(parts[9])

        geofence.append((lat, lon))

    return geofence

# ═══════════════════════════════════════════════════
#  COORDINATE MATH
# ═══════════════════════════════════════════════════
GEOFENCE = parse_waypoints(filepath)
R_EARTH = 6_371_000.0

def ll2xy(lat, lon, rlat, rlon):
    c = math.cos(math.radians(rlat))
    return (math.radians(lon - rlon) * R_EARTH * c,
            math.radians(lat - rlat) * R_EARTH)

def xy2ll(x, y, rlat, rlon):
    c = math.cos(math.radians(rlat))
    return (rlat + math.degrees(y / R_EARTH),
            rlon + math.degrees(x / (R_EARTH * c)))

def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R_EARTH * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def longest_edge_angle(pts):
    best, ang = 0, 0
    for i in range(len(pts)):
        dx = pts[(i+1) % len(pts)][0] - pts[i][0]
        dy = pts[(i+1) % len(pts)][1] - pts[i][1]
        d2 = dx*dx + dy*dy
        if d2 > best:
            best, ang = d2, math.degrees(math.atan2(dy, dx))
    return ang

def generate_lawnmower(cell, spacing, buf, rlat, rlon):
    if cell.is_empty or cell.area < 1: return []
    try:
        inner = cell.buffer(-buf)
        if inner.is_empty or inner.area < 1:
            raise ValueError(f"Buffer ({buf}m) is too large for this cell (area={cell.area:.1f}m²); drone will not fly on geofence boundary.")
    except ValueError as e:
        print(f"  WARNING: {e}")
        inner = cell
    if isinstance(inner, MultiPolygon): inner = max(inner.geoms, key=lambda g: g.area)
    coords = list(inner.exterior.coords)[:-1]
    if len(coords) < 3: return []

    sa = longest_edge_angle(coords)
    rot = affinity.rotate(inner, -sa, origin=(0, 0))
    bx, by, mx, my = rot.bounds
    path, y, rev = [], by + spacing/2, False

    while y < my:
        hit = LineString([(bx-1, y), (mx+1, y)]).intersection(rot)
        if not hit.is_empty:
            segs = []
            if hit.geom_type == 'LineString': segs.append(list(hit.coords))
            elif hit.geom_type == 'MultiLineString':
                for p in hit.geoms: segs.append(list(p.coords))
            if rev:
                segs.reverse()
                for s in segs: s.reverse()
            for s in segs: path.extend(s)
            rev = not rev
        y += spacing

    if not path: return []
    ca, sa2 = math.cos(math.radians(sa)), math.sin(math.radians(sa))
    return [xy2ll(x*ca - y*sa2, x*sa2 + y*ca, rlat, rlon) for x, y in path]



def orient_paths_collision_safe(paths, cells, rlat, rlon):
    n = len(paths)
    # Always choose start point as the path endpoint closest to each drone's own launch pad
    for i in range(n):
        drone_idx = i + 1
        if LAUNCH_POSITIONS and drone_idx in LAUNCH_POSITIONS:
            launch_ll = LAUNCH_POSITIONS[drone_idx]
        else:
            launch_ll = LAUNCH_LATLON
        if len(paths[i]) >= 2:
            if haversine(launch_ll[0], launch_ll[1], paths[i][-1][0], paths[i][-1][1]) < \
               haversine(launch_ll[0], launch_ll[1], paths[i][0][0], paths[i][0][1]):
                paths[i].reverse()
    return paths

def plan_altitudes(n: int) -> list:
    """
    No drone delay — all drones take off together at the same altitude.
    No transit conflict detection needed.
    """
    return [{'transit_alt': ALTITUDE, 'coverage_alt': ALTITUDE,
             'conflicts': [], 'needs_climb': False}
            for _ in range(n)]

def build_mission(path_gps, ap):
    if not path_gps: return []
    m = []
    if ap['needs_climb']:
        m.append((path_gps[0][0], path_gps[0][1], ap['transit_alt']))
        m.append((path_gps[0][0], path_gps[0][1], ap['coverage_alt']))
        for lat, lon in path_gps[1:]:
            m.append((lat, lon, ap['coverage_alt']))
    else:
        for lat, lon in path_gps:
            m.append((lat, lon, ap['coverage_alt']))
    return m



def analyze_mission(path):
    if len(path) < 2: return 0., 0.
    t_trans, t_turn, t_yaw, prev_h = 0, 0, 0, None
    for i in range(len(path)-1):
        d = haversine(path[i][0], path[i][1], path[i+1][0], path[i+1][1])
        h = math.degrees(math.atan2(path[i+1][1]-path[i][1], path[i+1][0]-path[i][0]))
        if prev_h is not None:
            t_yaw += abs((h - prev_h + 180) % 360 - 180) / YAW_RATE
        prev_h = h
        if i % 2 == 0: t_trans += d
        else: t_turn += d
    t = t_trans/CRUISE_SPEED + t_turn/(CRUISE_SPEED*0.6) + t_yaw + len(path)*SETTLING_TIME
    return t, t_trans + t_turn



def clip_half(poly, pi, pj, wi, wj):
    d = pj - pi; dist = np.linalg.norm(d)
    if dist < 1e-10: return poly
    n = d/dist
    mid = (pi+pj)*0.5 + n*((wi-wj)/(2.*dist))
    perp = np.array([-n[1], n[0]]); far = dist*12
    clip = Polygon([mid+perp*far, mid-perp*far, mid-perp*far-n*far, mid+perp*far-n*far])
    try:
        r = poly.intersection(clip)
        if isinstance(r, MultiPolygon): r = max(r.geoms, key=lambda g: g.area)
        return r if isinstance(r, Polygon) and not r.is_empty else Polygon()
    except: return Polygon()

def power_cells(sites, w, fence):
    cells = []
    for i in range(len(sites)):
        cell = fence
        for j in range(len(sites)):
            if i != j:
                cell = clip_half(cell, sites[i], sites[j], w[i], w[j])
                if cell.is_empty: break
        cells.append(cell)
    return cells

def seed_points(n, fence):
    bx, by, mx, my = fence.bounds
    step = max(mx-bx, my-by) / max(int(np.sqrt(n*8)), 4)
    pts = []
    x = bx + step*0.5
    while x < mx:
        y = by + step*0.5
        while y < my:
            if fence.contains(Point(x, y)): pts.append([x, y])
            y += step
        x += step
    pts = np.array(pts) if pts else np.array([[fence.centroid.x, fence.centroid.y]])
    if len(pts) <= n: return [tuple(p) for p in pts]
    sel = [0]
    for _ in range(n-1):
        d = np.min([np.linalg.norm(pts - pts[s], axis=1) for s in sel], axis=0)
        sel.append(int(np.argmax(d)))
    return [tuple(pts[s]) for s in sel]

def balance(seeds, fence, rlat, rlon, spacing, buf, iters=80, tol=0.03):
    n = len(seeds); w = np.zeros(n); sites = np.array(seeds)
    dd = [np.linalg.norm(sites[i]-sites[j]) for i in range(n) for j in range(i+1,n)]
    md = np.mean(dd) if dd else 100.
    best, bd, bw, bp, bt, bi = None, np.inf, w.copy(), [], np.zeros(n), []
    MIN_CELL_AREA = spacing * spacing * 2  # must fit at least ~2 sweep strips

    for it in range(iters):
        cells = power_cells(sites, w, fence)
        ok, times, paths = True, np.zeros(n), []
        for i in range(n):
            if cells[i].is_empty or cells[i].area < 1:
                ok = False; paths.append([]); continue
            if cells[i].area < MIN_CELL_AREA:
                print(f"  WARNING: Cell {i+1} area ({cells[i].area:.1f}m²) is too small for lawnmower paths (min {MIN_CELL_AREA:.1f}m²); skipping.")
                ok = False; paths.append([]); continue
            paths.append(generate_lawnmower(cells[i], spacing, buf, rlat, rlon))
        if not ok: w *= 0.6; w -= w.mean(); continue

        paths = orient_paths_collision_safe(paths, cells, rlat, rlon)
        infos = []
        for i in range(n):
            tc, dc = analyze_mission(paths[i])
            try:
                if not paths[i]:
                    raise ValueError(f"Path list for drone {i+1} is empty after lawnmower generation.")
                drone_idx = i + 1
                if LAUNCH_POSITIONS and drone_idx in LAUNCH_POSITIONS:
                    l_lat, l_lon = LAUNCH_POSITIONS[drone_idx]
                else:
                    l_lat, l_lon = LAUNCH_LATLON
                tt = haversine(l_lat, l_lon,
                               paths[i][0][0], paths[i][0][1]) / CRUISE_SPEED
            except ValueError as e:
                print(f"  ERROR: {e}")
                tt = 0
            times[i] = tt + tc
            infos.append({'t_travel': tt, 't_coverage': tc, 'dist': dc,
                          'n_wp': len(paths[i]), 'area': cells[i].area})

        avg = times.mean()
        dev = np.abs(times - avg).max() / avg if avg > 0 else 0
        if dev < bd:
            bd, best, bw, bp, bt, bi = dev, list(cells), w.copy(), paths, times.copy(), infos
        if dev < tol: break
        g = times - avg; gm = np.abs(g).max()
        if gm > 1e-10: w -= md**2 * 0.08 * (0.95**it) * (g/gm)
        w -= w.mean()

    return best or cells, bw, bd, bp, bt, bi

# ═══════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════

COL  = ['#1D9E75','#534AB7','#D85A30','#D4537E','#378ADD','#639922']
FILL = ['#E1F5EE','#EEEDFE','#FAECE7','#FBEAF0','#E6F1FB','#EAF3DE']

def plot_mission(cells, paths, infos, times, fence, rlat, rlon, alt_plans, save_path):
    # uses module-level LAUNCH_POSITIONS for per-drone launch markers
    n = len(cells)
    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_axes([0.04, 0.06, 0.48, 0.86])
    gx, gy = fence.exterior.xy
    ax.plot(gx, gy, 'k-', lw=2.2)
    for i in range(n):
        drone_idx = i + 1
        # Convert launch lat/lon to local XY
        if LAUNCH_POSITIONS and drone_idx in LAUNCH_POSITIONS:
            l_lat, l_lon = LAUNCH_POSITIONS[drone_idx]
        else:
            l_lat, l_lon = rlat, rlon   # fallback to origin
        lx, ly = ll2xy(l_lat, l_lon, rlat, rlon)
        c = COL[i % 6]
        ax.plot(lx, ly, 's', color=c, ms=12, mec='white', mew=1.5, zorder=10)
        ax.annotate(f'LAUNCH_{drone_idx}', (lx, ly),
                    fontsize=7, fontweight='bold', ha='center', color=c,
                    va='top', xytext=(0, -10), textcoords='offset points', zorder=10,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=c, alpha=0.85))
        # Draw arrow from launch to first waypoint
        if paths[i]:
            pxs = [ll2xy(la, lo, rlat, rlon)[0] for la, lo in paths[i]]
            pys = [ll2xy(la, lo, rlat, rlon)[1] for la, lo in paths[i]]
            ap = alt_plans[i]
            ls = '--' if ap['needs_climb'] else '-'
            ax.annotate('', xy=(pxs[0], pys[0]), xytext=(lx, ly),
                        arrowprops=dict(arrowstyle='->', color=c, lw=1.8, ls=ls, alpha=0.6))

    for i in range(n):
        if cells[i].is_empty: continue
        c, f = COL[i%6], FILL[i%6]
        cx, cy = cells[i].exterior.xy
        ax.fill(cx, cy, alpha=0.2, color=f, ec=c, lw=1.5)
        if paths[i]:
            pxs = [ll2xy(la, lo, rlat, rlon)[0] for la, lo in paths[i]]
            pys = [ll2xy(la, lo, rlat, rlon)[1] for la, lo in paths[i]]
            ax.plot(pxs, pys, '-', color=c, lw=0.6, alpha=0.7)
            ap = alt_plans[i]
            lbl = f'D{i+1}\nSTART'
            if ap['needs_climb']: lbl += f'\n{ap["transit_alt"]:.0f}m→{ap["coverage_alt"]:.0f}m'
            ax.plot(pxs[0], pys[0], 'o', color=c, ms=9, mec='white', mew=1.5, zorder=8)
            ax.annotate(lbl, (pxs[0],pys[0]), fontsize=6, fontweight='bold', color=c,
                        ha='center', va='bottom', xytext=(0,8), textcoords='offset points', zorder=9)
            ax.plot(pxs[-1], pys[-1], 'X', color=c, ms=8, mew=2, zorder=8)
            ax.annotate('END', (pxs[-1],pys[-1]), fontsize=5.5, color=c, ha='center',
                        va='bottom', xytext=(0,6), textcoords='offset points', zorder=9)
            if ap['needs_climb']:
                ax.text(pxs[0]/2, pys[0]/2, f'{ap["transit_alt"]:.0f}m', fontsize=6.5,
                        color=c, ha='center', va='bottom', fontweight='bold', alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.15', fc='yellow', ec=c, alpha=0.6, lw=0.5))
            mid = len(pxs)//2
            if mid+1 < len(pxs):
                ax.annotate('', xy=(pxs[mid+1],pys[mid+1]), xytext=(pxs[mid],pys[mid]),
                            arrowprops=dict(arrowstyle='->', color=c, lw=2.5, alpha=0.4))
        cent = cells[i].centroid
        info = infos[i] if i < len(infos) else {}
        ax.text(cent.x, cent.y, f'D{i+1}\n{info.get("area",0):.0f}m²\n{info.get("n_wp",0)} wps',
                fontsize=7, ha='center', va='center', color=c, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=c, alpha=0.75, lw=0.5))

    ax.set_aspect('equal'); ax.grid(True, alpha=0.1)
    ax.set_title(f'Multi-drone coverage — {n} drones @ {CRUISE_SPEED} m/s', fontsize=13, fontweight='bold')
    ax.set_xlabel('East (m)'); ax.set_ylabel('North (m)')

    ax2 = fig.add_axes([0.56, 0.5, 0.42, 0.42]); ax2.axis('off')
    hdr = f'{"Drone":>6}{"Area":>8}{"WPs":>6}{"Travel":>8}{"Cover":>8}{"Total":>8}{"Dist":>8}\n{"─"*53}\n'
    body = ''.join(f'  D{i+1:<4}{infos[i].get("area",0):>7.0f}{infos[i].get("n_wp",0):>6}'
               f'{infos[i].get("t_travel",0):>7.1f}s{infos[i].get("t_coverage",0):>7.1f}s'
               f'{times[i]:>7.1f}s{infos[i].get("dist",0):>7.0f}m\n' for i in range(n))
    tots = [t for t in times if t > 0]
    foot = (f'\n{"─"*61}\n  Max: {max(tots):.1f}s ({max(tots)/60:.1f}min)\n'
            f'  Min: {min(tots):.1f}s ({min(tots)/60:.1f}min)\n'
            f'  Dev: {max(tots)-min(tots):.1f}s ({(max(tots)-min(tots))/np.mean(tots)*100:.1f}%)\n'
            f'\n  Speed: {CRUISE_SPEED} m/s | Alt(of 1st drone): {ALTITUDE}m\n'
            f'  HFOV: {HFOV}° | Buffer: {BUFFER_WIDTH}m\n'
            f'  Area: {fence.area:.0f}m² ({fence.area/10000:.2f}ha)\n') if tots else ''
    ax2.text(0, 1, hdr+body+foot, transform=ax2.transAxes, fontsize=9,
             fontfamily='monospace', va='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='#f8f8f6', ec='#ccc', alpha=0.95))

    if tots:
        ax3 = fig.add_axes([0.58, 0.06, 0.38, 0.36])
        y = np.arange(n)
        tr = [infos[i].get('t_travel',0) for i in range(n)]
        co = [infos[i].get('t_coverage',0) for i in range(n)]
        cs = [COL[i%6] for i in range(n)]
        ax3.barh(y, tr, height=0.5, color=[c+'55' for c in cs], ec=cs, lw=0.8, label='Travel')
        ax3.barh(y, co, height=0.5, left=tr, color=cs, alpha=0.5, ec=cs, lw=0.8, label='Coverage')
        ax3.axvline(np.mean(tots), color='gray', lw=0.8, ls=':', alpha=0.6, label='Avg')
        ax3.set_yticks(y); ax3.set_yticklabels([f'D{i+1}' for i in range(n)], fontweight='bold')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Mission time (real lawnmower estimate)', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=8, loc='lower right'); ax3.grid(axis='x', alpha=0.15)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Plot → {save_path}')


def export_missions(missions, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    exported = []
    for i, m in enumerate(missions):
        if not m: continue
        fp = os.path.join(out_dir, f'drone_{i+1}.waypoints')
        drone_idx = i + 1
        if LAUNCH_POSITIONS and drone_idx in LAUNCH_POSITIONS:
            home_lat, home_lon = LAUNCH_POSITIONS[drone_idx]
        else:
            home_lat, home_lon = LAUNCH_LATLON
        with open(fp, 'w') as f:
            f.write('QGC WPL 110\n')
            f.write(
                f'0\t1\t0\t16\t0\t0\t0\t0\t'
                f'{home_lat:.6f}\t{home_lon:.6f}\t0\t1\n'
            )
            f.write(
                f'1\t0\t3\t22\t0.000000\t0.000000\t0.000000\t0.000000\t'
                f'{home_lat:.6f}\t{home_lon:.6f}\t'
                f'{ALTITUDE + (2*i):.6f}\t1\n'
            )
            for j, (lat, lon, alt) in enumerate(m):
                f.write(f'{j+2}\t0\t3\t16\t0.000000\t0.000000\t0.000000\t0.000000\t'
                        f'{lat:.6f}\t{lon:.6f}\t{alt + (2*i):.6f}\t1\n')
            rtl_index = len(m) + 2

            f.write(
                f'{rtl_index}\t0\t3\t20\t0\t0\t0\t0\t'
                f'0\t0\t0\t1\n'
            )
        a0, a1 = m[0][2], m[-1][2]
        alt_s = f'{a0:.0f}→{a1:.0f}m' if a0 != a1 else f'{a1:.0f}m'
        print(f'    {fp}  ({len(m)} wps, alt={alt_s})')
        exported.append(fp)
    return exported



def export_missions_csv(missions, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    exported = []

    for i, mission in enumerate(missions):
        if not mission:
            continue

        drone_idx = i + 1
        fp = os.path.join(out_dir, f'drone_{drone_idx}_mission.csv')

        if LAUNCH_POSITIONS and drone_idx in LAUNCH_POSITIONS:
            home_lat, home_lon = LAUNCH_POSITIONS[drone_idx]
        else:
            home_lat, home_lon = LAUNCH_LATLON

        with open(fp, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['drone_id', 'wp_index', 'lat', 'lon', 'alt_m', 'wp_type'])

            writer.writerow([drone_idx, 0, f'{home_lat:.7f}', f'{home_lon:.7f}', 0, 'HOME'])

            takeoff_alt = ALTITUDE + (2 * i)
            writer.writerow([drone_idx, 1, f'{home_lat:.7f}', f'{home_lon:.7f}',
                             f'{takeoff_alt:.2f}', 'TAKEOFF'])

            for j, (lat, lon, alt) in enumerate(mission):
                writer.writerow([drone_idx, j + 2, f'{lat:.7f}', f'{lon:.7f}',
                                 f'{alt + (2 * i):.2f}', 'COVERAGE'])

            writer.writerow([drone_idx, len(mission) + 2, '', '', '', 'RTL'])

        print(f'  CSV  → {fp}')
        exported.append(fp)

    return exported


def send_file_over_socket(filepath, target_ip, target_port):
    filename = os.path.basename(filepath).encode()
    with open(filepath, 'rb') as f:
        data = f.read()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((target_ip, target_port))
        s.sendall(len(filename).to_bytes(4, 'big'))
        s.sendall(filename)
        s.sendall(len(data).to_bytes(8, 'big'))
        s.sendall(data)

        ack = s.recv(1)
        if ack == b'\x01':
            print(f'  Drone {target_ip} ACK received — file delivered')
        else:
            print(f'  WARNING: No ACK from {target_ip}:{target_port}')

    print(f'  Sent {filepath} → {target_ip}:{target_port}')

def main():
    print(f'  MULTI-DRONE COVERAGE PLANNER')
    print(f'  {N_DRONES} drones | {CRUISE_SPEED} m/s | alt={ALTITUDE}m')

    # ── Fetch drone GPS from MAVLink telemetry streams ──
    global LAUNCH_POSITIONS, LAUNCH_LATLON
    udp_positions = read_all_drone_positions()
    if udp_positions:
        LAUNCH_POSITIONS = udp_positions

    # Use launch positions received from MAVLink telemetry.
    # If none available, LAUNCH_POSITIONS will be empty — fall back to GEOFENCE centroid
    GEOFENCE_LATLON = GEOFENCE

    if LAUNCH_POSITIONS:
        # Use drone 1's position as the coordinate reference origin
        rlat, rlon = LAUNCH_POSITIONS[1]
        print(f'  Launch origin (D1): lat={rlat:.7f}  lon={rlon:.7f}')
        for idx, (lat, lon) in sorted(LAUNCH_POSITIONS.items()):
            print(f'  D{idx} launch: lat={lat:.7f}  lon={lon:.7f}')
    else:
        # Standalone fallback — use centroid of geofence as origin
        rlat = sum(p[0] for p in GEOFENCE_LATLON) / len(GEOFENCE_LATLON)
        rlon = sum(p[1] for p in GEOFENCE_LATLON) / len(GEOFENCE_LATLON)
        print(f'  WARNING: No launch positions provided — using geofence centroid as origin')

    LAUNCH_LATLON = (rlat, rlon)

    # Convert geofence to local XY coords centred on launch origin
    local = [ll2xy(la, lo, rlat, rlon) for la, lo in GEOFENCE_LATLON]
    fence = Polygon(local)
    if not fence.is_valid:
        fence = fence.buffer(0)

    spacing = 2 * ALTITUDE * math.tan(math.radians(HFOV / 2))
    print(f'\n  Geofence: {fence.area:.0f}m²')
    print(f'  Sweep spacing: {spacing:.1f}m')

    # Generate balanced Voronoi cells and lawnmower paths
    if LAUNCH_POSITIONS and len(LAUNCH_POSITIONS) == N_DRONES:
        seeds = []
        for idx in range(1, N_DRONES + 1):
            la, lo = LAUNCH_POSITIONS[idx]
            x, y = ll2xy(la, lo, rlat, rlon)
            pt = Point(x, y)
            if not fence.contains(pt):
                nearest = fence.exterior.interpolate(fence.exterior.project(pt))
                x, y = nearest.x, nearest.y
                print(f'  WARNING: D{idx} launch pad outside geofence — snapped to boundary')
            seeds.append((x, y))
        if len(set(seeds)) == 1:
            print('  WARNING: All launch positions are identical — falling back to geometric seeds')
            seeds = seed_points(N_DRONES, fence)
        else:
            print(f'  Seeds from launch positions: {seeds}')
    else:
        seeds = seed_points(N_DRONES, fence)
    cells, wts, dev, paths, times, infos = balance(
        seeds, fence, rlat, rlon, spacing, BUFFER_WIDTH
    )
    print(f'  Deviation: {dev*100:.1f}%')

    # No drone delay — simple altitude plan, no transit conflicts
    alt_plans = plan_altitudes(N_DRONES)
    print(f'\n  All drones: no transit conflicts (no delay) → {ALTITUDE:.0f}m')

    # Build missions
    missions = []
    for i in range(N_DRONES):
        try:
            m = build_mission(paths[i], alt_plans[i])
            if not m:
                raise ValueError(f"Mission for drone {i+1} is empty (no waypoints).")
            missions.append(m)
        except ValueError as e:
            print(f"  WARNING: {e}")
            missions.append([])

    # Print summary
    print(f'\n  {"─"*55}')
    for i in range(N_DRONES):
        info = infos[i]
        print(f'  D{i+1}: {info["area"]:.0f}m² | {info["n_wp"]} wps | '
              f'travel={info["t_travel"]:.1f}s  cover={info["t_coverage"]:.1f}s  '
              f'total={times[i]:.1f}s | alt={ALTITUDE:.0f}m')
    print(f'  {"─"*55}')
    tots = [t for t in times if t > 0]
    if tots:
        print(f'  Range: {min(tots):.1f}–{max(tots):.1f}s '
              f'(Δ{max(tots)-min(tots):.1f}s / '
              f'{(max(tots)-min(tots))/np.mean(tots)*100:.1f}%)')

    # Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_mission(cells, paths, infos, times, fence, rlat, rlon,
                 alt_plans, os.path.join(OUTPUT_DIR, 'mission_plan.png'))

    # print(f'\n  Exporting to {OUTPUT_DIR}/:')
    # exported = export_missions(missions, OUTPUT_DIR)
    # print(f'\n  Done. {len(exported)} files exported to {OUTPUT_DIR}/')

    csv_paths = export_missions_csv(missions, OUTPUT_DIR)
    for i, path in enumerate(csv_paths):
        try:
            send_file_over_socket(path, TARGET_IP[i], TARGET_PORT[i])
            print(f'Sent CSV mission for Drone {i+1} to {TARGET_IP[i]}:{TARGET_PORT[i]}')
        except Exception as e:
            print(f'  ERROR: Failed to send CSV mission for Drone {i+1}: {e}')

if __name__ == '__main__':
    main()
