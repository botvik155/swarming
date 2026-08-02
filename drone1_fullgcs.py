#!/usr/bin/env python3

import math
import sys
import time
import select
from pymavlink import mavutil
import socket
import csv
import numpy as np

# ── CONFIG ───────────────────────────────────────────────────────────────────
LISTEN_PORT    = 9000          # must match TARGET_PORT[i] on GCS for this drone
CSV_SAVE_PATH  = '/home/yuvraj/swarming_guided/drone_1_mission.csv'
CONNECTION     = 'udpin:0.0.0.0:14555'                       
TAKEOFF_ALT    = 6.0    # metres AGL
WP_RADIUS      = 0.4    # metres — waypoint is "reached" within this distance
ALT_BAND       = 0.5    # metres — takeoff done when alt ≥ (TAKEOFF_ALT - ALT_BAND)
DRONE_ID       = 'D1'
HEADING_TOLERANCE = 10
P_YAW = 0.75
P_VEL = 1
# ─────────────────────────────────────────────────────────────────────────────

# ArduCopter custom_mode values
GUIDED_MODE = 4   # GUIDED
RTL_MODE    = 6   # RTL  (for reference)

# SET_POSITION_TARGET_GLOBAL_INT type_mask: ignore velocity / accel / yaw
_POS_ONLY_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE    |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE    |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE    |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE    |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE    |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE    |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE   |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f'[{DRONE_ID}] {msg}', flush=True)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Horizontal distance in metres between two GPS coordinates."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def bearing_error(lat1: float, lon1: float, lat2: float, lon2: float, heading: float) -> float:
    """
    Returns the difference between the current heading and the bearing to the target.
    Positive = target is to the right, Negative = target is to the left.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    bearing_to_target = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    error = (bearing_to_target - heading + 540.0) % 360.0 - 180.0
    return error


def receive_and_save_csv(listen_port: int, save_path: str) -> None:
    """
    Blocks until a CSV file is received from the GCS over TCP.
    Saves it to save_path and sends back an ACK byte.
    """
    log(f'Waiting for mission CSV on port {listen_port} ...')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', listen_port))
        s.listen(1)
        conn, addr = s.accept()
        with conn:
            log(f'GCS connected from {addr}')

            # Read filename
            name_len = int.from_bytes(conn.recv(4), 'big')
            filename = conn.recv(name_len).decode()
            print(f'Received file: {filename}')

            # Read file contents
            file_size = int.from_bytes(conn.recv(8), 'big')
            received = b''
            while len(received) < file_size:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                received += chunk

            with open(save_path, 'wb') as f:
                f.write(received)

            # Send ACK
            conn.sendall(b'\x01')

    log(f'CSV received and saved → {save_path}')


# def parse_waypoints(path: str) -> list:
#     """
#     Parse a QGC WPL 110 waypoints file.
#     Returns [(lat, lon, alt), …] — home, takeoff, and RTL entries are skipped.
#     """
#     waypoints = []
#     with open(path) as f:
#         lines = f.readlines()

#     if not lines or not lines[0].startswith('QGC WPL'):
#         raise ValueError(f'Not a valid QGC WPL file: {path}')

#     for line in lines[1:]:                  # skip 'QGC WPL 110' header
#         parts = line.strip().split('\t')
#         if len(parts) < 11:
#             continue
#         current = int(parts[1])             # 1 = home row
#         cmd     = int(parts[3])             # MAV_CMD
#         if current == 1:                    # home — skip
#             continue
#         if cmd == 22:                       # NAV_TAKEOFF — skip (we do it ourselves)
#             continue
#         if cmd == 20:                       # NAV_RETURN_TO_LAUNCH — skip
#             continue
#         lat = float(parts[8])
#         lon = float(parts[9])
#         alt = float(parts[10])
#         waypoints.append((lat, lon, alt))

#     return waypoints

def parse_csv_waypoints(path: str) -> list:
    waypoints = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row['wp_type'].strip() != 'COVERAGE':
                continue
            lat = float(row['lat'])
            lon = float(row['lon'])
            alt = float(row['alt_m'])
            waypoints.append((lat, lon, alt))
    return waypoints


def wait_for_confirmation(timeout_seconds: int = 600) -> bool:

    minutes = timeout_seconds // 60
    log(f" SAFETY PAUSE: Type 'yes' and press Enter to start mission.")
    
    # select.select waits for keyboard input (sys.stdin) until the timeout hits
    ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    
    if ready:
        # The user typed something and pressed Enter
        response = sys.stdin.readline().strip().lower()
        if response in ['yes', 'y']:
            log("✓ Mission confirmed. Proceeding...")
            return True
        else:
            log(f"✗ User typed '{response}'. Aborting...")
            return False
    else:
        # The timer ran out
        log(" 10-minute timeout reached with no input. Aborting...")
        return False
# ── MAIN STEPS ────────────────────────────────────────────────────────────────

def connect() -> mavutil.mavlink_connection:
    log(f'Connecting → {CONNECTION}')
    master = mavutil.mavlink_connection(CONNECTION)
    master.wait_heartbeat()
    log(f'Heartbeat OK  system={master.target_system}  component={master.target_component}')
    return master


def set_mode_guided(master) -> None:
    """Command the drone to switch to GUIDED mode autonomously."""
    log('Switching to GUIDED mode … (waiting for FC to accept)')
    last_cmd_time = 0
    while True:
        now = time.time()
        # Send command every 2 seconds until confirmed
        if now - last_cmd_time > 2.0:
            master.mav.command_long_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                GUIDED_MODE,
                0, 0, 0, 0, 0
            )
            last_cmd_time = now

        hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if hb is None:
            continue

        if hb.custom_mode == GUIDED_MODE:
            log('✓ GUIDED mode confirmed')
            return


def arm_drone(master) -> None:
    """Command the drone to arm its motors autonomously."""
    log('Arming drone … (waiting for GPS lock and Pre-Arm checks)')
    last_cmd_time = 0
    while True:
        now = time.time()
        # Send command every 2 seconds until confirmed
        if now - last_cmd_time > 2.0:
            master.mav.command_long_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1,  # 1 to ARM
                0, 0, 0, 0, 0, 0
            )
            last_cmd_time = now

        hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if hb is None:
            continue

        if bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            log('✓ Drone ARMED successfully')
            return


def send_takeoff_and_wait(master, target_alt: float) -> None:
    log(f'Initiating takeoff loop to {target_alt} m …')

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,            # confirmation
        0,            # param1: min pitch (deg)
        0, 0, 0,      # param2-4: unused
        0, 0,         # param5-6: lat/lon  (0 = use current position)
        target_alt    # param7: altitude (m AGL)
    )


    while True:

        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
        if msg is None:
            continue
            
        rel_alt = msg.relative_alt / 1000.0    # mm → m
        print(f'\r  [{DRONE_ID}]  altitude = {rel_alt:.1f} m  /  {target_alt} m   ',
              end='', flush=True)
              
        if rel_alt >= target_alt - ALT_BAND:
            print()
            log(f'Takeoff complete  ({rel_alt:.1f} m)')
            return




# def goto_waypoint(master, lat: float, lon: float, alt: float) -> None:
#     """Send drone to target and block until within horizontal WP_RADIUS."""
#     master.mav.set_position_target_global_int_send(
#         0,                                                
#         master.target_system,
#         master.target_component,
#         mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
#         _POS_ONLY_MASK,
#         int(lat * 1e7),   
#         int(lon * 1e7),   
#         alt,              
#         0, 0, 0,          
#         0, 0, 0,          
#         0, 0              
#     )
#     while True:
#         msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5)
#         if msg is None:
#             continue
            
#         cur_lat = msg.lat / 1e7
#         cur_lon = msg.lon / 1e7
        
#         # Calculate horizontal distance only
#         dist_2d = haversine_m(cur_lat, cur_lon, lat, lon)
        
#         print(f'\r  [{DRONE_ID}]  target=({lat:.6f}, {lon:.6f})  dist={dist_2d:.1f} m   ',
#               end='', flush=True)
              
#         # Drone only checks if it is within the 2D radius
#         if dist_2d <= WP_RADIUS:
#             print()
#             return


def goto_waypoint(master, lat: float, lon: float, alt: float) -> None:
    lat_int = int(lat * 1e7)
    lon_int = int(lon * 1e7)

    POSITION_ONLY_MASK = 0x0DF8  # ignore velocity, acceleration, yaw, yaw_rate

    master.mav.set_position_target_global_int_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        POSITION_ONLY_MASK,
        lat_int, lon_int, alt,
        0, 0, 0,
        0, 0, 0,
        0, 0,
    )

    while True:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
        if msg is None:
            continue

        cur_lat = msg.lat / 1e7
        cur_lon = msg.lon / 1e7
        d_2d = haversine_m(cur_lat, cur_lon, lat, lon)

        print(f'\r  [{DRONE_ID}]  target=({lat:.6f}, {lon:.6f})  dist={d_2d:.1f} m   ', end='', flush=True)

        if d_2d <= WP_RADIUS:
            print()
            return

        



def send_rtl(master) -> None:
    log('Sending RTL …')
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
        0, 0, 0, 0, 0, 0, 0, 0
    )


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    master = connect()

    # ── Receive mission CSV from GCS before waiting for confirmation ──────────
    receive_and_save_csv(LISTEN_PORT, CSV_SAVE_PATH)

    if not wait_for_confirmation(600):
        log("Exiting program.")
        sys.exit(0)

    # ── Load waypoints from received CSV ──────────────────────────────────────
    try:
        waypoints = parse_csv_waypoints(CSV_SAVE_PATH)
    except FileNotFoundError:
        log(f'ERROR: CSV not found: {CSV_SAVE_PATH}')
        sys.exit(1)

    if not waypoints:
        log('ERROR: no COVERAGE waypoints found in CSV — aborting')
        sys.exit(1)

    log(f'Loaded {len(waypoints)} waypoints from {CSV_SAVE_PATH}')

    # ── Rest of mission unchanged ─────────────────────────────────────────────
    set_mode_guided(master)
    arm_drone(master)
    send_takeoff_and_wait(master, TAKEOFF_ALT)
    time.sleep(2)

    for i, (lat, lon, alt) in enumerate(waypoints):
        log(f'→ WP {i + 1}/{len(waypoints)}   ({lat:.7f}, {lon:.7f}, {alt:.1f} m)')
        goto_waypoint(master, lat, lon, alt)
        log(f'✓ WP {i + 1} reached')
        time.sleep(1)

    send_rtl(master)
    log('Mission complete — returning home')


if __name__ == '__main__':
    main()