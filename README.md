# swarming

**Multi-drone area coverage: partition a survey area, plan a lawnmower route per
drone, and fly the fleet over MAVLink.**

The planner splits a geofence into cells whose *mission times* — not areas — are
balanced, so every drone finishes together. Missions are shipped to per-drone
agents over TCP, and each agent flies its own route in GUIDED mode with a
yaw-then-translate controller.

```
oldtest_uas.txt (geofence)          drone launch GPS
        │                                 │  UDP 14553+
        └────► main_height_diff_guided.py ◄┘
                         │
        balanced power diagram → lawnmower paths
                         │
          drone_N_mission.csv  +  mission_plan.png
                         │  TCP
                         ▼
              droneN_fullgcs.py ──MAVLink──► ArduCopter
```

| File | Role |
|---|---|
| `main_height_diff_guided.py` | **Planner / GCS** — partitions the area, plans routes, exports and dispatches missions |
| `drone1_fullgcs.py` | **Per-drone agent** — receives its CSV, arms, takes off, flies waypoints, RTLs |
| `oldtest_uas.txt` | Survey geofence, QGC WPL 110 format |

---

## How the area is divided

A plain Voronoi split gives equal *areas*, which is the wrong objective — a long
thin cell costs far more flight time than a compact cell of the same area,
because of turns and yaw settling. This planner optimises the thing that
actually matters.

**1. Power diagram.** Cells come from a *weighted* Voronoi (power) diagram: each
site carries a weight that shifts its bisectors, so a cell can grow or shrink
without moving the site. `clip_half()` builds each cell by successive half-plane
intersections against every other site.

**2. Realistic time model.** For each candidate cell, `generate_lawnmower()`
lays a boustrophedon path along the cell's **longest edge** — fewer, longer
sweeps means fewer turns — at spacing `2 · ALTITUDE · tan(HFOV/2)`, inset by
`BUFFER_WIDTH`. `analyze_mission()` then costs that path properly:

```
t_cover = straight/CRUISE_SPEED + turns/(CRUISE_SPEED·0.6)
        + Σ|Δheading|/YAW_RATE + n_wp·SETTLING_TIME
```

Turn segments are charged at 60 % of cruise speed, every heading change costs
yaw time, and each waypoint costs 1.5 s of settling.

**3. Iterative balancing.** `balance()` runs up to 80 iterations, pushing weight
from the slow drones towards the fast ones with a decaying step
(`0.95^iteration`), and keeps the best partition it finds rather than whatever
the last iteration produced. It stops at ≤3 % deviation. A degenerate cell
damps the weights and retries instead of aborting.

The balanced quantity is the **full** mission time:

```
total = launch_delay + transit_to_cell + coverage
```

With `BALANCE_ON_LANDING_TIME = True`, the pad wait is included, so later drones
get proportionally less area and the whole fleet **lands at the same moment**.
Set it `False` to balance flight time alone; the fleet then finishes staggered
by `LAUNCH_DELAY`.

**Seeding.** Seeds are the drones' **real launch pads**, read live from their
telemetry streams, so each drone gets the cell it is already standing in and
transit-to-start is near zero. Two guards:

- a pad outside the geofence is snapped to the nearest boundary point;
- pads closer together than `0.25·√(area/N)` make the power diagram degenerate
  — bisector directions become noise and cells collapse, leaving part of the
  geofence uncovered. The planner falls back to well-spread geometric seeds,
  greedily assigned so each drone still keeps the cell nearest its pad.

**Vertical profile.** Every drone takes off and cruises at `TRANSIT_ALT` (15 m)
at `TRANSIT_SPEED` (8 m/s), then descends to the common survey altitude
`ALTITUDE` (6 m) over its first waypoint. Launches are staggered by
`LAUNCH_DELAY` (10 s) — drone *i* holds on the pad for `i · LAUNCH_DELAY`
seconds, which is what keeps the fleet separated during the climb-out.

---

## Mission dispatch

Missions are exported as tab-separated CSV:

```
drone_id  wp_index  lat          lon          alt_m  wp_type   delay_s
1         0         -35.3632621  149.1652374  0      HOME      0.0
1         1         -35.3632621  149.1652374  15.00  TAKEOFF   0.0
1         2         -35.3629975  149.1650748  15.00  TRANSIT   0.0
1         3         -35.3629975  149.1650748  6.00   COVERAGE  0.0
…
1         27                                         RTL       0.0
```

The first waypoint appears twice: once as `TRANSIT` at cruise altitude, then as
`COVERAGE` at survey altitude — that pair *is* the descent. `delay_s` on the
`TAKEOFF` row is how long the drone holds on the pad before lifting off.

`send_file_over_socket()` pushes each file over TCP with length-prefixed framing
(4-byte name length, name, 8-byte payload length, payload) and waits for a
`0x01` acknowledgement byte, so a mission that fails to arrive is reported
rather than silently dropped. A missing IP/port entry is logged, not fatal.

Set `EXPORT_WAYPOINTS = True` to also write QGC `.waypoints` files alongside the
CSVs, and `DEBUG = True` for `[debug]` lines tracing the balancing iterations.

## The per-drone agent

`drone1_fullgcs.py` is the drone-side program (one per aircraft, differing only
in ID, ports, and CSV path). Its sequence:

1. Connect to the flight controller over MAVLink.
2. Listen on `LISTEN_PORT`, receive and save its mission CSV.
3. **Wait for operator confirmation on stdin** — 600 s timeout.
4. Set GUIDED → arm → take off to `TAKEOFF_ALT`, holding until within `ALT_BAND`.
5. Fly each waypoint in turn, arriving within `WP_RADIUS` (0.4 m).
6. RTL.

`goto_waypoint()` yaws before it translates: when the bearing error to the next
waypoint exceeds `HEADING_TOLERANCE` (10°), the drone rotates on the spot under
proportional control (`P_YAW = 0.75`) and only then commands forward velocity
(`P_VEL = 1`). This keeps a fixed forward camera pointed along the track and
avoids the sideways drift a direct position target produces.

> **Nothing arms without a human.** Step 3 is a blocking prompt — agents can be
> started, given their missions, and left waiting until the operator is ready.

### Known gap between planner and agent

The agent's `parse_csv_waypoints()` keeps only rows with `wp_type == 'COVERAGE'`
and does not read `delay_s`. So with the current pair:

- the `TRANSIT` row is dropped — the drone climbs to its own `TAKEOFF_ALT`
  rather than flying the planned high-altitude transit;
- the staggered launch is **not** executed on the drone, even though the planner
  balanced the partition assuming it would be.

Closing this means handling `TRANSIT` and honouring `delay_s` on the agent side.
Until then, run with `LAUNCH_DELAY = 0` so the plan matches what actually flies.

---

## Run

```bash
pip install pymavlink numpy matplotlib shapely
```

**1. Start the drones** (SITL or real), each streaming telemetry to its planner
port in `DRONE_UDP_PORTS` and to its agent's `CONNECTION` endpoint.

**2. Start each agent** — one terminal per drone:

```bash
python3 drone1_fullgcs.py
```

It connects, then blocks waiting for its mission.

**3. Run the planner:**

```bash
python3 main_height_diff_guided.py
```

It reads every drone's GPS, balances the partition, writes
`drone_N_mission.csv` and `mission_plan.png`, and dispatches each mission.

**4. Confirm at each agent prompt** to arm and fly.

### Output

Per drone: cell area, waypoint count, pad wait / transit / coverage / total
time, and the altitude profile. Then the fleet time spread — the number to
watch, since it shows how well the partition balanced.

`mission_plan.png` renders the geofence, each drone's cell and route, and the
per-drone timings.

### Parameters

All in the config block at the top of `main_height_diff_guided.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `N_DRONES` | `1` | Fleet size — `TARGET_IP` and `TARGET_PORT` need one entry each |
| `ALTITUDE` | `6` m | Survey altitude AGL |
| `TRANSIT_ALT` | `15` m | Climb-out / transit altitude AGL |
| `ALT_SEPARATION` | `5` m | Vertical spacing between drones (reported in the plot) |
| `LAUNCH_DELAY` | `10` s | Pad hold: drone *i* lifts off at `i · LAUNCH_DELAY` |
| `HFOV` | `87°` | Camera horizontal FOV — sets sweep spacing |
| `BUFFER_WIDTH` | `2.0` m | Keep-out inset from the cell boundary |
| `CRUISE_SPEED` | `2.0` m/s | Survey speed |
| `TRANSIT_SPEED` | `8.0` m/s | Launch → survey-area speed |
| `YAW_RATE` | `45°/s` | Time-model input |
| `SETTLING_TIME` | `1.5` s | Per-waypoint cost in the time model |
| `BALANCE_ON_LANDING_TIME` | `True` | Include pad wait, so the fleet lands together |
| `GEOFENCE_FILE` | `oldtest_uas.txt` | Survey area |
| `OUTPUT_DIR` | absolute path | **Change for your machine** |
| `DRONE_UDP_PORTS` | `{1: 14553}` | Telemetry port per drone |

`OUTPUT_DIR`, and `CSV_SAVE_PATH` in the agent, are absolute paths pointing at a
development machine — update both before running elsewhere.

### Ports

| Purpose | Default |
|---|---|
| Planner reads drone GPS (UDP) | 14553, 14554, … |
| Agent ↔ flight controller (UDP) | `CONNECTION` in the agent |
| Planner → agent, mission CSV (TCP) | `TARGET_PORT` / `LISTEN_PORT` |

Planner and agent must not share a MAVLink UDP port — two processes on one
socket steal each other's packets. Fan the link out with MAVProxy or
mavlink-router if needed.
