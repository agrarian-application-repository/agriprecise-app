"""Turn the drone experiment captures into passes the simulator can replay.

It imports the receiver, so it needs its dependencies (psycopg2, requests).

The drone dataset (rover on a DJI Mavic 3, base at Montells, u-blox F20P, 2026-10-16) needs
three things before the receiver can use it:

- The base stream has no station position message. The receiver takes the base position
  from the RTCM 1005/1006 the base sends (ant2-postype=rinexhead), so a 1006 built from the
  surveyed position is inserted every 10 s, as a real base would send it.
- There are no ephemerides in RTCM, only in the UBX logs. They are converted beforehand with
  RTKLIB's str2str (see README) and deduplicated here.
- Each flight is cut into passes of at most ROVER_BUFFER_EPOCHS epochs, each with the base
  epochs of the same span and the ephemerides, written as B_/R_/N_<nn>.rtcm3.

    python tools/prepare_drone_test.py --base montells.rtcm3 --nav montells_nav.rtcm3 \\
        --rover drone_run_1.rtcm3 drone_run_2.rtcm3 \\
        --llh 41.6473109249999 2.349313816 263.604 --out data/drone
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agrarian_app import crc24q, msm_tow, rtcm_frames  # noqa: E402

STATION_POSITION_EVERY_MS = 10_000


def llh_to_ecef(lat_deg, lon_deg, h):
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    return ((n + h) * math.cos(lat) * math.cos(lon),
            (n + h) * math.cos(lat) * math.sin(lon),
            (n * (1 - e2) + h) * math.sin(lat))


def rtcm_1006(station_id, x, y, z):
    """RTCM 1006 frame: antenna reference point in ECEF, antenna height 0."""
    fields = [(1006, 12), (station_id, 12), (0, 6), (1, 1), (0, 1), (1, 1), (0, 1),
              (round(x * 10000), 38), (0, 1), (0, 1), (round(y * 10000), 38), (0, 2),
              (round(z * 10000), 38), (0, 16)]
    bits = nbits = 0
    for value, width in fields:
        bits = (bits << width) | (value & ((1 << width) - 1))
        nbits += width
    payload = bits.to_bytes(nbits // 8, "big")
    frame = b"\xd3" + len(payload).to_bytes(2, "big") + payload
    return frame + crc24q(frame).to_bytes(3, "big")


def epochs(data):
    """[(tow_ms, bytes)] in file order; other messages go with the epoch that follows them."""
    result, carry = {}, b""
    for start, end in rtcm_frames(data):
        frame = data[start:end]
        tow = msm_tow(frame)
        if tow is None:
            carry += frame
        else:
            result[tow] = result.get(tow, b"") + carry + frame
            carry = b""
    return sorted(result.items())


def runs(rover_epochs):
    """Split into runs of epochs 1 s apart (one per flight)."""
    out = []
    for tow, frames in rover_epochs:
        if not out or abs(tow - out[-1][-1][0] - 1000) > 1:
            out.append([])
        out[-1].append((tow, frames))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--nav", required=True, type=Path, help="Ephemerides in RTCM3 (str2str output)")
    parser.add_argument("--rover", required=True, type=Path, nargs="+")
    parser.add_argument("--llh", required=True, type=float, nargs=3, metavar=("LAT", "LON", "H"),
                        help="Base antenna position: degrees, degrees, ellipsoidal metres")
    parser.add_argument("--max-epochs", type=int, default=360, help="ROVER_BUFFER_EPOCHS of the receiver")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    base = epochs(args.base.read_bytes())
    first_msm = next(f for _, frames in base for s, e in rtcm_frames(frames)
                     if msm_tow(f := frames[s:e]) is not None)
    station_id = ((first_msm[4] & 0x0F) << 8) | first_msm[5]
    position = rtcm_1006(station_id, *llh_to_ecef(*args.llh))
    base = [(tow, (position if tow % STATION_POSITION_EVERY_MS == 0 else b"") + frames)
            for tow, frames in base]

    nav_data = args.nav.read_bytes()
    nav = b"".join(dict.fromkeys(nav_data[s:e] for s, e in rtcm_frames(nav_data)))

    rover = []
    for path in args.rover:
        rover += epochs(path.read_bytes())
    passes = []
    for run in runs(sorted(rover)):
        n = math.ceil(len(run) / args.max_epochs)
        size = math.ceil(len(run) / n)
        passes += [run[i:i + size] for i in range(0, len(run), size)]

    args.out.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(passes, 1):
        first, last = chunk[0][0], chunk[-1][0]
        base_chunk = [frames for tow, frames in base if first - 1 <= tow <= last + 1]
        (args.out / f"R_{i:02d}.rtcm3").write_bytes(b"".join(frames for _, frames in chunk))
        (args.out / f"B_{i:02d}.rtcm3").write_bytes(b"".join(base_chunk))
        (args.out / f"N_{i:02d}.rtcm3").write_bytes(nav)
        print(f"pass {i:02d}: rover {len(chunk)} epochs, TOW {first / 1000:.0f}-{last / 1000:.0f} s, "
              f"base {len(base_chunk)} epochs")
    print(f"station id {station_id}, ephemerides {len(nav)} bytes per pass")


if __name__ == "__main__":
    main()
