"""NTN store-and-forward simulator: replays an RTCM3 corpus over UDP, one burst per pass.

    python generic_simulator_files.py --mode file --dir ./data --prefix Rover \
        --ip 10.121.251.18 --port 19001 --interval 1800

    python generic_simulator_files.py --mode batch --msgs 1800 --dir ./OldData \
        --prefix Rover --ip 10.121.251.18 --port 19001 --interval 1800

At the Sateliot facility passes are opened by hand, so the rover is sent one file per
pass instead of on a clock. If a pass fails, send the same index again on the next one:

    python generic_simulator_files.py --once --index 3 --dir ./data --prefix Rover \
        --ip 10.121.251.18 --port 19001

The facility link carries 200 bytes every 5 seconds (Sateliot, 2026-10-19), measured after
base64, so a whole 165 KB file does not fit in a 3-minute window. --facility keeps only the
station position and 8 GPS/Galileo epochs spread over the file (~5 KB, 34 datagrams, ~170 s),
which still gave 3/8 FIXED within 1.1 cm of the full solution on R_1:

    python generic_simulator_files.py --once --index 3 --facility --dir ./data --prefix Rover \
        --ip 10.121.251.18 --port 19001
"""
import base64
import socket
import time
import logging
import logging.handlers
import argparse
import os
import glob

# Log rotation. WHEN: 'S' seconds | 'M' minutes | 'H' hours | 'D' days | 'midnight'.
# Rotations beyond LOG_BACKUP_COUNT are deleted.
LOG_ROTATE_WHEN = "H"
LOG_ROTATE_INTERVAL = 12
LOG_BACKUP_COUNT = 4

DATAGRAM_BYTES = 1400
# Base64 grows each datagram by 4/3: 1024 raw bytes become 1368 characters, which with the
# 16-byte IMSI prefix nordic_pass adds stays under its 1472-byte default MO size.
DATAGRAM_BYTES_BASE64 = 1024
INTER_PACKET_SLEEP = 0.002

# --facility: 150 raw bytes encode to exactly 200 base64 characters, the per-message limit.
FACILITY_DATAGRAM_BYTES = 150
FACILITY_PACE = 5.0
FACILITY_EPOCHS = 8

# What the receiver's RTKLIB config uses (pos1-navsys=9): station position, GPS and Galileo MSM.
# Everything else in the corpus (GLONASS, BeiDou, antenna and receiver descriptors) is dead weight.
STATION_TYPES = {1005, 1006}
MSM_TYPES = set(range(1071, 1078)) | set(range(1091, 1098))


def setup_logging(prefix):
    """Log to logs/sim_<prefix>.log with time-based rotation, and to stderr."""
    os.makedirs("logs", exist_ok=True)

    handler = logging.handlers.TimedRotatingFileHandler(
        f"logs/sim_{prefix}.log",
        when=LOG_ROTATE_WHEN,
        interval=LOG_ROTATE_INTERVAL,
        backupCount=LOG_BACKUP_COUNT,
    )

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - [{prefix.upper()}] - %(message)s',
        handlers=[handler, logging.StreamHandler()]
    )


def corpus_files(data_dir, prefix):
    return sorted(glob.glob(os.path.join(data_dir, f"{prefix}_*.rtcm3")))


def iter_bursts(data_dir, prefix, mode, msgs, start_index=1):
    """Yield (label, payload) for each satellite pass, looping over the corpus forever.

    In 'file' mode one pass is one whole .rtcm3 file. In 'batch' mode one pass is the
    next `msgs` RTCM messages of the file, keeping the read position across passes,
    which models a real pass carrying whatever accumulated since the previous one and
    allows long continuous captures to be used without splitting them up.

    `start_index` (1-based) only applies to the first round over the corpus.

    Yields (None, None) when no files match, so the caller can wait and retry.
    """
    skip = start_index - 1
    while True:
        file_list = corpus_files(data_dir, prefix)

        if not file_list:
            logging.warning(f"No files found in '{data_dir}' matching {prefix}_*.rtcm3. Retrying in 5s...")
            yield None, None
            continue

        logging.info(f"Found {len(file_list)} files matching '{prefix}' (mode={mode}).")

        file_list, skip = file_list[skip:], 0
        for file_path in file_list:
            name = os.path.basename(file_path)

            if mode == "file":
                try:
                    with open(file_path, 'rb') as f:
                        data = f.read()
                except Exception as e:
                    logging.error(f"Error reading file {name}: {e}")
                    continue
                if data:
                    yield name, data
                continue

            # Imported here so that --mode file, all the facility needs, runs without pyrtcm
            from pyrtcm import RTCMReader
            try:
                with open(file_path, 'rb') as f:
                    reader = RTCMReader(f)
                    burst = 0
                    while True:
                        buffer = bytearray()
                        read = 0
                        eof = False

                        while read < msgs:
                            try:
                                raw, parsed = reader.read()
                            except Exception:
                                continue
                            if parsed is None:
                                eof = True
                                break
                            buffer.extend(raw)
                            read += 1

                        if read:
                            burst += 1
                            yield f"{name} [burst {burst}, {read} msgs]", bytes(buffer)

                        if eof:
                            logging.info(f"End of {name} after {burst} bursts.")
                            break
            except Exception as e:
                logging.error(f"Error reading file {name}: {e}")
                continue

        logging.info("End of file list reached. Restarting sequence.")


def slim_payload(payload, epochs):
    """Keep the first station-position frame and `epochs` GPS/Galileo MSM epochs spread evenly.

    Epochs are spread rather than consecutive because ambiguity resolution needs satellite geometry
    to change: on R_1, 8 consecutive epochs gave 0 FIXED and 8 spread over 3 minutes gave 3.
    """
    station = None
    by_epoch = {}
    i = 0
    while i + 6 <= len(payload):
        if payload[i] != 0xD3:
            i += 1
            continue
        length = ((payload[i + 1] & 0x03) << 8) | payload[i + 2]
        frame = payload[i:i + length + 6]
        i += length + 6
        if len(frame) < 11:
            continue
        msg_type = (frame[3] << 4) | (frame[4] >> 4)
        if msg_type in STATION_TYPES and station is None:
            station = frame
        elif msg_type in MSM_TYPES:
            # Same field rtcm_tow_span() reads in agrarian_app.py: epoch time after type and station id
            tow = (int.from_bytes(frame[3:11], "big") >> 10) & 0x3FFFFFFF
            by_epoch.setdefault(tow, []).append(frame)

    all_epochs = list(by_epoch.values())
    step = max(1, len(all_epochs) // epochs)
    kept = all_epochs[::step][:epochs]
    return (station or b"") + b"".join(b"".join(frames) for frames in kept), len(kept), len(all_epochs)


def send_burst(sock, payload, server_ip, server_port, datagram_bytes, encoding, pace=INTER_PACKET_SLEEP):
    """Fragment one burst into UDP datagrams; returns how many were sent."""
    if encoding == "base64" and payload:
        # Same number of datagrams, but evenly sized: the receiver's auto-detection cannot
        # tell a base64 datagram of a few characters from raw bytes, so no tiny tail.
        count = -(-len(payload) // datagram_bytes)
        datagram_bytes = -(-len(payload) // count)
    udp_packets = 0
    for i in range(0, len(payload), datagram_bytes):
        chunk = payload[i:i + datagram_bytes]
        if encoding == "base64":
            chunk = base64.b64encode(chunk)
        try:
            sock.sendto(chunk, (server_ip, server_port))
            udp_packets += 1
            time.sleep(pace)
        except Exception as e:
            logging.error(f"UDP send error: {e}")
    return udp_packets


def send_once(data_dir, prefix, server_ip, server_port, index, datagram_bytes, encoding,
              pace=INTER_PACKET_SLEEP, epochs=None):
    """Send corpus file number `index` (1-based) once and return, for passes opened by hand."""
    file_list = corpus_files(data_dir, prefix)
    if not 1 <= index <= len(file_list):
        logging.error(f"--index {index} out of range: {len(file_list)} files match {prefix}_*.rtcm3 in '{data_dir}'")
        return 1

    file_path = file_list[index - 1]
    with open(file_path, 'rb') as f:
        payload = f.read()

    if epochs:
        original = len(payload)
        payload, kept, total = slim_payload(payload, epochs)
        logging.info(f"Slimmed {original} -> {len(payload)} bytes: station position + {kept}/{total} "
                     f"GPS/Galileo epochs")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    logging.info(f"Sending {os.path.basename(file_path)} (index {index}/{len(file_list)}, {len(payload)} bytes, "
                 f"{datagram_bytes}-byte datagrams, encoding={encoding}) -> {server_ip}:{server_port}")
    expected = -(-len(payload) // datagram_bytes)
    if pace >= 1:
        logging.info(f"{expected} datagrams, one every {pace:g}s: about {expected * pace:.0f}s")
    udp_packets = send_burst(sock, payload, server_ip, server_port, datagram_bytes, encoding, pace)
    logging.info(f"Transfer complete: {udp_packets} packets sent. "
                 f"Next pass: --index {index + 1} if this one went through, --index {index} again if not.")
    return 0


def run_simulator(data_dir, prefix, server_ip, server_port, interval, mode, msgs,
                  start_index, datagram_bytes, encoding):
    """Send one burst per tick, fragmented into UDP datagrams.

    The transmission clock is absolute, so a slow burst does not push the following
    passes back; falling behind is reported as a lag warning instead.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    logging.info(f"Starting forwarder '{prefix}' (mode={mode}) -> Dest: {server_ip}:{server_port}")
    logging.info(f"Interval set to {interval} seconds." + (f" Burst size: {msgs} RTCM msgs." if mode == "batch" else ""))

    next_tx_time = time.time() + interval

    for label, payload in iter_bursts(data_dir, prefix, mode, msgs, start_index):
        if payload is None:
            time.sleep(5)
            next_tx_time = time.time() + interval
            continue

        sleep_duration = next_tx_time - time.time()
        if sleep_duration > 0:
            logging.info(f"Waiting {sleep_duration:.2f}s...")
            time.sleep(sleep_duration)
        else:
            logging.warning(f"Lag detected! Sending immediately (behind by {abs(sleep_duration):.2f}s)")

        next_tx_time += interval

        logging.info(f"Sending: {label} ({len(payload)} bytes)")
        udp_packets = send_burst(sock, payload, server_ip, server_port, datagram_bytes, encoding)
        logging.info(f"Transfer complete: {udp_packets} packets sent.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NTN Simulator (Store & Forward)")
    parser.add_argument('--dir', type=str, required=True, help="Directory containing .rtcm3 files")
    parser.add_argument('--prefix', type=str, required=True, help="File prefix (e.g., 'Base', 'Rover')")
    parser.add_argument('--ip', type=str, required=True, help="Destination IP")
    parser.add_argument('--port', type=int, required=True, help="Destination UDP port")
    parser.add_argument('--interval', type=int, default=60, help="Wait time in seconds")
    parser.add_argument('--mode', choices=('file', 'batch'), default='file',
                        help="'file': one whole .rtcm3 per pass. 'batch': --msgs RTCM messages per pass")
    parser.add_argument('--msgs', type=int, default=1800,
                        help="RTCM messages per pass (only used with --mode batch)")
    parser.add_argument('--once', action='store_true',
                        help="Send the file given by --index and exit, for passes opened by hand (--mode file only)")
    parser.add_argument('--index', type=int, default=1,
                        help="1-based position of the file in the sorted corpus: the file sent with --once, "
                             "or the first one of the loop without it")
    parser.add_argument('--datagram-bytes', type=int, default=None,
                        help=f"Payload bytes per UDP datagram, before encoding "
                             f"(default {DATAGRAM_BYTES}, or {DATAGRAM_BYTES_BASE64} with --encoding base64)")
    parser.add_argument('--encoding', choices=('raw', 'base64'), default='raw',
                        help="base64 sends ASCII-only datagrams; the receiver detects them by itself "
                             "(ROVER_PAYLOAD_ENCODING=auto)")
    parser.add_argument('--pace', type=float, default=None,
                        help=f"Seconds between datagrams (default {INTER_PACKET_SLEEP})")
    parser.add_argument('--epochs', type=int, default=None,
                        help="Send only the station position and this many GPS/Galileo epochs, spread "
                             "evenly over the file (--once only)")
    parser.add_argument('--facility', action='store_true',
                        help=f"Sateliot link budget: --encoding base64 --datagram-bytes {FACILITY_DATAGRAM_BYTES} "
                             f"--pace {FACILITY_PACE:g} --epochs {FACILITY_EPOCHS}, each overridable")

    args = parser.parse_args()
    if args.once and args.mode != 'file':
        parser.error("--once only works with --mode file")
    if args.facility:
        args.encoding = 'base64'
        args.datagram_bytes = args.datagram_bytes or FACILITY_DATAGRAM_BYTES
        args.pace = FACILITY_PACE if args.pace is None else args.pace
        args.epochs = args.epochs or FACILITY_EPOCHS
    if args.pace is None:
        args.pace = INTER_PACKET_SLEEP
    if args.epochs and not args.once:
        parser.error("--epochs (and --facility) only work with --once")
    if args.index < 1:
        parser.error("--index starts at 1")
    datagram_bytes = args.datagram_bytes or (DATAGRAM_BYTES_BASE64 if args.encoding == 'base64' else DATAGRAM_BYTES)

    setup_logging(args.prefix)

    if args.once:
        raise SystemExit(send_once(args.dir, args.prefix, args.ip, args.port, args.index,
                                   datagram_bytes, args.encoding, args.pace, args.epochs))

    run_simulator(args.dir, args.prefix, args.ip, args.port, args.interval, args.mode, args.msgs,
                  args.index, datagram_bytes, args.encoding)
