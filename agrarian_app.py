"""AGRI-PRECISE receiver: RTCM3 over UDP in, RTK positions with anomaly labels out.

Three streams arrive on their own UDP ports (base observations, rover observations and
navigation messages) and are buffered to disk. A window is processed when the rover buffer
holds ROVER_BUFFER_EPOCHS epochs, or when no data has arrived for IDLE_TIMEOUT seconds (the
end of a satellite pass). Each window is converted with convbin, solved with rnx2rtkp
(RTK when base data is present, single point positioning otherwise), checked for anomalies
on every run of 120 consecutive 1 Hz rover epochs, and inserted into PostgreSQL.
"""
import base64
import binascii
import threading
import json
import logging
import logging.handlers
import subprocess
import time
from datetime import datetime
from pathlib import Path
import os
import re
from queue import Queue
import socket
import psycopg2
import requests

DEBUG = int(os.getenv("DEBUG", "0"))

# Log rotation. WHEN: 'S' seconds | 'M' minutes | 'H' hours | 'D' days | 'midnight'.
# Rotations beyond LOG_BACKUP_COUNT are deleted.
LOG_ROTATE_WHEN = "H"
LOG_ROTATE_INTERVAL = 12
LOG_BACKUP_COUNT = 4

# RTKLIB binaries: bundled .exe on Windows, on PATH inside the container.
if os.name == 'nt':
    CONVBIN_PATH = "./RTKLIB_EX_2.5.0/convbin.exe"
    RNX2RTKP_PATH = "./RTKLIB_EX_2.5.0/rnx2rtkp.exe"
else:
    CONVBIN_PATH = "convbin"
    RNX2RTKP_PATH = "rnx2rtkp"
# Without -k, rnx2rtkp applies its own defaults (GPS only, L1+L2, continuous
# ambiguity resolution) and yields 0% FIXED. A missing file is not an error either:
# it silently falls back to those defaults.
RTKLIB_CONFIG_FILE = os.getenv("RTKLIB_CONFIG_FILE", "./config/config_rtkpost_ntn.conf")

# Start date and time of a replayed capture, "YYYY/MM/DD hh:mm:ss". Leave empty with live
# data, where the system clock is already correct. See convbin_cmd().
RTCM_START_TIME = os.getenv("RTCM_START_TIME", "").strip()

# Where buffers, intermediates and pending .pos files live.
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))

# Upper bound on the rover epochs of a window, for a stream that never pauses; a pass is
# processed when it ends (IDLE_TIMEOUT) whatever its size. Not smaller: RTK ambiguities are
# resolved anew in every window, so short windows spend a larger share converging.
ROVER_BUFFER_EPOCHS = int(os.getenv("ROVER_BUFFER_EPOCHS", "360"))
# Seconds without data on any stream after which the window is processed with what it has.
# Over a store-and-forward link one message arrives every few seconds, so this must be well
# above that gap or a pass is split into fragments too small to solve.
IDLE_TIMEOUT = float(os.getenv("IDLE_TIMEOUT", "60"))
# Safety cap per stream buffer: a base or nav stream arriving without rover would otherwise
# grow without limit on a disk-limited computer.
MAX_BUFFER_BYTES = int(os.getenv("MAX_BUFFER_BYTES", str(20 * 1024 * 1024)))

# Consecutive 1 Hz rover epochs per anomaly window. Must match epochs_per_chunk in the
# anomaly service's parse_rtcm_file(), which cuts the file into non-overlapping windows of
# this size, in order, and drops any tail.
ANOMALY_WINDOW_EPOCHS = 120
ANOMALY_EPOCH_STEP_MS = 1000
AI_PREDICTOR_URL = os.getenv("AI_PREDICTOR_URL", "http://ai_predictor:8000/anomaly_detection")
# The anomaly service only opens files under its own data directory and takes the path
# relative to it, so it mounts OUTPUT_DIR there under this name (see docker-compose.yml).
ANOMALY_SHARED_DIR = "rover_windows"
# Per window: in hybrid_engine the service tries csa (killed after 30 s) and then falls back
# to ONNX on the CPU; generous, since a slow answer is only a late label.
ANOMALY_TIMEOUT_S = 300

ROVER_ID = os.getenv("ROVER_ID", "ROVER_01")
# 0: no database at all (a test without the AGRARIAN network): what would be inserted and
# updated is written to the log instead, and the files are deleted as if uploaded.
DB_UPLOAD = int(os.getenv("DB_UPLOAD", "1"))
DRY_RUN_SAMPLE_ROWS = 5

UDP_IP = os.getenv("UDP_IP", "0.0.0.0")
# The three streams are told apart only by destination port: the payload carries no
# source marker.
PORTS = {
    "base": int(os.getenv("PORT_BASE", "5005")),
    "rover": int(os.getenv("PORT_ROVER", "5006")),
    "nav": int(os.getenv("PORT_NAV", "5007")),
}

# Sateliot's nordic_pass prepends the UE's IMSI and a ';' (e.g. b"001010000150258;") to every
# datagram it forwards. Left in, it lands inside RTCM frames and breaks their CRC. RTCM bytes
# at the start of a datagram are never 15 ASCII digits plus ';', so stripping it is safe.
UE_PREFIX = re.compile(rb"^\d{15};")

# The satellite link asks for ASCII payloads, so senders may base64-encode. "auto" accepts
# both, per datagram: one made only of base64 characters is decoded, anything else is taken
# as raw RTCM. 1400 random-looking RTCM bytes all falling in the 64-character alphabet has
# probability ~(65/256)^1400, i.e. never. Below AUTO_BASE64_MIN_CHARS the test is not
# conclusive and the datagram is kept raw. "raw" or "base64" force one format.
PAYLOAD_ENCODING = os.getenv("PAYLOAD_ENCODING", "auto").strip().lower()
BASE64_DATAGRAM = re.compile(rb"[A-Za-z0-9+/]*={0,2}")
AUTO_BASE64_MIN_CHARS = 16

# GPS and Galileo MSM1-7: both carry a 30-bit time of week in ms right after the 12-bit
# message type and the 12-bit station id (DF004 / DF248), on the same GPS-aligned scale.
MSM_TOW_TYPES = set(range(1071, 1078)) | set(range(1091, 1098))
# Largest possible RTCM3 frame: 3-byte header, 1023-byte payload, 3-byte CRC.
MAX_FRAME_BYTES = 1029


def _crc24q_table():
    table = []
    for i in range(256):
        crc = i << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= 0x1864CFB
        table.append(crc & 0xFFFFFF)
    return table


CRC24Q_TABLE = _crc24q_table()


def crc24q(data):
    crc = 0
    for byte in data:
        crc = ((crc << 8) & 0xFFFFFF) ^ CRC24Q_TABLE[(crc >> 16) ^ byte]
    return crc


def rtcm_frames(data):
    """Yield (start, end) of every RTCM3 frame in `data` whose CRC checks.

    Bytes lost or corrupted on the link only cost the frames they hit. An incomplete frame
    at the end of `data` is not yielded.
    """
    i = 0
    while True:
        i = data.find(b"\xd3", i)
        if i < 0 or i + 6 > len(data):
            return
        end = i + 3 + (((data[i + 1] & 0x03) << 8) | data[i + 2]) + 3
        if end > len(data) or crc24q(data[i:end - 3]) != int.from_bytes(data[end - 3:end], "big"):
            i += 1
            continue
        yield i, end
        i = end


def msm_tow(frame):
    """GPS time of week in ms of a GPS/Galileo MSM frame, or None for any other message.

    The week number is not in MSM messages: a window crossing the Saturday/Sunday
    rollover is seen as two runs.
    """
    if len(frame) >= 13 and ((frame[3] << 4) | (frame[4] >> 4)) in MSM_TOW_TYPES:
        return (int.from_bytes(frame[3:11], "big") >> 10) & 0x3FFFFFFF
    return None


def msm_epochs(data):
    """Group the GPS/Galileo MSM frames of an RTCM3 capture by epoch: {tow_ms: frames}."""
    epochs = {}
    for start, end in rtcm_frames(data):
        tow = msm_tow(data[start:end])
        if tow is not None:
            epochs[tow] = epochs.get(tow, b"") + data[start:end]
    return epochs


def epoch_second(tow, step=ANOMALY_EPOCH_STEP_MS):
    """Epoch number of a time of week in ms: receivers (u-blox) time-tag 1 Hz epochs at x.999
    or x.001 s for long stretches, so epochs are counted by the nearest second."""
    return (tow + step // 2) // step


def round_msm_tow(frame):
    """The same MSM frame with its time of week rounded to the whole second.

    The anomaly service numbers epochs by int(tow_ms / 1000), which truncates: at a drift from
    x.000 to x.999 two epochs fall in one second and it counts one epoch less than we sent,
    which shifts every window after it. With whole-second tags both counts agree.
    """
    tow = msm_tow(frame)
    rounded = epoch_second(tow) * ANOMALY_EPOCH_STEP_MS % (7 * 86400 * 1000)
    if rounded == tow:
        return frame
    bits = int.from_bytes(frame[3:11], "big") & ~(0x3FFFFFFF << 10) | (rounded << 10)
    body = frame[:3] + bits.to_bytes(8, "big") + frame[11:-3]
    return body + crc24q(body).to_bytes(3, "big")


def consecutive_windows(tows, size=ANOMALY_WINDOW_EPOCHS):
    """Split sorted epoch times into non-overlapping runs of `size` epochs in consecutive seconds.

    Returns (windows, tail): tail is the incomplete run at the end, which later data may
    still complete. Incomplete runs before a gap never will and are left out: the detector
    was trained on uninterrupted 1 Hz sequences only.
    """
    windows, run = [], []
    for tow in tows:
        if run and epoch_second(tow) - epoch_second(run[-1]) != 1:
            run = []
        run.append(tow)
        if len(run) == size:
            windows.append(run)
            run = []
    return windows, run


class GNSSBatchManager:
    def __init__(self):
        self.path = OUTPUT_DIR
        self.path.mkdir(parents=True, exist_ok=True)

        # Listener threads append while the watchdog closes a window: both under this lock.
        self.lock = threading.Lock()
        # One pipeline at a time: consecutive windows share last_nav.
        self.pipeline_lock = threading.Lock()
        self.last_activity = time.time()
        # Data arrived since the last window was closed. Without it the idle trigger would
        # fire every second on a leftover partial frame.
        self.pending = False
        # Ephemerides stay valid for hours and are not repeated in every window, so the
        # RINEX nav of the previous window is passed to rnx2rtkp along with the new one.
        self.last_nav = None
        # Rover and base record continuously and send their backlog at every pass, so a run
        # of consecutive epochs spans windows. Epochs not yet analysed ({tow_ms: frames}, the
        # incomplete run at the end) are kept until the run reaches ANOMALY_WINDOW_EPOCHS;
        # epochs up to analysed_until are never sent again. Both live in memory only: a
        # restart loses at most one incomplete run.
        self.anomaly_epochs = {}
        self.analysed_until = -ANOMALY_EPOCH_STEP_MS
        # GPS week of the latest solution: the anomaly labels are applied to rows already in
        # the database by (week, time of week), and MSM messages carry no week.
        self.last_week = None
        self.pending_updates = self.path / "anomaly_updates.json"

        logging.info(f"A window is processed at {ROVER_BUFFER_EPOCHS} rover epochs or after "
                     f"{IDLE_TIMEOUT:g} s without data")
        missing = [v for v in ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD") if not os.getenv(v)]
        if not DB_UPLOAD:
            logging.warning("DB_UPLOAD=0: nothing is sent to the database, the rows are logged instead")
        elif missing:
            logging.error(f"{', '.join(missing)} not set: uploads will fail and every .pos will be "
                          f"kept in {self.path} for retry")

        self.sweep_orphans()
        self.rotate_filenames()

        threading.Thread(target=self.watchdog, daemon=True).start()

    def rotate_filenames(self):
        """Mint fresh filenames so following data goes to empty buffers."""
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        self.files = {stream: self.path / f"{stream}_{ts}.rtcm3" for stream in PORTS}
        self.file_output = self.path / f"output_{ts}.pos"
        self.rover_datagrams = 0
        self.rover_bytes = 0
        # Incremental scan of the rover buffer: first frame offset of each epoch, in arrival order
        self.rover_scanned = 0
        self.rover_epoch_starts = {}

    def add_data(self, stream, data):
        with self.lock:
            with open(self.files[stream], "ab") as f:
                f.write(data)
            if stream == "rover":
                if self.rover_datagrams == 0:
                    logging.info("ROVER: receiving data")
                self.rover_datagrams += 1
                self.rover_bytes += len(data)
            self.last_activity = time.time()
            self.pending = True

    def buffer_size(self, stream):
        path = self.files[stream]
        return path.stat().st_size if path.exists() else 0

    def rover_cut(self):
        """Offset where epoch ROVER_BUFFER_EPOCHS + 1 starts in the rover buffer, or None.

        Only the bytes added since the last call are scanned, from the end of the last
        complete frame, so a frame still arriving is picked up on the next call.
        """
        path = self.files["rover"]
        if not path.exists() or path.stat().st_size <= self.rover_scanned:
            return None
        with open(path, "rb") as f:
            f.seek(self.rover_scanned)
            data = f.read()
        offset = self.rover_scanned
        for start, end in rtcm_frames(data):
            tow = msm_tow(data[start:end])
            if tow is not None and tow not in self.rover_epoch_starts:
                self.rover_epoch_starts[tow] = offset + start
            self.rover_scanned = offset + end
        if len(self.rover_epoch_starts) > ROVER_BUFFER_EPOCHS:
            return list(self.rover_epoch_starts.values())[ROVER_BUFFER_EPOCHS]
        return None

    def watchdog(self):
        while True:
            time.sleep(1)
            with self.lock:
                cut = self.rover_cut()
                sizes = {stream: self.buffer_size(stream) for stream in PORTS}
                idle = time.time() - self.last_activity > IDLE_TIMEOUT
                if cut is not None:
                    reason = "rover buffer full"
                elif idle and self.pending:
                    reason = "no data for {:g} s".format(IDLE_TIMEOUT)
                elif max(sizes.values()) > MAX_BUFFER_BYTES:
                    reason = "buffer size limit"
                else:
                    continue
                window = self.close_window(cut, reason)
            threading.Thread(target=self.run_pipeline, args=(window,), daemon=True).start()

    def close_window(self, rover_cut, reason):
        """Hand the current buffers to the pipeline and start new ones.

        Data keeps arriving while a window is closed, so a frame may be split between the
        old and the new buffer. Whatever follows the last complete frame is moved to the new
        buffer instead of being lost; for the rover, so is everything from the first frame
        of the epoch past the buffer limit, so that no epoch is split between two windows.
        """
        logging.info(f"Window closed ({reason}): rover {len(self.rover_epoch_starts)} epochs, "
                     f"{self.rover_datagrams} datagrams / {self.rover_bytes} bytes. Processing...")
        window = dict(self.files, pos=self.file_output)
        self.rotate_filenames()
        # Epochs moved past a full rover buffer still have to be processed, even if nothing follows
        self.pending = rover_cut is not None
        for stream in PORTS:
            old = window[stream]
            if not old.exists():
                continue
            data = old.read_bytes()
            if stream == "rover" and rover_cut is not None:
                cut = rover_cut
            else:
                last_end = max((end for _, end in rtcm_frames(data)), default=0)
                cut = max(last_end, len(data) - MAX_FRAME_BYTES)
            if cut < len(data):
                self.files[stream].write_bytes(data[cut:])
                with open(old, "r+b") as f:
                    f.truncate(cut)
        return window

    def detect_anomaly(self, rover_file):
        """Run the detector on every run of ANOMALY_WINDOW_EPOCHS consecutive 1 Hz rover epochs.

        The epochs of this window are added to those kept from previous windows, so a run
        may span several passes. Returns [(first_tow_s, last_tow_s, flag), ...], one entry
        per window analysed. The windows are sent in a single file; the service cuts it into
        chunks of the same size, in order, and answers one prediction per chunk.
        """
        received = msm_epochs(rover_file.read_bytes())
        seconds = {epoch_second(tow) for tow in self.anomaly_epochs}
        new = {}
        for tow, frames in sorted(received.items()):
            second = epoch_second(tow)
            if second > epoch_second(self.analysed_until) and second not in seconds:
                new[tow] = frames
                seconds.add(second)
        self.anomaly_epochs.update(new)
        epochs = self.anomaly_epochs
        windows, tail = consecutive_windows(sorted(epochs))
        self.anomaly_epochs = {tow: epochs[tow] for tow in tail}
        if not windows:
            logging.info(f"ANOMALY: {len(new)} new rover epochs, run of {len(tail)}/{ANOMALY_WINDOW_EPOCHS} "
                         f"consecutive 1 s epochs: waiting for more")
            return []
        # Never sent again, even if the detector fails: each call is a job on the Rupia service
        self.analysed_until = windows[-1][-1]

        window_file = rover_file.with_name(rover_file.name.replace("rover_", "anomaly_"))
        with open(window_file, "wb") as f:
            for window in windows:
                for tow in window:
                    frames = epochs[tow]
                    f.write(b"".join(round_msm_tow(frames[s:e]) for s, e in rtcm_frames(frames)))
        logging.info(f"ANOMALY: {len(new)} new rover epochs, {len(windows)} window(s) of "
                     f"{ANOMALY_WINDOW_EPOCHS} consecutive epochs sent to the detector, "
                     f"{len(tail)} epochs waiting for the next")

        labels = []
        try:
            response = requests.post(AI_PREDICTOR_URL, json={"file": f"{ANOMALY_SHARED_DIR}/{window_file.name}"},
                                     timeout=ANOMALY_TIMEOUT_S * len(windows))
            if response.status_code != 200:
                logging.error(f"Anomaly server returned {response.status_code}: {response.text[:500]}")
            else:
                logging.info(f"ANOMALY JSON: {response.text}")
                predictions = response.json().get("Predictions") or []
                # The service skips a window whose inference fails and still answers 200, so
                # a short list cannot be matched to the windows sent.
                if len(predictions) != len(windows):
                    logging.warning(f"{len(predictions)} predictions for {len(windows)} windows: "
                                    f"positions left without anomaly label")
                else:
                    labels = [(w[0] / 1000, w[-1] / 1000, int(p)) for w, p in zip(windows, predictions)]
        except Exception as e:
            logging.error(f"Could not reach the anomaly server: {e}")
        self.free_space([window_file], "anomaly windows")
        return labels

    def debug_results(self, result, output):
        logging.info(f"DEBUG RTKLIB: {output.name}\n--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}")

    def run(self, command, output):
        result = subprocess.run([str(c) for c in command], capture_output=True, text=True)
        if DEBUG:
            self.debug_results(result, output)
        return output.exists() and output.stat().st_size > 0

    def convbin_cmd(self, *args):
        """Build the convbin command, adding -ts when replaying an old capture.

        Without -ts, convbin derives its time reference from the modification time of
        the input file, and the files here have just been written (mtime = now). For a
        capture recorded weeks earlier that resolves the data to the wrong week: the GPS
        week stored in the database comes out wrong, and if the observation and navigation
        streams shift differently rnx2rtkp produces no epochs at all.
        """
        command = [CONVBIN_PATH, "-r", "rtcm3"]
        if RTCM_START_TIME:
            command += ["-ts"] + RTCM_START_TIME.split()
        return command + list(args)

    def run_pipeline(self, window):
        with self.pipeline_lock:
            try:
                self.process_window(window)
            except Exception as e:
                logging.error(f"RTKLIB pipeline error: {e}")

    def process_window(self, window):
        base, rover, nav, pos = (Path(window[k]) for k in ("base", "rover", "nav", "pos"))
        obs_base = base.with_suffix(".obs")
        obs_rover = rover.with_suffix(".obs")
        nav_rinex = nav.with_suffix(".nav")
        pos_events = pos.with_name(f"{pos.stem}_events{pos.suffix}")
        have = {p: p.exists() and p.stat().st_size > 0 for p in (base, rover, nav)}
        intermediates = [base, rover, nav, obs_base, obs_rover, pos_events]

        if have[nav] and self.run(self.convbin_cmd("-n", nav_rinex, nav), nav_rinex):
            navs = [n for n in (self.last_nav, nav_rinex) if n]
            if self.last_nav:
                intermediates.append(self.last_nav)
            self.last_nav = nav_rinex
        else:
            navs = [self.last_nav] if self.last_nav else []
            intermediates.append(nav_rinex)

        if not have[rover]:
            logging.info("Window without rover data: ephemerides kept, nothing to solve")
            self.free_space(intermediates, "window without rover")
            return

        # Before solving, which deletes the rover buffer; the labels are applied after the
        # upload, since they may also cover rows of this window
        labels = self.detect_anomaly(rover)
        try:
            self.solve(rover, base, obs_rover, obs_base, navs, pos, have, intermediates)
        finally:
            self.queue_labels(labels)
            self.flush_labels()

    def solve(self, rover, base, obs_rover, obs_base, navs, pos, have, intermediates):
        """Compute and upload the positions of a window, then delete its files."""
        if not navs:
            logging.error("No ephemerides received yet: rover window discarded")
            self.free_space(intermediates + [pos], "window without ephemerides")
            return
        if not self.run(self.convbin_cmd("-o", obs_rover, rover), obs_rover):
            logging.error(f"convbin found no observations in {rover.name}: window discarded")
            self.free_space(intermediates + [pos], "unreadable rover")
            return

        # A missing -k file is not an error for rnx2rtkp: it silently reverts to its
        # defaults and never fixes, so the absence is reported here.
        if not Path(RTKLIB_CONFIG_FILE).is_file():
            logging.error(f"{RTKLIB_CONFIG_FILE} not found: rnx2rtkp falls back to its defaults")

        mode = "RTK"
        solved = (have[base] and self.run(self.convbin_cmd("-o", obs_base, base), obs_base)
                  and self.run([RNX2RTKP_PATH, "-k", RTKLIB_CONFIG_FILE, "-o", pos, obs_rover, obs_base, *navs], pos)
                  and self.count_epochs(pos)[1] > 0)
        if not solved:
            # No base, or base not overlapping the rover: -p 0 overrides the posmode of the
            # config (rnx2rtkp loads -k before reading the other options).
            mode = "single"
            logging.info("No base observations overlapping this window: single point positioning"
                         if have[base] else "No base observations for this window: single point positioning")
            self.run([RNX2RTKP_PATH, "-k", RTKLIB_CONFIG_FILE, "-p", "0", "-o", pos, obs_rover, *navs], pos)

        fixed, total = self.count_epochs(pos)
        if total == 0:
            logging.warning(".POS file is empty: no position for this window")
            self.free_space(intermediates + [pos], "window without solution")
            return
        # analyze_logs_testbed.py parses this exact wording
        logging.info(f"RTK Result: {fixed}/{total} FIXED epochs ({fixed / total * 100:.1f}%), mode {mode}")
        with open(pos) as f:
            self.last_week = next(int(line.split()[0]) for line in f if line.strip() and not line.startswith("%"))

        uploaded = upload_pos_file(pos, ROVER_ID)
        if uploaded:
            intermediates.append(pos)
        else:
            logging.warning(f"Database upload failed: keeping {pos.name} to retry")
        self.free_space(intermediates, "processed window")
        if uploaded:
            self.retry_pending_pos(current_pos=pos)

    def count_epochs(self, pos_file):
        """Return (fixed, total) epochs of a .pos file; Q=1 is FIXED."""
        fixed = total = 0
        if pos_file.exists():
            with open(pos_file) as f:
                for line in f:
                    if line.startswith("%") or not line.strip():
                        continue
                    total += 1
                    parts = line.split()
                    if len(parts) > 5 and parts[5] == "1":
                        fixed += 1
        return fixed, total

    def sweep_orphans(self):
        """Delete files left behind mid-pipeline by a previous run.

        The .pos files are left alone: they may hold positions that never reached the
        database, and retry_pending_pos() picks them up.
        """
        orphans = set()
        for pattern in ("*.rtcm3", "*.obs", "*.nav", "*_events.pos"):
            orphans.update(self.path.glob(pattern))
        if orphans:
            logging.warning(f"Startup: {len(orphans)} orphan files from a previous run")
            self.free_space(sorted(orphans), "startup orphans")

    def load_pending_labels(self):
        try:
            return json.loads(self.pending_updates.read_text()) if self.pending_updates.is_file() else []
        except Exception as e:
            logging.error(f"Could not read {self.pending_updates.name}: {e}")
            return []

    def queue_labels(self, labels):
        """Add anomaly labels to the file of updates still to be applied to the database.

        Kept on disk, like a .pos whose upload failed, so neither a database outage nor a
        restart loses them.
        """
        if labels:
            pending = self.load_pending_labels()
            pending += [[self.last_week, first, last, flag] for first, last, flag in labels]
            self.pending_updates.write_text(json.dumps(pending))

    def flush_labels(self):
        """Apply the pending anomaly labels to the rows already in the database.

        Runs after retry_pending_pos(), so the rows a label covers are inserted first. A
        label computed before any solution has no GPS week yet and takes the latest one.
        """
        pending = self.load_pending_labels()
        if not pending:
            return
        for label in pending:
            if label[0] is None:
                label[0] = self.last_week
        ready = [label for label in pending if label[0] is not None]
        if ready and update_anomaly(ROVER_ID, ready):
            pending = [label for label in pending if label[0] is None]
        if pending:
            self.pending_updates.write_text(json.dumps(pending))
            logging.warning(f"{len(pending)} anomaly label(s) pending, kept in {self.pending_updates.name}")
        else:
            self.pending_updates.unlink(missing_ok=True)

    def retry_pending_pos(self, current_pos=None):
        """Retry the .pos files kept on disk because the database was unreachable.

        Runs after every successful upload, the best moment to drain the backlog, and
        stops at the first failure, since it means the database went away again.
        """
        pending = sorted(p for p in self.path.glob("output_*.pos")
                         if not p.name.endswith("_events.pos")
                         and (current_pos is None or p.name != current_pos.name))
        if not pending:
            return
        logging.info(f"Retrying {len(pending)} .pos files pending upload")
        for pos_pending in pending:
            if upload_pos_file(pos_pending, ROVER_ID):
                self.free_space([pos_pending], f"pending {pos_pending.name} uploaded")
            else:
                logging.warning(f"{pos_pending.name} still pending: stopping retries")
                break

    def free_space(self, files_to_delete, reason=""):
        """Delete files that are no longer needed and log how much space was freed.

        Disk is limited on the satellite: the .obs/.nav are regenerable and the .rtcm3/.pos
        have already been processed and uploaded, so none of them are kept.
        """
        freed = 0
        for f_del in files_to_delete:
            try:
                if f_del.exists():
                    freed += f_del.stat().st_size
                    f_del.unlink()
            except Exception as e:
                logging.error(f"Could not delete {f_del}: {e}")
        logging.info(f"Cleanup ({reason}): {freed / 1024:.0f} KB freed")


def db_connect():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        # Without this, an unreachable database blocks the pipeline thread for
        # about 21 s before giving up. libpq accepts a minimum of 2 s.
        connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT", "5"))
    )


def upload_pos_file(file_path, rover_id):
    """Insert every row of a .pos file into the database; returns 1 on success, 0 otherwise.

    The anomaly column is inserted as NULL ("not analysed") and set later by
    update_anomaly(), once the run of epochs a row belongs to is complete.
    """
    records_to_insert = []
    try:
        with open(file_path) as f:
            for line in f:
                if line.startswith('%') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 13:
                    records_to_insert.append((
                        rover_id, int(parts[0]), float(parts[1]),
                        float(parts[2]), float(parts[3]), float(parts[4]), int(parts[5]),
                        float(parts[10]), float(parts[11]), float(parts[12]),
                    ))
    except FileNotFoundError:
        logging.error(f"File '{file_path}' not found.")
        return 0

    if not records_to_insert:
        logging.info("No valid data found to insert.")
        return 0

    if not DB_UPLOAD:
        rows = records_to_insert
        if len(rows) > DRY_RUN_SAMPLE_ROWS:
            rows = rows[:DRY_RUN_SAMPLE_ROWS - 2] + [None] + rows[-2:]
        lines = ["rover_id  gps_week  gps_tow  latitude  longitude  height  q_flag  sdne  sdeu  sdun  anomaly"]
        lines += ["  ..." if r is None else
                  f"{r[0]}  {r[1]}  {r[2]:.3f}  {r[3]:.9f}  {r[4]:.9f}  {r[5]:.4f}  {r[6]}  "
                  f"{r[7]:.4f}  {r[8]:.4f}  {r[9]:.4f}  NULL" for r in rows]
        logging.info(f"DB (DB_UPLOAD=0): would insert {len(records_to_insert)} rows into rtk_positions:\n  "
                     + "\n  ".join(lines))
        return 1

    try:
        connection = db_connect()
        cursor = connection.cursor()
        # anomaly NULL explicitly: the table's column may default to 0, which means "normal"
        cursor.executemany("""
            INSERT INTO rtk_positions
            (rover_id, gps_week, gps_tow, latitude, longitude, height, q_flag, sdne, sdeu, sdun, anomaly)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL)
            ON CONFLICT (rover_id, gps_week, gps_tow) DO NOTHING
            """, records_to_insert)
        connection.commit()
        logging.info(f"Success: Inserted {cursor.rowcount} records into DB for rover '{rover_id}'.")
        return 1
    except Exception as error:
        logging.error(f"Database error: {error}")
        return 0
    finally:
        # No return statement may be added here: it would override the except's
        # return 0, and the caller would delete the .pos as if it had been uploaded.
        if 'connection' in locals() and connection:
            cursor.close()
            connection.close()


def update_anomaly(rover_id, labels):
    """Set the anomaly flag of the rows covered by each [week, first_tow_s, last_tow_s, flag].

    Limitation: a window crossing the end of a GPS week is labelled with one week only;
    handling it would need the week of every epoch, which MSM messages do not carry.
    """
    if not DB_UPLOAD:
        for week, first, last, flag in labels:
            logging.info(f"DB (DB_UPLOAD=0): would set anomaly = {flag} on rover {rover_id}, "
                         f"week {week}, gps_tow {first:.3f}-{last:.3f}")
        return 1
    try:
        connection = db_connect()
        cursor = connection.cursor()
        updated = 0
        for week, first, last, flag in labels:
            cursor.execute("""
                UPDATE rtk_positions SET anomaly = %s
                WHERE rover_id = %s AND gps_week = %s AND gps_tow BETWEEN %s AND %s
                """, (flag, rover_id, week, first - 0.5, last + 0.5))
            updated += cursor.rowcount
        connection.commit()
        logging.info(f"Anomaly labels of {len(labels)} window(s) applied to {updated} rows")
        return 1
    except Exception as error:
        logging.error(f"Database error applying anomaly labels: {error}")
        return 0
    finally:
        if 'connection' in locals() and connection:
            cursor.close()
            connection.close()


class UDPSocketReceiver:
    def __init__(self, ip=UDP_IP):
        self.ip = ip
        self.running = False

        # Logging must be configured BEFORE the batch manager is instantiated: its
        # __init__ already logs (the orphan sweep) and those lines would be lost.
        os.makedirs("logs", exist_ok=True)
        rotating = logging.handlers.TimedRotatingFileHandler(
            "logs/server_execution.log",
            when=LOG_ROTATE_WHEN,
            interval=LOG_ROTATE_INTERVAL,
            backupCount=LOG_BACKUP_COUNT,
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - [SERVER] - %(message)s',
            handlers=[rotating, logging.StreamHandler()]
        )

        if PAYLOAD_ENCODING not in ("auto", "raw", "base64"):
            raise SystemExit(f"PAYLOAD_ENCODING must be 'auto', 'raw' or 'base64', not {PAYLOAD_ENCODING!r}")

        self.batch_manager = GNSSBatchManager()
        # One queue per stream, so a slow disk write never delays the socket
        self.queues = {stream: Queue() for stream in PORTS}
        self.sockets = []

    def start(self):
        self.running = True
        for stream, port in PORTS.items():
            threading.Thread(target=self.listen_udp, args=(port, self.queues[stream], stream.upper()),
                             daemon=True).start()
            threading.Thread(target=self.worker, args=(stream,), daemon=True).start()
        try:
            logging.info("Init GNSS Multi-Port server...")
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def listen_udp(self, port, target_queue, name):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.ip, port))
        self.sockets.append(sock)
        logging.info(f"Listening {name} at {self.ip}:{port} (payload encoding: {PAYLOAD_ENCODING})")

        prefix_seen = False
        # A whole burst sent with the wrong encoding is many bad datagrams: report it as a
        # count every 10 s rather than one line each.
        bad_base64, bad_base64_reported = 0, 0.0
        detected = None
        while self.running:
            try:
                data, addr = sock.recvfrom(2048)

                match = UE_PREFIX.match(data)
                if match:
                    if not prefix_seen:
                        logging.info(f"{name}: stripping UE prefix {match.group().decode()} from datagrams")
                        prefix_seen = True
                    data = data[match.end():]

                if PAYLOAD_ENCODING == "auto":
                    is_base64 = (len(data) >= AUTO_BASE64_MIN_CHARS and len(data) % 4 == 0
                                 and BASE64_DATAGRAM.fullmatch(data) is not None)
                    if is_base64:
                        data = base64.b64decode(data)
                    # Logged on change only, so the log shows which format the link delivers
                    if len(data) >= AUTO_BASE64_MIN_CHARS and detected != is_base64:
                        detected = is_base64
                        logging.info(f"{name}: datagrams arriving as {'base64' if is_base64 else 'raw binary'}")
                elif PAYLOAD_ENCODING == "base64":
                    try:
                        data = base64.b64decode(data, validate=True)
                    except (binascii.Error, ValueError):
                        bad_base64 += 1
                        if time.time() - bad_base64_reported > 10:
                            logging.error(f"{name}: dropped {bad_base64} datagram(s) that are not valid base64")
                            bad_base64, bad_base64_reported = 0, time.time()
                        continue

                if data:
                    target_queue.put(data)
            except Exception as e:
                if self.running:
                    logging.error(f"Socket Error {name} ({port}): {e}")

    def worker(self, stream):
        while self.running:
            try:
                self.batch_manager.add_data(stream, self.queues[stream].get())
            except Exception as e:
                logging.error(f"Could not buffer {stream} data: {e}")

    def stop(self):
        self.running = False
        logging.info("Stop. UDP Socket close...")
        for sock in self.sockets:
            try:
                sock.close()
            except Exception as e:
                logging.error(f"Error closing socket: {e}")


if __name__ == "__main__":
    UDPSocketReceiver().start()
