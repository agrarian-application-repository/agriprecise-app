## AGRI-PRECISE

AGRI-PRECISE computes centimetre-level RTK positions for agricultural machinery from GNSS
correction data delivered over a non-terrestrial (satellite NB-IoT) link, and flags anomalous
GNSS observations with an autoencoder running on an onboard AI co-processor. The application
is designed for testbed 3 (LEO NB-IoT with onboard AI: Sateliot constellation + Ubotica
CogniSAT) and for the precision-agriculture positioning use case. It was developed by the
Wireless Networks (WiNE) research group of the Universitat Oberta de Catalunya (UOC).

### Node Type
Satellite. Both containers run on the satellite node: the receiver and RTK engine
(`ntn_server`) and the anomaly detector (`ntn_anomaly`), which runs its model on the Ubotica
CogniSAT-XE2 (Myriad X VPU) through Ubotica's `csa` application, falling back to the same model
in ONNX on the CPU when `csa` fails or takes over 30 s. Positions are written to the AGRARIAN
PostgreSQL data storage.

### Application Objective
- Receive three RTCM3 streams over UDP: rover observations, base station observations and
  navigation messages (ephemerides).
- Buffer them and, when the rover buffer is full or the link has been idle for a while (the
  end of a satellite pass), compute the positions with RTKLIB: **RTK** when base observations
  are available, **single point positioning** otherwise.
- Analyse every run of **120 consecutive rover epochs at 1 s** with the anomaly detector,
  waiting across passes until a run is complete, and label the positions it covers.
- Insert every position into the `rtk_positions` table, keeping on disk and retrying whatever
  could not be uploaded.

### Use Case Description
RTK positioning needs a continuous stream of corrections from a nearby base station, which
rural fields often lack because there is no terrestrial coverage. AGRI-PRECISE studies whether
the same accuracy can be obtained when the data travels over a store-and-forward satellite
link: messages of a few hundred bytes, delivered in passes of a few minutes. The receiver
accumulates what each pass brings and solves once the pass is over, and the onboard anomaly
detector screens the raw observables before the positions reach
the farm management side. On the reference dataset, 77 % of the epochs reach an RTK FIXED
solution (Q=1), with a dispersion of 6-22 mm.

### Anomaly Detection Model
A convolutional autoencoder learns to reconstruct nominal GNSS observations and flags as
anomalous (jamming, spoofing) the 120-epoch windows it reconstructs badly. Input: GPS (L1, L2)
and Galileo (E1, E5a, E5b) MSM observations parsed from RTCM 107x/109x, as a `1 x 4 x 148 x 120`
tensor (C/N0, pseudorange, carrier phase and code-minus-carrier for 148 satellite-signal pairs
over 120 epochs at 1 s), normalised with `data_scaler.json`. Output: the reconstruction error,
mapped to a probability by a calibrated lognormal CDF and thresholded at 0.93. It was trained
on the public GNSS interference and spoofing dataset of Wang et al. (Data in Brief, 2024), and
is adapted to a new site or receiver by recalibrating the CDF and the normalisation statistics,
without retraining (as done for the drone dataset below).

Full documentation by the model's author, in `docs/anomaly_detection/`:

| Document | Contents |
|---|---|
| [`AnomalyGeneralReadme/anomaly_general_readme.md`](docs/anomaly_detection/AnomalyGeneralReadme/anomaly_general_readme.md) | Model objective, preprocessing and tensor structure, architecture, anomaly scoring, datasets and training, performance, recalibration protocol, known limitations |
| [`TestbedReadme/anomaly_docker_integration_readme.md`](docs/anomaly_detection/TestbedReadme/anomaly_docker_integration_readme.md) | The `ntn_anomaly` container: files, inference engine modes, pre- and postprocessing |

The trained model with its standalone PyTorch inference pipeline is published separately on
Zenodo by its author.

### Use instructions

#### Repository structure

```text
├── agrarian_app.py         # Receiver and RTK engine (ntn_server)
├── Dockerfile.rtk          # ntn_server image: Python + RTKLIB (convbin, rnx2rtkp) built from source
├── config/                 # RTKLIB processing options
├── agriprecise_files/      # Anomaly detection service (ntn_anomaly), as delivered by its author:
│   ├── http_server.py      #   HTTP API, preprocessing, csa / ONNX inference, scoring
│   ├── http_client.py      #   manual test client
│   └── data/               #   model (.unn for the CogniSAT, .onnx for the CPU), scaler, calibration,
│                           #   csa_config.json.example
├── Dockerfile.rupia        # ntn_anomaly image
├── ubotica/                # Mounted at /ubotica: Ubotica's bin/csa and its data folders (not included)
├── docker-compose.yml      # Both services, plus the test launcher
├── .env.example            # Configuration template
├── data/                   # Sample captures: 9 min of July 2026 (B_, R_, N_) and the drone test (drone/)
├── tools/                  # RTCM simulator for rover, base and nav, pass launcher, analysis scripts
└── docs/anomaly_detection/ # Documentation of the anomaly detection model
```

#### Architecture

```text
 UDP 5005  base  ─┐
 UDP 5006  rover ─┼─► buffers ─► convbin ─► rnx2rtkp ─► .pos ─► PostgreSQL rtk_positions
 UDP 5007  nav   ─┘                │      (RTK or single)       ▲
                                   └─► 120 consecutive epochs ─► ntn_anomaly (csa / ONNX) ─┘ anomaly label
```

A window is closed and processed when either:
- the rover buffer holds `ROVER_BUFFER_EPOCHS` epochs (360 = 6 minutes at 1 Hz), or
- no data has arrived on any stream for `IDLE_TIMEOUT` seconds (60), i.e. a pass has ended.

Each window is solved independently and its result uploaded, whatever its quality: the
`q_flag` column tells FIXED (1), FLOAT (2) and SINGLE (5) apart. Ephemerides are kept from one
window to the next, since they stay valid for hours. Without any ephemerides yet, a rover
window cannot be solved and is discarded.

Rover and base record continuously and send their backlog at every pass, so a run of
consecutive epochs usually spans several passes. The anomaly detector therefore works apart
from the windows: rover epochs are kept until 120 consecutive ones at 1 s are available, then
sent to the detector, and the prediction is written to the rows of those epochs that are
already in the database (`UPDATE`). Until then their `anomaly` is NULL. A gap in time drops
the incomplete run before it, which can no longer be completed. Epochs are counted by the
nearest second, so receivers that time-tag at x.999 or x.001 s (u-blox) are handled.

#### Prerequisites
- Docker with Compose v2.
- Network access from the satellite node to the AGRARIAN PostgreSQL (`DB_HOST`).
- For inference on the CogniSAT (`INFERENCE_ENGINE_MODE` `hybrid_engine` or `csa_engine`), files
  supplied by Ubotica and the model's author, not included in this repository:
  - `ubotica/bin/csa`, Ubotica's AI deployment application, provided by the host together with
    `ubotica/data/nbiot_input/` and `ubotica/data/output/`, all mounted at `/ubotica` as
    Ubotica's container guidelines require;
  - `agriprecise_files/data/unnHelpers/csa_config.json`: copy `csa_config.json.example` and put
    in the Rupia token. It is baked into the `ntn_anomaly` image, so that image must not be
    pushed to a public registry.

  Without them `hybrid_engine` still labels every window, with ONNX on the CPU; `onnx_engine`
  never calls `csa`.

#### Configuration
Copy `.env.example` to `.env` and fill it in.

| Variable | Required | Default | Description |
|---|---|---|---|
| `DB_HOST` | yes | - | AGRARIAN PostgreSQL host (`10.160.101.65`) |
| `DB_PORT` | no | `5432` | PostgreSQL port |
| `DB_NAME` | yes | - | Database (`agri_precise_db`) |
| `DB_USER` | yes | - | User (`agri_precise`) |
| `DB_PASSWORD` | yes | - | Password, from the AGRARIAN onboarding |
| `DB_CONNECT_TIMEOUT` | no | `5` | Seconds before an upload is given up and kept for retry |
| `DB_UPLOAD` | no | `1` | `0`: no database; a sample of the rows that would be inserted, and the anomaly updates, are written to the log |
| `ROVER_ID` | no | `ROVER_01` | Value of the `rover_id` column |
| `ROVER_BUFFER_EPOCHS` | no | `360` | Rover epochs per window when the stream never pauses |
| `IDLE_TIMEOUT` | no | `60` | Seconds without data that close a window |
| `MAX_BUFFER_BYTES` | no | 20 MB | Safety cap per stream buffer |
| `PAYLOAD_ENCODING` | no | `auto` | `auto` accepts raw RTCM3 and base64 datagrams alike; `raw`, `base64` |
| `RTKLIB_CONFIG_FILE` | no | `./config/config_rtkpost_ntn.conf` | RTKLIB processing options |
| `OUTPUT_DIR` | no | `./output` | Buffers and pending results (`/app/output` in the compose) |
| `RTCM_START_TIME` | no | empty | Only when replaying an old capture: its start, `YYYY/MM/DD hh:mm:ss` |
| `PORT_BASE` / `PORT_ROVER` / `PORT_NAV` | no | `5005` / `5006` / `5007` | UDP ports, on the host and in the container |
| `INFERENCE_ENGINE_MODE` | no | `hybrid_engine` | `hybrid_engine` (`csa`, ONNX on the CPU if it fails or takes over 30 s), `csa_engine`, `onnx_engine` |
| `UBOTICA_DIR` | no | `./ubotica` | Host folder mounted at `/ubotica` (`bin/csa`, `data/nbiot_input/`, `data/output/`) |
| `ANOMALY_SCALER` | no | built-in | Normalisation statistics of the receiver the data comes from (`./data/drone/data_scaler.json` for the drone dataset) |

The **base station coordinates** are taken from the base's own RTCM 1005/1006 messages
(`ant2-postype=rinexhead` in `config/config_rtkpost_ntn.conf`), so a base that moves needs no
change here; the base stream must carry one of them at least every few seconds. Set
`ant2-postype=llh` and `ant2-pos1/2/3` instead for a base that does not send them. The same
file selects the constellations (`pos1-navsys`, GPS + Galileo) and frequencies (L1+L2+L5).

#### Inputs
| Port | Protocol | Content |
|---|---|---|
| 5005/udp | RTCM3 | Base station observations (MSM) and station position (1005/1006) |
| 5006/udp | RTCM3 | Rover observations (MSM), 1 Hz |
| 5007/udp | RTCM3 | Navigation messages (1019 GPS, 1045/1046 Galileo ephemerides) |

The streams are told apart by port only. Datagrams can be raw RTCM3 cut at any byte or
base64 text (the satellite link requires ASCII); a 15-digit IMSI prefix followed by `;`, which
the Sateliot UE adds, is stripped.

#### Running

```bash
cp .env.example .env              # fill in DB_PASSWORD
docker compose up -d --build
docker logs -f ntn_server
```

#### Testing without a GNSS feed
`data/` holds 9 minutes of a real capture (`B_*`, `R_*`, `N_*`, recorded on 2026-07-28). The
`launcher` service replays it as three satellite passes (base, ephemerides and rover of each
3-minute file sent at once, then 90 s for the receiver to close and process the window) and
exits. Set `RTCM_START_TIME=2026/07/28 12:00:00` in `.env` (an old capture needs it), and
`DB_UPLOAD=0` to test without the database, then:

```bash
docker compose up -d --build
docker compose run --rm launcher
```

For each pass `logs/server_execution.log` shows the RTK result, the detector's answer and, with
`DB_UPLOAD=0`, what would have reached the database:

```text
Window closed (no data for 60 s): rover 181 epochs, ...
ANOMALY: 181 new rover epochs, 1 window(s) of 120 consecutive epochs sent to the detector, 61 epochs waiting for the next
ANOMALY JSON: {"Predictions": [0], "Probabilities": [0.27478452395176867]}
RTK Result: 139/181 FIXED epochs (76.8%), mode RTK
DB (DB_UPLOAD=0): would insert 181 rows into rtk_positions:
  rover_id  gps_week  gps_tow  latitude  longitude  height  q_flag  sdne  sdeu  sdun  anomaly
  ROVER_01  2429  216948.000  41.341772522  2.165680626  56.4347  2  -0.4054  0.9058  -1.2790  NULL
  ROVER_01  2429  216949.000  41.341772177  2.165680588  56.4927  2  -0.3635  0.7014  -0.9691  NULL
  ROVER_01  2429  216950.000  41.341772373  2.165680587  56.4835  2  -0.3228  0.5923  -0.8108  NULL
    ...
  ROVER_01  2429  217127.000  41.341773786  2.165675696  56.1345  1  -0.0022  0.0038  -0.0048  NULL
  ROVER_01  2429  217128.000  41.341773782  2.165675684  56.1364  1  -0.0022  0.0038  -0.0048  NULL
DB (DB_UPLOAD=0): would set anomaly = 0 on rover ROVER_01, week 2429, gps_tow 216948.000-217067.000
```

The three passes give 139/181, 136/180 and 137/180 FIXED and 4 anomaly windows. The detector's
own log, step by step (parsing, `csa` run, ONNX fallback), is `logs/anomaly/server.log`.
Stopping the base (`docker compose run --rm launcher` with the `B_*` files moved away) shows the
single point fallback (`mode single`, `q_flag` 5). A single file can also be sent by hand, e.g.
the first rover file to a server on another host:

```bash
python tools/generic_simulator_files.py --once --index 1 --dir ./data --prefix R --ip <host> --port 5006
```

#### Validation with the drone dataset
`data/drone/` holds a flight test of 2026-10-16: rover on a DJI Mavic 3, base at Montells
(41.6473109249999°, 2.349313816°, 263.604 m), both u-blox F20P with ANN-MB antennas (GPS
L1/L5, Galileo E1/E5a), two flights of 741 and 440 s at 1 Hz. It was prepared from the raw
captures with `tools/prepare_drone_test.py`: the base sends no position message, so a 1006 with
the surveyed position is inserted every 10 s; the ephemerides were only in the UBX logs and
were converted with RTKLIB `str2str -in file://montells.ubx#ubx -msg "1019,1045,1046" -out
file://montells_nav.rtcm3#rtcm3`; each flight is cut into passes of at most 360 epochs (5 in
all). `data/drone/data_scaler.json` holds the anomaly normalisation statistics for this
receiver. In `.env`:

```text
ROVER_ID=DRONE_01
RTCM_START_TIME=2026/10/16 15:00:00
ANOMALY_SCALER=./data/drone/data_scaler.json
```

```bash
docker compose up -d --build
docker compose run --rm launcher drone
```

| Pass | Rover epochs | RTK FIXED | Anomaly windows |
|---|---|---|---|
| 1 | 247 | 79/246 (32.1 %) | 2 |
| 2 | 247 | 192/246 (78.0 %) | 2 |
| 3 | 247 | 159/246 (64.6 %) | 2 |
| 4 | 220 | 4/219 (1.8 %) | 1 |
| 5 | 220 | 63/219 (28.8 %) | 2 |

1176 positions, 497 FIXED; 9 windows of 120 epochs analysed, all normal (probability
0.07-0.63, threshold 0.93), labelling 1077 rows; the other 99 are the tails of the two flights,
too short for a window, and stay NULL. Each pass is solved on its own, which costs fixes:
solving each flight whole gives 578/741 and 186/434.

This run is stored in the AGRARIAN database as `DRONE_01` (validated on 2026-11-19). The table
is unique on (`rover_id`, `gps_week`, `gps_tow`) and duplicates are skipped, so repeating the
test against the database needs a new `ROVER_ID` (e.g. `DRONE_02`) to see the rows inserted.

#### Outputs
- **`rtk_positions`** in PostgreSQL, one row per epoch (schema in `tools/Init_DB.py`):
  `rover_id`, `gps_week`, `gps_tow` (s, GPS time), `latitude`, `longitude` (deg),
  `height` (m, ellipsoidal), `q_flag` (1 FIXED, 2 FLOAT, 5 SINGLE), `sdne`, `sdeu`, `sdun` (m),
  `anomaly` (1 anomalous, 0 normal, **NULL not analysed**: inserted as NULL and set once the
  run of 120 consecutive 1 s epochs it belongs to is complete; it stays NULL if the epoch is
  not part of such a run or the detector gave no answer).
- **`logs/server_execution.log`**, rotated every 12 h, 4 kept. One `RTK Result: N/M FIXED
  epochs (P%), mode RTK|single` line per window; `tools/analyze_logs_testbed.py` summarises
  them.
- **`output/`** is empty in normal operation: every buffer and intermediate file is deleted
  once processed. While the database is unreachable, a `.pos` and the anomaly labels not yet
  applied (`anomaly_updates.json`) stay there and are uploaded after the next successful
  upload.

#### Tools
| Script | Purpose |
|---|---|
| `tools/generic_simulator_files.py` | RTCM3 simulator for any of the three streams (rover, base or nav, chosen by `--prefix` and `--port`): replays files over UDP (`--once --index N`, or one file every `--interval` s; `--encoding base64`) |
| `tools/replay_passes.sh` | Replays `data/<dir>` as passes and exits (the `launcher` service) |
| `tools/prepare_drone_test.py` | Builds `data/drone/` from the drone captures (inserts the base position, cuts passes) |
| `tools/analyze_logs_testbed.py` | Fix-rate statistics from `server_execution.log` |
| `tools/pos_analyze.py` | Dispersion of the FIXED epochs of a `.pos` |
| `tools/Init_DB.py` | Creates `rtk_positions` (the AGRARIAN database already has it) |

The anomaly detection service can also be tested on its own with the author's client and sample
file (`agriprecise_files/http_client.py`, `data/example_data.rtcm3`); see the docker
integration document above.

#### Acknowledgement
This work was supported by the AGRARIAN project under Horizon Europe call HORIZON-CL6-2023-GOVERNANCE-01-13, Grant 101134128. Views and opinions expressed are those of the authors only and do not necessarily reflect those of the European Union or European Commission.

#### License
Apache 2.0
