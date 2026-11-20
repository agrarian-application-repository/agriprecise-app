import os
import json
import subprocess
import uuid 
import shutil
import py7zr
import logging
import time
import contextvars

import onnxruntime as ort

from logging.handlers import RotatingFileHandler
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd

from pyrtcm import RTCMReader
from pathlib import Path
from typing import Tuple, Dict, Union
from scipy.stats import lognorm


############# File and Folder definition #############
REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = REPO_ROOT / "data"

INFERENCE_DEVICE = "MYRIAD"
CSA_TIMEOUT = 30

INFERENCE_ENGINE_MODE = os.environ.get("INFERENCE_ENGINE_MODE", "hybrid_engine").lower()

SCALER_PATH = EXPERIMENT_DIR / "dataHelpers" / "data_scaler.json"
CALIBRATED_LOGNORM = EXPERIMENT_DIR / "dataHelpers" / "conv_autoencoder_stats_calibrated.json"
ONNX_MODEL_PATH = EXPERIMENT_DIR / "dataHelpers" / "conv_autoencoder.onnx"
UNN_MODEL_PATH = EXPERIMENT_DIR / "unnModels"
MX_JSON_PATH = EXPERIMENT_DIR / "unnHelpers" / "csa_config.json"

UBOTICA_PATH = "/ubotica/bin/csa"
INPUT_DATA_PATH = "/ubotica/data/nbiot_input/" 
OUTPUT_DATA_PATH =  "/ubotica/data/output/"

UNN_HELPERS_DATA_PATH = EXPERIMENT_DIR / "unnHelpers"

# --- LOGGING CONFIGURATION ---
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "server.log"

request_id_var = contextvars.ContextVar("request_id", default="SYSTEM")

class RequestIDFilter(logging.Filter):
    """Injects the current request_id into the log record."""
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True

    
LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | "
    "[%(name)s] | [%(threadName)s] | [ReqID: %(request_id)s] | "
    "[%(funcName)s] | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.addFilter(RequestIDFilter())

file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,
    encoding="utf-8"
)
file_handler.setFormatter(formatter)
file_handler.addFilter(RequestIDFilter())

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler],
    force=True)

logger = logging.getLogger(__name__)


############# Global Variables #############
GLOBAL_ECDF_STATS = None
GLOBAL_SCALER_STATS = None
GLOBAL_THRESHOLD = 0.93
GLOBAL_ONNX_SESSION = None

############# Satellite Configuration #############
CONST_LEAP_SECONDS = {'107', '109', '111'}
BDT2GPS = 14
GPSLeapSeconds = 18
LABELMSM = 2

GNSSMAP = {
    "107": ("GPS", "DF004"), "108": ("GLONASS", "DF034"),
    "109": ("GALILEO", "DF248"), "110": ("SBAS", "DF004"),
    "111": ("QZSS", "DF428"), "112": ("BEIDOU", "DF427"),
    "113": ("NAVIC", "DF546"),
}

fields_of_interest = (
    'CELLPRN', 'CELLSIG', 'DF397', 'DF398', 'DF400', 'DF401', 'DF402', 'DF403', 'DF420'
)

SIG_OF_INTEREST = ('L1', 'L2', 'E1', 'E5A', 'E5B')

CSV_COLUMNS = [
    'Message_Identity', 'Epoch', 'MSM_Message_bit', 'Satellite_Mask', 'Signal_Mask', 'CELLPRN',
    'CELLCONST', 'CELLSIG', 'Seconds_Rough_Range', 'Seconds_Rough_Range_1millisecond',
    'Signal_Fine_PseudoRange', 'Signal_Fine_PhaseRange', 'Signal_LockTime_PhaseRange',
    'Signal_CNR', 'HalfCycle_Ambiguity'
]

GALILEO_SATS = [f"GALILEO_{i}" for i in
                [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 21, 23, 25, 26, 27, 29, 30, 31, 32, 33, 34,
                 36]]
GPS_SATS = [f"GPS_{i}" for i in range(1, 33)]

# Define valid signals per constellation
GALILEO_SIGS = ["E1", "E5A", "E5B"]
GPS_SIGS = ["L1", "L2"]

FREQ_MAP = {'L1': 1575.42e6, 'E1': 1575.42e6,
            'L2': 1227.60e6,
            'L5': 1176.45e6, 'E5A': 1176.45e6,
            'E5B': 1207.14e6}

# Speed of light in meters per millisecond
C_MS = 299792.458
# Speed of light in meters per second (for wavelength calculation)
C_S = 299792458.0

# Generate ONLY the valid Satellite+Signal combinations
SAT_SIG_PAIRS = [f"{sat}_{sig}" for sat in GALILEO_SATS for sig in GALILEO_SIGS] + \
                [f"{sat}_{sig}" for sat in GPS_SATS for sig in GPS_SIGS]

VALID_SAT_SIG_SET = set(SAT_SIG_PAIRS)

FEATURES_LIST = ['Signal_CNR', 'Signal_PseudoRange', 'Signal_CP', 'CMC_raw_meters']

############# Functions #############

def csv_to_custom_numpy_array(_filtered_df: pd.DataFrame) -> np.ndarray:
    """
    Parses a pandas Dataframe and builds a 3D numpy array.
    Dimensions: (Epochs, Features, Sat_Sig_Combinations)
    """
    _filtered_df['Sat_Sig'] = (
        _filtered_df['CELLCONST']
        .str.cat(_filtered_df['CELLPRN'].astype(str), sep='_')
        .str.cat(_filtered_df['CELLSIG'], sep='_')
    )
    valid_mask = _filtered_df['Sat_Sig'].isin(VALID_SAT_SIG_SET)

    df_valid = _filtered_df.loc[valid_mask, ['Epoch', 'Sat_Sig'] + FEATURES_LIST]

    sat_sig_cat = pd.Categorical(df_valid['Sat_Sig'], categories=SAT_SIG_PAIRS)
    sat_sig_coords = sat_sig_cat.codes

    epochs, ep_coords = np.unique(df_valid['Epoch'].values, return_inverse=True)

    shape = (len(epochs), len(FEATURES_LIST), len(SAT_SIG_PAIRS))
    data_array = np.full(shape, np.nan, dtype=np.float32)

    for f_idx, feature_name in enumerate(FEATURES_LIST):
        data_array[ep_coords, f_idx, sat_sig_coords] = df_valid[feature_name].values

    return data_array


def reconstruct_absolute_observables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstructs the absolute Pseudorange (meters) and Carrier Phase (cycles)
    from parsed RTCM 1075 (GPS) and 1095 (Galileo) MSM5 messages.
    """
    df = df.copy()

    tof_pseudo_ms = (
            df['Seconds_Rough_Range'] +
            df['Seconds_Rough_Range_1millisecond'] +
            df['Signal_Fine_PseudoRange']
    )
    df['Signal_PseudoRange'] = tof_pseudo_ms * C_MS

    tof_phase_ms = (
            df['Seconds_Rough_Range'] +
            df['Seconds_Rough_Range_1millisecond'] +
            df['Signal_Fine_PhaseRange']
    )

    phaserange_meters = tof_phase_ms * C_MS

    frequencies = df['CELLSIG'].map(FREQ_MAP)
    wavelengths = C_S / frequencies
    df['Signal_CP'] = phaserange_meters / wavelengths

    if 'Signal_PhaseRange' in df.columns:
        df['Doppler'] = -df['Signal_PhaseRange'] / wavelengths

    cmc_ms = df['Signal_Fine_PseudoRange'] - df['Signal_Fine_PhaseRange']
    df['CMC_raw_meters'] = cmc_ms * C_MS

    return df


def process_msm_messages(msg):
    """Optimized MSM processing to minimize getattr and dynamic strings."""
    msmlist = []
    m_dict = msg.__dict__

    identity = msg.identity
    prefix = identity[:3]

    is_msm7 = identity.endswith('7')

    msm5_to_msm7_field_map = {
        'DF400': 'DF405',  # Fine Pseudorange -> High-res Pseudorange
        'DF401': 'DF406',  # Fine PhaseRange -> High-res PhaseRange
        'DF402': 'DF407',  # Lock Time Indicator -> High-res Lock Time
        'DF403': 'DF408',  # CNR -> High-res CNR
    }

    sat_info = {}
    gmap_attr = GNSSMAP[prefix]
    for j in range(1, msg.NSat + 1):
        prn = m_dict.get(f"PRN_{j:02d}")
        sat_info[prn] = (
            m_dict.get(f"DF397_{j:02d}"),
            m_dict.get(f"DF398_{j:02d}"),
            m_dict.get(f"DF399_{j:02d}")
        )

    raw_time = int(m_dict.get(gmap_attr[1], 0)/1000)
    if prefix in CONST_LEAP_SECONDS:
        epoch = raw_time - GPSLeapSeconds
    elif prefix == '108':
        epoch = raw_time
    else:
        epoch = raw_time + BDT2GPS - GPSLeapSeconds

    df393 = m_dict.get('DF393')
    df394 = m_dict.get('DF394')
    df395 = m_dict.get('DF395')

    base_row = [identity, epoch, df393, df394, df395]

    for i in range(1, msg.NCell + 1):
        cell_prn = m_dict.get(f'CELLPRN_{i:02d}')
        cell_sig = m_dict.get(f'CELLSIG_{i:02d}')
        if cell_sig not in SIG_OF_INTEREST:
            continue
        row = base_row.copy()

        for attr in fields_of_interest:
            if attr == 'DF397':
                row.append(sat_info.get(cell_prn, (None, None))[0])
            elif attr == 'DF398':
                row.append(sat_info.get(cell_prn, (None, None))[1])
            elif attr == 'DF399':
                row.append(sat_info.get(cell_prn, (None, None))[2])
            elif attr == 'CELLPRN':
                row.append(int(cell_prn))
                row.append(gmap_attr[0])
            else:
                target_attr = msm5_to_msm7_field_map.get(attr, attr) if is_msm7 else attr
                row.append(m_dict.get(f"{target_attr}_{i:02d}"))

        msmlist.append(row)
    return msmlist


def parse_rtcm_file(file_path: Path, epochs_per_chunk: int = 120):
    """
    Generator that continuously reads an RTCM3 file and yields a Pandas DataFrame 
    every time it collects `epochs_per_chunk` epochs.
    """
    all_observations = []
    unique_epochs = set()

    # Validate file extension
    if file_path.suffix.lower() not in (".rtcm3", ".bin", ".rtcm"):
        logger.warning(f"File {file_path.name} does not have a valid RTCM extension.")
        return

    try:
        with open(file_path, "rb") as stream:
            rtr = RTCMReader(stream, labelmsm=LABELMSM)
            for _, parsed_data in rtr:
                if not parsed_data:
                    continue

                msg_identity = parsed_data.identity

                if msg_identity != '1005' and (msg_identity.startswith('107') or msg_identity.startswith('109')):
                    processed_rows = process_msm_messages(parsed_data)

                    if not processed_rows:
                        continue

                    # Extract the epoch timestamp from the first row of the processed message.
                    if len(processed_rows)>1 and isinstance(processed_rows[1], dict):
                        epoch_val = processed_rows[1].get('Epoch')
                    else:
                        epoch_val = processed_rows[0][1]

                    # If this is a NEW epoch, and we already have exactly 120 unique epochs,
                    # we yield the chunk BEFORE adding the new epoch to the list.
                    if epoch_val not in unique_epochs and len(unique_epochs) == epochs_per_chunk:
                        yield pd.DataFrame(all_observations, columns=CSV_COLUMNS)
                        
                        # Flush the memory buffers to start collecting the next 120 epochs
                        all_observations = []
                        unique_epochs = set()

                    unique_epochs.add(epoch_val)
                    all_observations.extend(processed_rows)

            # Yield the final chunk if it happens to perfectly end on a 120-epoch boundary
            if all_observations and len(unique_epochs) == epochs_per_chunk:
                logger.info(f"RTCM processing completed | File: {file_path.name} | Epochs: {len(unique_epochs)} | "
                            f"Observations: {len(all_observations)}")
                yield pd.DataFrame(all_observations, columns=CSV_COLUMNS)

    except Exception as e:
        logger.error(f"Error reading RTCM file {file_path.name}: {e}")


def normalize(X_raw: np.ndarray, window_size: int = 120,
              method: str = "min-max", masking_option: bool = True,
              target_dtype: np.dtype = np.float32, random_seed: int = 1714,
              external_stats: dict = None) -> np.ndarray:
    """
    Windows the raw X data to match Y, shuffles them together, splits, and normalizes strictly on Train stats.
    """
    if method not in ["min-max", "standard"]:
        raise ValueError(f"Invalid method '{method}'. Choose 'min-max' or 'standard'.")

    logger.debug(f"Normalizing input data | Shape: {X_raw.shape} | Method: {method} | Window size: {window_size}")
    total_epochs, features, sat_sig = X_raw.shape
    num_groups = total_epochs // window_size

    reshaped_data = X_raw[:num_groups * window_size, :, :].reshape(num_groups, window_size, features, sat_sig)

    X_4d = reshaped_data.transpose(0, 2, 3, 1)

    if method == "min-max":
        train_ranges = external_stats["maxs"] - external_stats["mins"]
        train_ranges[train_ranges == 0] = 1e-6
        normalized = (X_4d - external_stats["mins"]) / train_ranges
    elif method == "standard":
        train_stat_std = external_stats["stds"]
        train_stat_std[train_stat_std == 0] = 1e-6
        normalized = (X_4d - external_stats["means"]) / train_stat_std

    if masking_option:
        mask = (~np.isnan(normalized)).astype(target_dtype)
        normalized = np.nan_to_num(normalized, nan=0.0).astype(target_dtype)
        return np.concatenate([normalized, mask], axis=1)
    else:
        return np.nan_to_num(normalized, nan=0.0).astype(target_dtype)



def load_scaler_dict(file_path: Union[str, Path]) -> Dict:
    """
    Loads a JSON file containing scaler statistics and safely reconstructs
    the original dictionary by converting lists back into NumPy arrays.
    """
    with open(file_path, 'r') as f:
        loaded_dict = json.load(f)

    return {k: np.array(v) if isinstance(v, list) else v for k, v in loaded_dict.items()}


def read_ecdf(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

    
def _build_unn_execution_command(binary_path, mx_json_path) -> list[str]:
    return [str(binary_path), "-j", str(mx_json_path)]

def _run_shell_execution(_command, timeout_sec):
    try:
        result = subprocess.run(
            _command,
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=timeout_sec
        )
        command_output = {
            'command_succeeded': True,
            'return_code': result.returncode,
            'command_stdout': result.stdout,
            'command_stderr': result.stderr,
            'timed_out': False
        }
    except subprocess.TimeoutExpired as exc:
        command_output = {
            'command_succeeded': False,
            'return_code': -1,
            'command_stdout': exc.stdout or "",
            'command_stderr': exc.stderr or "",
            'timed_out': True
        }
    except subprocess.CalledProcessError as exc:
        command_output = {
            'command_succeeded': False,
            'return_code': exc.returncode,
            'command_stdout': exc.stdout or "",
            'command_stderr': exc.stderr or "",
            'timed_out': True
        }
    return command_output


def run_unn_inference(input_window: np.ndarray,
                      threshold: float, baseline_type: str = "ecdf",
                      ecdf_obj=None,
                      lognorm_stats: dict = None,
                      temp_input_dir: Path = None, 
                      temp_output_dir: Path = None, 
                      temp_json_dir: Path = None) -> Tuple[np.ndarray, np.ndarray]:
    
    """Runs production inference with concurrent thread-safe I/O."""


    inference_start = time.perf_counter()

    logger.info(f"Starting hardware inference | Output directory: {temp_output_dir.name}")

    if temp_output_dir.exists():
        shutil.rmtree(temp_output_dir)
        logger.info(f"Deleting previous output directory: {temp_output_dir.name}")
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    input_file = Path(temp_input_dir / 'input_data.bin')
    logger.debug("Preparing inference input file.")
    custom_mx_json_path = temp_json_dir / 'csa_config.json'

    try:

        input_data = input_window.astype(np.float16 if INFERENCE_DEVICE == "MYRIAD" else np.float32)
        input_data.tofile(input_file)

        with open(MX_JSON_PATH, 'r') as f:
            mx_config = json.load(f)
            mx_config['folderStructure']['cnnsAndPcdFolder'] = str(UNN_MODEL_PATH) + '/'

        with open(custom_mx_json_path, 'w') as f:
            json.dump(mx_config, f, indent=4)

        unn_command = _build_unn_execution_command(binary_path=UBOTICA_PATH, 
                                                   mx_json_path=custom_mx_json_path)
        logger.debug(f"Executing command: {' '.join(unn_command)}")
        logger.info(f"Executing CSA hardware inference for temp_dir {temp_output_dir.name}...")
        command_result = _run_shell_execution(unn_command, timeout_sec=CSA_TIMEOUT)

        inference_time = time.perf_counter() - inference_start
      
        if command_result['command_stdout']:
            logger.debug(f"Inference stdout: {command_result['command_stdout']}")

        if command_result['command_stderr']:
            logger.warning(f"Inference stderr: {command_result['command_stderr']}")

        if command_result.get('timed_out'):
            logger.error("CSA hardware inference HUNG and was forcefully killed.")
            return np.array([-1]), np.array([-1])

        timeout_phrase = "The job has not finished"

        if (timeout_phrase in command_result['command_stdout'] or 
            timeout_phrase in command_result['command_stderr']):

            logger.error(f"Inference hardware timed out: '{timeout_phrase}' detected in output.")
            return np.array([-1]), np.array([-1])

        error_keywords = ["error", "failed", "exception", "segmentation fault"]
        out_text = (command_result['command_stdout'] + "\n" + command_result['command_stderr']).lower()
        has_error_text = any(kw in out_text for kw in error_keywords)

        if not command_result['command_succeeded'] or has_error_text:            
            logger.error("Inference Hardware Failed (Non-zero exit code or error text detected)!")
            logger.error(f"Stdout: {command_result['command_stdout']}")
            logger.error(f"Stderr: {command_result['command_stderr']}")
            return None, None

        logger.info(f"Hardware inference completed | Execution time: {inference_time:.3f}s")

        archive_file = temp_output_dir / 'results.7z'

        try:
            with py7zr.SevenZipFile(archive_file, mode='r') as z:
                z.extractall(path=temp_output_dir)
        except Exception as e:
            logger.error(f"Native 7z Extraction Failed: {str(e)}")
            return None, None

        # zip_command = _build_7zip_command(source_path=archive_file, 
        #                                     destination_path=temp_output_dir)
        # unzip_process = _run_shell_execution(zip_command)

        # if not unzip_process['command_succeeded']:
        #     print("[ERROR] 7z Extraction Failed!")
        #     print(f"Stdout: {unzip_process['command_stdout']}")
        #     print(f"Stderr: {unzip_process['command_stderr']}")
        #     return None, None
        
        mxo_file = temp_output_dir / 'output' / 'input_data.mxo'
        try:
            output_data = np.fromfile(mxo_file, 
                                        dtype=np.float16 if INFERENCE_DEVICE == "MYRIAD" else np.float32)
        except:
            logger.error(f"Error loading output data: {str(e)}")
            return None, None
        
        output_data = output_data.astype(np.float32)
        reconstructed_data = output_data.reshape(1, 4, 148, 120)

        diff = reconstructed_data - input_window
        diff_flat = diff.reshape(diff.shape[0], -1)
        raw_errors = np.sum(diff_flat ** 2, axis=1)

        if baseline_type == "ecdf":
            if ecdf_obj is None: raise ValueError("ecdf_obj must be provided.")
            probs = ecdf_obj.cdf.evaluate(raw_errors)
        elif baseline_type == "lognorm":
            if lognorm_stats is None: raise ValueError("lognorm_stats must be provided.")
            probs = lognorm.cdf(raw_errors,
                                s=lognorm_stats['shape'],
                                loc=lognorm_stats['loc'],
                                scale=lognorm_stats['scale'])
        else:
            raise ValueError(f"Unknown baseline_type: {baseline_type}")

        predictions = (probs >= threshold).astype(int)
        return predictions, probs

    except Exception as e:
        logger.exception(f"Error processing output: {e}")
        return None, None


def run_onnx_inference(input_window: np.ndarray,
                       session: ort.InferenceSession,
                       threshold: float, baseline_type: str = "ecdf",
                       ecdf_obj=None,
                       lognorm_stats: dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """Runs production inference using an exported ONNX Autoencoder model."""
    inference_start = time.perf_counter()
    logger.info("Starting ONNX software inference...")

    input_name = session.get_inputs()[0].name
    input_data = input_window.astype(np.float32)

    reconstructed_data = session.run(None, {input_name: input_data})[0]

    diff = reconstructed_data - input_data
    diff_flat = diff.reshape(diff.shape[0], -1)
    raw_errors = np.sum(diff_flat ** 2, axis=1)

    if baseline_type == "ecdf":
        if ecdf_obj is None:
            raise ValueError("ecdf_obj must be provided for ECDF mode.")
        probs = ecdf_obj.cdf.evaluate(raw_errors)
    elif baseline_type == "lognorm":
        if lognorm_stats is None:
            raise ValueError("lognorm_stats dictionary must be provided for Lognorm mode.")
        probs = lognorm.cdf(raw_errors,
                            s=lognorm_stats['shape'],
                            loc=lognorm_stats['loc'],
                            scale=lognorm_stats['scale'])
    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")

    predictions = (probs >= threshold).astype(int)

    inference_time = time.perf_counter() - inference_start
    logger.info(f"ONNX software inference completed | Execution time: {inference_time:.3f}s")
    
    return predictions, probs
   

def get_anomaly_probability(_rtcm_file_path: Path, request_id: str):
    """
    Processes the file via the generator and aggregates results from ALL chunks.
    """

    all_predictions = []
    all_probabilities = []

    temp_input_dir = INPUT_DATA_PATH
    temp_output_dir = OUTPUT_DATA_PATH
    temp_json_dir = UNN_HELPERS_DATA_PATH

    chunk_counter = 0
    # Iterates through the entire file, chunk by chunk
    for parsed_data_df in parse_rtcm_file(_rtcm_file_path, epochs_per_chunk=120):
        if parsed_data_df.empty:
            continue

        chunk_counter += 1

        chunk_start = time.perf_counter()

        logger.info(f"File: {_rtcm_file_path.name} | Starting processing for Chunk {chunk_counter}")

        processed_data_reconstructed = reconstruct_absolute_observables(parsed_data_df)
        data_numpy_array = csv_to_custom_numpy_array(processed_data_reconstructed)
        
        normalized_data = normalize(data_numpy_array, window_size=120, method="standard",
                                    masking_option=False, external_stats=GLOBAL_SCALER_STATS)

        if INFERENCE_ENGINE_MODE == "onnx_engine":
            predictions, anomaly_probability = run_onnx_inference(
                input_window=normalized_data,
                session=GLOBAL_ONNX_SESSION,
                threshold=GLOBAL_THRESHOLD,
                baseline_type='lognorm',
                lognorm_stats=GLOBAL_ECDF_STATS
            )
        else:
            predictions, anomaly_probability = run_unn_inference(
                input_window=normalized_data,
                baseline_type='lognorm',
                threshold=GLOBAL_THRESHOLD,
                lognorm_stats=GLOBAL_ECDF_STATS,
                temp_input_dir=Path(temp_input_dir),
                temp_output_dir=Path(temp_output_dir),
                temp_json_dir=Path(temp_json_dir))

            hw_returned_none = (predictions is None or anomaly_probability is None)
            hw_timed_out = not hw_returned_none and (len(predictions) == 1 and predictions[0] == -1)

            if (hw_returned_none or hw_timed_out) and INFERENCE_ENGINE_MODE == "hybrid_engine":
                logger.warning(f"Chunk {chunk_counter} | CSA inference timed out. Triggering ONNX fallback (hybrid mode)...")
                predictions, anomaly_probability = run_onnx_inference(
                    input_window=normalized_data,
                    session=GLOBAL_ONNX_SESSION,
                    threshold=GLOBAL_THRESHOLD,
                    baseline_type='lognorm',
                    lognorm_stats=GLOBAL_ECDF_STATS
                )

        if predictions is None or anomaly_probability is None:
            logger.warning(f"Chunk {chunk_counter} | Inference failed completely. Skipping chunk...")
            continue

        preds_list = predictions.flatten().tolist()
        probs_list = anomaly_probability.flatten().tolist()

        chunk_time = time.perf_counter() - chunk_start

        if len(preds_list) > 1 or preds_list[0] != -1:
            logger.info(f"Chunk {chunk_counter} Results | Total predictions: {len(preds_list)} | "
                        f"Anomalies: {sum(preds_list)} | Normal: {len(preds_list) - sum(preds_list)}")
            # logger.info(f"Chunk {chunk_counter} Probability statistics | Min: {np.min(anomaly_probability):.4f} | "
            #             f"Max: {np.max(anomaly_probability):.4f} | Mean: {np.mean(anomaly_probability):.4f}")
            # logger.info(f"Chunk {chunk_counter} Results | Predictions: {preds_list}")
            logger.info(f"Chunk {chunk_counter} Results | Probabilities: {[round(p, 4) for p in probs_list]}")
        else:
            logger.warning(f"Chunk {chunk_counter} Results | Timeout (-1) array passed downstream.")

        logger.info(f"Chunk {chunk_counter} completed | Processing time: {chunk_time:.3f}s")

        # Aggregate lists for the final JSON payload        
        all_predictions.extend(preds_list)
        all_probabilities.extend(probs_list)


    return all_predictions, all_probabilities


def load_server_globals():
    """Loads the global statistics required for the ML model into memory."""
    
    global GLOBAL_ECDF_STATS, GLOBAL_SCALER_STATS, GLOBAL_THRESHOLD, GLOBAL_ONNX_SESSION
    
    logger.info("Loading server global statistics...")

    try:
        GLOBAL_ECDF_STATS = read_ecdf(CALIBRATED_LOGNORM)
        GLOBAL_SCALER_STATS = load_scaler_dict(SCALER_PATH)
        GLOBAL_THRESHOLD = 0.93

        logger.info(f"Global statistics loaded | Threshold: {GLOBAL_THRESHOLD} | Scaler: {SCALER_PATH.name}")

        if INFERENCE_ENGINE_MODE in ["onnx_engine", "hybrid_engine"]:
            logger.info(f"Mode is '{INFERENCE_ENGINE_MODE}'. Loading ONNX model from {ONNX_MODEL_PATH.name}...")
            GLOBAL_ONNX_SESSION = ort.InferenceSession(str(ONNX_MODEL_PATH))
            logger.info("ONNX model loaded successfully.")

    except Exception:
        logger.exception("Failed to load global statistics or ONNX model")
        raise


def validate_rtcm_file(filename: str) -> Path:
    """
    Validates that the requested RTCM file exists inside EXPERIMENT_DIR.
    """

    requested_path = (EXPERIMENT_DIR / filename).resolve()

    try:
        requested_path.relative_to(EXPERIMENT_DIR.resolve())
    except ValueError:
        raise ValueError("Invalid file path.")

    if requested_path.suffix.lower() not in (".rtcm3", ".bin", ".rtcm"):
        raise ValueError("Invalid RTCM file extension.")

    if not requested_path.is_file():
        raise FileNotFoundError("RTCM file not found.")

    return requested_path



class SimpleHandler(BaseHTTPRequestHandler):
    def do_POST(self):

        request_start = time.perf_counter()
        request_id = str(uuid.uuid4())

        token = request_id_var.set(request_id)
      
        try: 
            client_ip = self.client_address[0]
            logger.info(f"Request received | ID: {request_id} | Client: {client_ip} | Endpoint: {self.path}")

            if self.path == '/anomaly_detection':
                try:
                    content_length = int(self.headers.get('Content-Length', 0))
                    if content_length == 0:
                        logger.error("Missing or zero Content-Length header")
                        self.send_error(400, "Missing or zero Content-Length header")
                        return
                    post_data = self.rfile.read(content_length)
                    
                    dades_rebudes = json.loads(post_data.decode('utf-8'))
                    ruta_fitxer = dades_rebudes.get('file')

                    if not ruta_fitxer:
                        logger.warning("Missing 'file' key in JSON payload.")
                        self.send_error(400, "Missing 'file' key in JSON payload.")
                        return

                    try:
                        rtcm_file_path = validate_rtcm_file(ruta_fitxer)

                    except ValueError:
                        logger.warning("Request rejected: invalid file path or extension.")
                        self.send_error(400, "Invalid RTCM file.")
                        return

                    except FileNotFoundError:
                        logger.warning("Request rejected: requested RTCM file not found.")
                        self.send_error(404, "RTCM file not found.")
                        return

                    logger.info(f"Processing RTCM file: {rtcm_file_path.name}")
                    
                    if not rtcm_file_path.exists():
                        logger.error(f"File {ruta_fitxer} not found on server.")
                        self.send_error(404, f"File {ruta_fitxer} not found on server.")
                        return
                    
                    logger.info("Starting anomaly probability processing...")
                    predictions, anomaly_probabilities = get_anomaly_probability(_rtcm_file_path=rtcm_file_path, 
                                                                                request_id=request_id)
                    
                    resultat = {
                        'Predictions': predictions,
                        'Probabilities': anomaly_probabilities}
                
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    self.wfile.write(json.dumps(resultat).encode('utf-8'))

                    request_time = time.perf_counter() - request_start
                    
                    logger.info(f"Request completed | ID: {request_id} | Status: 200 | Execution time: {request_time:.3f}s")

                except Exception as e:
                    logger.exception(f"Internal Server Error processing request | Request failed | ID: {request_id}")
                    self.send_error(500, "Internal Server Error. Please contact the server administrator.")
                
            else:
                logger.warning(f"404 Not Found: Requested path {self.path}")
                self.send_response(404)
                self.end_headers()

        finally:
            request_id_var.reset(token)


if __name__ == '__main__':
    
    logger.info("========================================")
    logger.info("Starting GNSS Anomaly Detection Server")
    logger.info(f"Inference engine: {INFERENCE_ENGINE_MODE}")
    logger.info(f"Listening address: 0.0.0.0:8000")
    logger.info("========================================")

    load_server_globals()

    logger.info("Initializing the server completed.")
    
    server_address = ('0.0.0.0', 8000)
    httpd = ThreadingHTTPServer(server_address, SimpleHandler)
    logger.info("Listening to port 8000...")
    httpd.serve_forever()