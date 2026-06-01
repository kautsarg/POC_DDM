import numpy as np
import struct
from pathlib import Path
import re
import matplotlib.pyplot as plt

import functions

# v05/v06 readout tail: first uint16 is often 11 (field count after sentinel), not A_total.
V05_TAIL_SENTINEL = 11


def binary_file_read(file: Path) -> list:
    """ Unpacks .bin file and returns the data in a list

    Parameters
    ----------
    file : Path
        Path of the .bin file name
    Returns
    -------
    list
        data_list : list of all the data in the input binary file
    """
    fw = open(file, 'rb')
    data_byte = fw.read()
    data_len = len(data_byte) >> 1
    data_list = []
    for n in range(0, data_len):
        (data, ) = struct.unpack('>H', data_byte[2*n:2*(n+1)])
        data_list.append(data)
    return data_list


def list_to_numpy_readout(
    data_list: list,
    frame_start: int = 0,
    nrows: int = 290,
    ncols: int = 204,
    n_a_type="v04",
    return_meta: bool = False,
) -> (np.ndarray, np.ndarray):
    """
    Converts list of the data from the .bin readout file to a numpy array

    Parameters
    ----------
    data_list : list
        list of input data to be converted into numpy array
    frame_start : int
        first frame to be considered
    nrows : int
        Optional. Number of rows in the chip; default 290 for the Titan chip. Change to 78 for Titanicks
    ncols : int
        Optional. Number of columns in the chip; default 204 for the Titan chip. Change to 56 for Titanicks

    Returns
    -------
    tuple
        (time_vect, frame_3d) where
        - time_vect : np.ndarray
            1D array of the sampling times
        - frame_3d : np.ndarray
            3D array of sensor array outputs. Has shape NxMxT, where NxM is (number of rows x number of columns), T is time
    """
    # Pixels per (single) frame
    P = nrows * ncols  # P=number of pixels per ref-frame

    # v06 uses the same on-disk readout format as v05 (firmware "version=5" tail);
    # the difference is purely in how the host pipeline interprets temperature
    # (see titan_load_and_preprocessing). Treat them identically here.
    if n_a_type == "v05" or n_a_type == "v06":
        # v05/v06 writer serializes:
        #   pixels for all refs (concatenated): len = P * n_refs
        #   then a parameter block that starts with a count, followed by:
        #     timestamp + vref_array + [chem_avg, temp_avg, temp_avg_Vs_shifted, adc, tam, tch, n_wells, version]
        #
        # IMPORTANT: different writers use different conventions for the first count value:
        # - some store A = total param-block length INCLUDING the count word itself
        # - others store A = number of fields AFTER the count word (i.e. excludes itself)
        #
        # We handle both by inferring a consistent record length from the file.
        #
        # We infer n_refs by scanning small candidates and looking for self-consistency:
        # data_list[P*n_refs] should equal 10+n_refs and total file length divisible by (P*n_refs + A).
        if len(data_list) < P + 2:
            raise Exception("RAISED list_to_numpy_readout: v05 file too short")

        inferred_n_refs = None
        inferred_A_total = None  # total parameter-block length INCLUDING the count word
        inferred_A_after = None  # parameter count value as stored in-file (fields AFTER the count, if that convention is used)
        # Typical ref counts are small; keep scan bounded for performance.
        # We do NOT require A == 10 + n_refs, because some writers keep A fixed
        # or use a different convention. We only require the file to be
        # self-consistent with a repeated frame record of length (P*n_refs + A).
        for cand_n_refs in range(1, 33):
            idx_A = P * cand_n_refs
            if idx_A >= len(data_list):
                break

            cand_A_raw = int(data_list[idx_A])
            # Raw count must be plausible.
            # Minimum fields after the count: timestamp (1) + 8 fixed fields = 9.
            # If raw is "total including itself" then min is 10. We accept either.
            if cand_A_raw < 9 or cand_A_raw > 512:
                continue

            # Try both conventions:
            # 1) raw includes itself => A_total = raw
            # 2) raw excludes itself => A_total = raw + 1
            for A_total, A_after in ((cand_A_raw, None), (cand_A_raw + 1, cand_A_raw)):
                cand_N = (P * cand_n_refs) + A_total
                if cand_N <= 0:
                    continue
                if len(data_list) % cand_N != 0:
                    continue
                # Extra sanity: timestamp field should exist (count + timestamp)
                if idx_A + 1 >= len(data_list):
                    continue
                inferred_n_refs = cand_n_refs
                inferred_A_total = A_total
                inferred_A_after = A_after
                break

            # Some firmware stores a count word that is neither A_total nor (A_total-1)
            # (e.g. serialised block is 13 uint16s but the first word is 11).  Resolve
            # A_total by scanning a small band and requiring an even file split.
            if inferred_n_refs is None:
                a_lo = max(9, cand_A_raw - 2)
                a_hi = min(512, cand_A_raw + 12)
                for A_total in range(a_lo, a_hi + 1):
                    if A_total in (cand_A_raw, cand_A_raw + 1):
                        continue
                    cand_N = (P * cand_n_refs) + A_total
                    if cand_N <= 0 or len(data_list) % cand_N != 0:
                        continue
                    if idx_A + 1 >= len(data_list):
                        continue
                    inferred_n_refs = cand_n_refs
                    inferred_A_total = A_total
                    # Use A-1 for meta (fields after count); stored count is nonstandard.
                    inferred_A_after = None
                    break

            if inferred_n_refs is not None:
                break

        if inferred_n_refs is None:
            raise Exception(
                "RAISED list_to_numpy_readout: Could not infer v05 n_refs/A from file. "
                "Expected A at offset P*n_refs and file length divisible by (P*n_refs + A)."
            )

        n_refs = inferred_n_refs
        A = inferred_A_total
        P_frame = P * n_refs
        N = P_frame + A

    elif n_a_type == "v04" or n_a_type == "v03" or n_a_type == "v02":
        # Legacy format: single pixel frame (P) then parameter block length A (including itself)
        A = int(data_list[P])
        if len(data_list) % (P + A) != 0:
            raise "RAISED list_to_numpy_readout: Unexpected number of data points in the binary file"
        N = P + A
    elif n_a_type == "v01":
        if len(data_list) == 0:
            raise Exception("RAISED list_to_numpy_readout: Empty readout file")
        elif len(data_list) % (P+8) == 0:
            A = 8
        elif len(data_list) % (P+7) == 0:
            A = 7
        else:
            raise "RAISED list_to_numpy_readout: Could not find the number of additional parameters"
        N = P + A
    else:
        raise f"RAISED list_to_numpy_readout: Could not find conversion for n_a_type = {n_a_type}"

    frame_end = len(data_list)//N
    n_time = frame_end - frame_start
    frame_3d = np.zeros((nrows, ncols, n_time))
    time_vect = np.zeros(n_time)

    meta = None
    if return_meta:
        meta = {
            "n_params": int(A),
            "timestamp_raw": np.zeros(n_time),
            "vref_array": [],
            "chem_avg": np.zeros(n_time),
            "temp_avg": np.zeros(n_time),
            "temp_avg_Vs_shifted": np.zeros(n_time),
            "adc": np.zeros(n_time),
            "tam": np.zeros(n_time),
            "tch": np.zeros(n_time),
            "n_wells": np.zeros(n_time),
            "version": np.zeros(n_time),
        }

    # For v05/v06, we expose a 4D tensor (rows, cols, n_refs, time)
    frame_4d = None
    if n_a_type == "v05" or n_a_type == "v06":
        frame_4d = np.zeros((nrows, ncols, n_refs, n_time))

    for i, n in enumerate(range(frame_start, frame_end)):
        pack = data_list[n * N:(n + 1) * N]
        if n_a_type == "v05" or n_a_type == "v06":
            # Pixels: P_frame = P * n_refs
            pix = pack[:P_frame]
            for ref_i in range(n_refs):
                start = ref_i * P
                stop = (ref_i + 1) * P
                frame_2d = np.array(pix[start:stop]).reshape(nrows, ncols, order='F')
                frame_4d[:, :, ref_i, i] = frame_2d
            # A and timestamp live after the pixel block
            # pack[P_frame] is A, pack[P_frame+1] is timestamp
            ts_raw = pack[P_frame + 1]
            # Use the raw timestamp value directly (no extra 1/10 scaling).
            time_vect[i] = ts_raw
            if return_meta:
                meta["timestamp_raw"][i] = ts_raw

                # vref_array length derived from the stored parameter count.
                # If the writer uses "raw excludes itself", we saved it as inferred_A_after.
                # When A_total was inferred by scan (inferred_A_after is None), prefer the
                # on-frame count word when it is the v05 sentinel (11) or otherwise plausible;
                # using A_total-1 mis-counts optional PID/extra tail words as vref slots and
                # shifts chem/temp/temp_avg_Vs_shifted (shows up as a flat temperature trace).
                count_word = int(pack[P_frame])
                if inferred_A_after is not None:
                    n_fields_after_count = int(inferred_A_after)
                elif count_word == V05_TAIL_SENTINEL or (
                    9 <= count_word <= int(A) - 1 and count_word != int(A) - 1
                ):
                    n_fields_after_count = count_word
                else:
                    n_fields_after_count = int(A) - 1

                # Layout after count:
                #   [timestamp] + vref_array + 8 fixed fields
                # => len(vref_array) = n_fields_after_count - 9
                n_vref = int(n_fields_after_count) - 9
                if n_vref < 0:
                    raise Exception(
                        f"RAISED list_to_numpy_readout: v05 invalid parameter count. "
                        f"fields_after_count={n_fields_after_count} implies n_vref={n_vref}"
                    )
                idx = P_frame + 2
                vref_arr = [int(x) for x in pack[idx: idx + n_vref]]
                meta["vref_array"].append(vref_arr)
                idx = idx + n_vref

                # Remaining fixed 8 fields
                if idx + 8 > P_frame + A:
                    raise Exception(
                        f"RAISED list_to_numpy_readout: v05 parameter block too short for A={A} (need vref_array + 8 fields)"
                    )
                meta["chem_avg"][i] = pack[idx + 0]
                meta["temp_avg"][i] = pack[idx + 1]
                meta["temp_avg_Vs_shifted"][i] = pack[idx + 2]
                meta["adc"][i] = pack[idx + 3]
                meta["tam"][i] = pack[idx + 4]
                meta["tch"][i] = pack[idx + 5]
                meta["n_wells"][i] = pack[idx + 6]
                meta["version"][i] = pack[idx + 7]

        else:
            frame_1d = np.array(pack[:P])
            frame_2d = frame_1d.reshape(nrows, ncols, order='F')
            frame_3d[:, :, i] = frame_2d

        if n_a_type == "v04" or n_a_type == "v03" or n_a_type == "v02":
            # Convention: timestamp stored in the first parameter slot after A.
            ts_raw = pack[P + 1]
            # Use the raw timestamp value directly (no extra 1/10 scaling).
            time_vect[i] = ts_raw
            if return_meta:
                meta["timestamp_raw"][i] = ts_raw
        else:
            if n_a_type == "v01":
                # Use the raw timestamp value directly (no extra 1/10 scaling).
                time_vect[i] = pack[P]

    if return_meta:
        if n_a_type == "v05" or n_a_type == "v06":
            return time_vect, frame_4d, meta
        return time_vect, frame_3d, meta
    if n_a_type == "v05" or n_a_type == "v06":
        return time_vect, frame_4d
    return time_vect, frame_3d


def list_to_numpy_calib(data_list: list, nrows: int = 290, ncols: int = 204) -> np.ndarray:
    """
    Converts the input data_list into a 3d representation of the calibration data

    Parameters
    ----------
    data_list: list
        list of input data imported from the calibration binary file
    nrows : int
        Optional. Number of rows in the chip; default 290 for the Titan chip. Change to 78 for Titanicks
    ncols : int
        Optional. Number of columns in the chip; default 204 for the Titan chip. Change to 56 for Titanicks

    Returns
    -------
    ndarray
        frame_3d: A 3D numpy array of shape (ROWS, COLS, n_frames) of calibration data
    """
    # Number of parameters per frame
    N = nrows * ncols  # number of pixels
    # there are no additional parameters

    n_frames = len(data_list)//N
    frame_3d = np.zeros((nrows, ncols, n_frames))
    for i, n in enumerate(range(n_frames)):
        pack = data_list[n * N:(n + 1) * N]
        frame_1d = np.array(pack)
        frame_2d = frame_1d.reshape(nrows, ncols, order='F')
        frame_3d[:, :, i] = frame_2d
    return frame_3d


TTN_DVREF_MULTI_TAIL_MAGIC = 0x5A


def parse_ttn_dvref_multi_payload(payload: bytes) -> dict:
    """
    Decode ``ttn_sweep_search_vref`` binary (firmware ``TTN_Build_HostPayload_dVref_Multi``).

    Matches ``parse_ttn_dvref_multi_payload()`` in
    ``Lacewing_Integrated_new_temp_read/Lacewing_UI_v05/Lacewing_Thread.py``.

    Layout:
      - ``n`` (u8), ``n`` × Vref DAC (u16 BE)
      - optional tail when next byte is ``TTN_DVREF_MULTI_TAIL_MAGIC`` (0x5A):
        coarse sweep + model summary fields
    """
    empty: dict = {"vrefs": [], "coarse": None, "model": None}
    if payload is None or len(payload) < 1:
        return empty

    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(
            f"parse_ttn_dvref_multi_payload: expected bytes/bytearray, got {type(payload)}"
        )

    b = bytes(payload)
    n = int(b[0])
    need_v = 1 + 2 * n
    if len(b) < need_v:
        raise ValueError(
            f"parse_ttn_dvref_multi_payload: need {need_v} bytes for n={n}, got {len(b)}"
        )

    off = 1
    vrefs: list[int] = []
    for _ in range(n):
        vrefs.append(int.from_bytes(b[off : off + 2], byteorder="big", signed=False))
        off += 2

    if off >= len(b) or b[off] != TTN_DVREF_MULTI_TAIL_MAGIC:
        return {"vrefs": vrefs, "coarse": None, "model": None}

    off += 1
    tail_need = 1 + 1 + 1 + 4 + 4
    if len(b) < off + tail_need:
        raise ValueError("parse_ttn_dvref_multi_payload: truncated after magic")
    coarse_n = int(b[off])
    off += 1
    model_pass_fail = int(b[off])
    off += 1
    model_vrefs_to_pass = int(b[off])
    off += 1
    model_total = int.from_bytes(b[off : off + 4], byteorder="big", signed=True)
    off += 4
    coarse_thr = int.from_bytes(b[off : off + 4], byteorder="big", signed=True)
    off += 4

    rec_len = 2 + 2 + 4 + 4
    need_coarse = coarse_n * rec_len
    if len(b) < off + need_coarse:
        raise ValueError(
            f"parse_ttn_dvref_multi_payload: need {need_coarse} coarse bytes, have {len(b) - off}"
        )

    cv: list[int] = []
    cc: list[int] = []
    cd: list[int] = []
    cdd: list[int] = []
    for _ in range(coarse_n):
        cv.append(int.from_bytes(b[off : off + 2], byteorder="big", signed=False))
        off += 2
        cc.append(int.from_bytes(b[off : off + 2], byteorder="big", signed=False))
        off += 2
        cd.append(int.from_bytes(b[off : off + 4], byteorder="big", signed=True))
        off += 4
        cdd.append(int.from_bytes(b[off : off + 4], byteorder="big", signed=True))
        off += 4

    return {
        "vrefs": vrefs,
        "coarse": {"vref": cv, "cnt_fast": cc, "d_cnt_fast": cd, "dd_cnt_fast": cdd},
        "model": {
            "pass_fail": model_pass_fail,
            "vrefs_to_pass": model_vrefs_to_pass,
            "total_dcnt_x16": model_total,
            "coarse_dcnt_x16_threshold": coarse_thr,
        },
    }


def list_to_numpy_vref(data_list: list) -> (int, np.ndarray):
    """
    Converts a vref sweep binary payload into a count + vref vector.

    Expected writer format (as provided):
        binary_file_write(..., [len(vref_list)] + vref_list)

    Where the first stored value is the number of vref entries that follow.

    Parameters
    ----------
    data_list : list
        Raw unpacked values from :func:`binary_file_read`.

    Returns
    -------
    tuple
        (n_vrefs, vref_vect) where
        - n_vrefs : int
            number of vref entries stored
        - vref_vect : np.ndarray
            1D array of length n_vrefs containing the stored vref values
    """
    if data_list is None or len(data_list) == 0:
        raise Exception("RAISED list_to_numpy_vref: Empty vref file")

    n_vrefs = int(data_list[0])
    if n_vrefs < 0:
        raise Exception(f"RAISED list_to_numpy_vref: Invalid n_vrefs={n_vrefs}")

    expected_len = 1 + n_vrefs
    if len(data_list) < expected_len:
        raise Exception(
            f"RAISED list_to_numpy_vref: Unexpected payload length. "
            f"Expected at least {expected_len} values (1 + n_vrefs), got {len(data_list)}"
        )

    # If writer appended extra metadata in the future, ignore the tail safely.
    vref_vect = np.array(data_list[1:expected_len])
    return n_vrefs, vref_vect


def load_vref_sweep(exp_path: Path, return_telemetry: bool = False):
    """
    Load calibrated Vref DAC codes for an experiment folder.

    Prefers ``*_dvref_multi_payload.bin`` (Lacewing v05+ ``TTN_Calibrate_Vref``) and
    falls back to legacy ``*_vref_sweep.bin`` / ``*_vref_swepp.bin``.

    Parameters
    ----------
    exp_path : Path
        Experiment directory containing ``.bin`` files.
    return_telemetry : bool
        If True and a dvref multi payload is present, also return the parsed
        ``parse_ttn_dvref_multi_payload`` dict (coarse sweep + model fields).

    Returns
    -------
    tuple
        ``(n_vrefs, vref_vect)`` or ``(n_vrefs, vref_vect, telemetry)`` when
        ``return_telemetry=True``.
    """
    exp_path = Path(exp_path)
    vref_path = find_most_recent_vref(exp_path)
    tele = None

    if "dvref_multi_payload" in vref_path.name.lower():
        with open(vref_path, "rb") as fp:
            payload = fp.read()
        tele = parse_ttn_dvref_multi_payload(payload)
        vrefs = tele["vrefs"]
        n_vrefs = len(vrefs)
        vref_vect = np.array(vrefs, dtype=float)
    else:
        vref_data = binary_file_read(vref_path)
        n_vrefs, vref_vect = list_to_numpy_vref(vref_data)

    if return_telemetry:
        return n_vrefs, vref_vect, tele
    return n_vrefs, vref_vect


def list_to_numpy_gain(
    data_list: list,
    nrows: int = 290,
    ncols: int = 204,
    n_a_type="v04",
    exp_path=None,
    exp_path_readout=None,
    plt_show=False,
    n_refs: int | None = None,
    return_4d: bool = False,
):
    """
    Converts the input data_list into a 3d representation of the gain data

    Parameters
    ----------
    data_list: list
        list of input data imported from the gain binary file
    nrows : int
        Optional. Number of rows in the chip; default 290 for the Titan chip. Change to 78 for Titanicks
    ncols : int
        Optional. Number of columns in the chip; default 204 for the Titan chip. Change to 56 for Titanicks

    Returns
    -------
    ndarray
        frame_3d: A 3D numpy array of shape (ROWS, COLS, n_frames) of gain data
    """
    # Pixels per (single) ref-frame
    P = nrows * ncols  # P=number of pixels

    if n_a_type == "v05" or n_a_type == "v06":
        # v05/v06 gain writer (as provided) flattens ALL ref frames sequentially into the pixel stream.
        # Some writers may append a parameter footer; others may store pixels only.
        if data_list is None or len(data_list) == 0:
            raise Exception("RAISED list_to_numpy_gain: Empty gain file")

        if n_refs is None:
            raise Exception(
                "RAISED list_to_numpy_gain: v05 requires n_refs (e.g. from vref sweep file: n_vrefs) to parse gain pixels reliably"
            )
        if n_refs <= 0:
            raise Exception(f"RAISED list_to_numpy_gain: v05 invalid n_refs={n_refs}")

        P_frame = P * int(n_refs)
        if len(data_list) < P_frame:
            raise Exception(
                f"RAISED list_to_numpy_gain: v05 file too short. Need at least {P_frame} pixels, got {len(data_list)}"
            )

        # Detect optional footer: first word after pixels is often a count.
        A_total = 0
        idx_A = P_frame
        if idx_A < len(data_list):
            cand_A_raw = int(data_list[idx_A])
            # Try both conventions: raw includes itself, or raw excludes itself
            for cand_total in (cand_A_raw, cand_A_raw + 1):
                if 2 <= cand_total <= 512 and len(data_list) % (P_frame + cand_total) == 0:
                    A_total = cand_total
                    break

        if A_total > 0:
            N = P_frame + A_total
            n_time = len(data_list) // N
        else:
            # Pixels-only records (fixed length)
            if len(data_list) % P_frame != 0:
                raise Exception(
                    "RAISED list_to_numpy_gain: v05 pixels-only format expected file length divisible by (P*n_refs)"
                )
            N = P_frame
            n_time = len(data_list) // N

        # Parse into 4D then (optionally) collapse to 3D for downstream compatibility
        frame_4d = np.zeros((nrows, ncols, int(n_refs), n_time))
        for i in range(n_time):
            pack = data_list[i * N:(i + 1) * N]
            pix = pack[:P_frame]
            for ref_i in range(int(n_refs)):
                start = ref_i * P
                stop = (ref_i + 1) * P
                frame_2d = np.array(pix[start:stop]).reshape(nrows, ncols, order='F')
                frame_4d[:, :, ref_i, i] = frame_2d

        vref = None

        if plt_show:
            fig, ax = plt.subplots()
            # Plot the last-ref slice by default
            ax.plot(frame_4d[:, :, -1, :].reshape(-1, frame_4d.shape[3]).T)
            plt.show()

        # For v05/v06 we always return the full 4D tensor:
        #   (nrows, ncols, n_refs, n_time)
        # If callers need a legacy 3D view, they can slice e.g. frame_4d[:, :, -1, :].
        return frame_4d, vref

    if n_a_type == "v04":
        A = data_list[P]
        vref = data_list[P+1]
        if len(data_list) % (P+A) != 0:
            raise "RAISED list_to_numpy_gain: Unexpected number of data points in the binary file"
    elif n_a_type == "v03" or n_a_type == "v02" or n_a_type == "v01":
        if len(data_list) == 0:
            raise "RAISED list_to_numpy_gain: Empty readout file"
        elif len(data_list) % (P + 4) == 0:
            A = 4
        elif len(data_list) % (P + 6) == 0:
            A = 6
        elif len(data_list) % (P + 10) == 0:
            A = 10
        else:
            raise "RAISED list_to_numpy_gain: Could not find the number of additional parameters"

        exp_path_log = find_corresp_file(exp_path, exp_path_readout, 'log')
        vref = extract_last_vref(exp_path_log)
    else: raise f"RAISED list_to_numpy_gain: Could not find list_to_numpy_gain conversion for n_a_type = {n_a_type}"
    N = P + A

    n_time = len(data_list)//N
    frame_3d = np.zeros((nrows, ncols, n_time))
    for i in range(n_time):
        pack = data_list[i * N:(i + 1) * N]
        frame_1d = np.array(pack[:P])
        frame_2d = frame_1d.reshape(nrows, ncols, order='F')
        frame_3d[:, :, i] = frame_2d

    # Check that the number of frames in the files is as expected
    if n_a_type == "v04" or n_a_type == "v03":
        if frame_3d.shape[2] != 8:
            raise f"RAISED list_to_numpy_gain: Number of frames in the gain file: frame_3d_gain has shape {frame_3d.shape}, expected (N, M, 8)"
    else:
        if frame_3d.shape[2] != 4:
            raise f"RAISED list_to_numpy_gain: Number of frames in the gain file: frame_3d_gain has shape {frame_3d.shape}, expected (N, M, 4)"

    if plt_show:
        fig, ax = plt.subplots()
        ax.plot(frame_3d.reshape(-1, frame_3d.shape[2]).T)
        plt.show()
    return frame_3d, vref


def unwrap_time_vect(timevect: np.ndarray) -> np.ndarray:
    """
    Unwraps a time vector to correct for discontinuities.
    This function takes a numpy array of time values that may contain
    samples where the time count resents. It corrects these discontinuities to create a
    smoothly increasing time vector.

    Parameters
    ----------
    timevect: np.ndarray
        A one-dimensional numpy array of time values that may contain discontinuities.

    Returns
    -------
    np.ndarray
        The corrected time vector with discontinuities removed.
    """
    unwrap_indeces = np.argwhere(np.diff(timevect) < 0)
    tmp_unwrap_compensation = 0
    for unwrap_idx in unwrap_indeces:
        timevect[int(unwrap_idx)+1:] = timevect[int(unwrap_idx)+1:] + timevect[int(unwrap_idx)] - tmp_unwrap_compensation
        tmp_unwrap_compensation = timevect[int(unwrap_idx)]
    return timevect


def find_corresp_file(exp_path: Path, exp_path_readout: Path, type_of_file: str) -> Path:
    """
    Finds the Path of the closest .bin calibration file preceding the readout exp_path_readout in exp_path.

    Parameters
    ----------
    exp_path: Path
        Path of the folder containing the experiment data
    exp_path_readout: Path
        Path of the .bin readout file
    type_of_file: str
        "calibration" or "idx_active"

    Returns
    -------
    Path
        path_file, the Path of the closest calibration/active pixels bin file preceding the readout
    """
    exp_path_readout_str = str(exp_path_readout)
    readout_time = int(exp_path_readout_str[exp_path_readout_str.rfind('T') + 1: exp_path_readout_str.rfind('T') + 5])

    # FOR THE LOG FILE, LOOK FOR FILE AFTER READOUT
    if type_of_file == 'log':
        potential_filename_list = [i for i in exp_path.glob("*lacewing_log*.txt")]

        smallest_to_readout_time = 1000
        for potential_file in potential_filename_list:
            potential_file_str = str(potential_file)
            calib_time = int(potential_file_str[potential_file_str.rfind('T') + 1: potential_file_str.rfind('T') + 5])
            readout_to_file_time = functions.find_time_difference(readout_time, calib_time)
            if 0 <= readout_to_file_time < smallest_to_readout_time:
                path_file = potential_file
                smallest_to_readout_time = readout_to_file_time
        if smallest_to_readout_time == 1000:
            raise f'RAISED: Matching file {type_of_file} before readout not found.'
        else:
            return path_file

    # FOR OTHERS, LOOK FOR FILE BEFORE
    if type_of_file == 'calibration':
        potential_filename_list = [i for i in exp_path.glob("*alib*.bin")]
    elif type_of_file == 'idx_active':
        potential_filename_list = [i for i in exp_path.glob("*find_active*.bin")]
    elif type_of_file == 'gain':
        potential_filename_list = [i for i in exp_path.glob("*gain*.bin")]
    else:
        raise f"RAISED: Code for type_of_file {type_of_file} not yet implemented"

    smallest_to_readout_time = 1000
    for potential_file in potential_filename_list:
        potential_file_str = str(potential_file)
        calib_time = int(potential_file_str[potential_file_str.rfind('T') + 1: potential_file_str.rfind('T') + 5])
        file_to_readout_time = functions.find_time_difference(calib_time, readout_time)
        if 0 <= file_to_readout_time < smallest_to_readout_time:
            path_file = potential_file
            smallest_to_readout_time = file_to_readout_time
    if smallest_to_readout_time == 1000:
        raise f'RAISED: Matching file {type_of_file} before readout not found.'

    # print(f'DEBUG: Matched files are readout {exp_path_readout}, {path_file}')
    return path_file


def extract_last_vref(file_path):
    # List to store the extracted numerical values
    vref_values = []

    # Open the file and read lines
    with open(file_path, 'r') as file:
        for line in file:
            # Look for the pattern "vref: ~ <number>"
            match = re.search(r'vref:\s*~\s*([-+]?\d*\.?\d+)', line, re.IGNORECASE)
            if match:
                # Convert the matched value to a number (float or int)
                value = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
                vref_values.append(value)

    if len(vref_values) == 0:
        raise "RAISED: The Vref value could not be found in the Lacewing log file"

    return vref_values[-1]


def find_most_recent_readout(exp_path: Path):
    """
    Finds the Path of the most recent readout file in the exp_path folder

    Parameters
    ----------
    exp_path: Path
        Path of the folder containing the experiment data

    Returns
    -------
    Path
        exp_path_readout, the Path of the most recent readout tile
    """
    filename_readout_list = [i for i in exp_path.glob("*eadout*.bin")]
    if len(filename_readout_list) == 1:
        exp_path_readout = filename_readout_list[0]
    elif len(filename_readout_list) == 0:
        raise f"RAISED: No readout file found in {exp_path}"
    else:
        for i_file, potential_file in enumerate(filename_readout_list):
            potential_file_str = str(potential_file)
            readout_time = int(potential_file_str[potential_file_str.rfind('T') + 1: potential_file_str.rfind('T') + 5])
            if i_file == 0:
                max_time = readout_time
                exp_path_readout = potential_file
            else:
                if readout_time > max_time:
                    exp_path_readout = potential_file
    return exp_path_readout


def find_most_recent_find_active(exp_path: Path) -> Path:
    """
    Path of the most recent ``*find_active*.bin`` in an experiment folder.

    Raises
    ------
    FileNotFoundError
        If no find-active file is present.
    """
    exp_path = Path(exp_path)
    candidates = list(exp_path.glob("*find_active*.bin"))
    if not candidates:
        raise FileNotFoundError(f"No find_active file found in {exp_path}")
    if len(candidates) == 1:
        return candidates[0]

    def _time_key(p: Path) -> int:
        s = str(p)
        t_pos = s.rfind("T")
        if t_pos >= 0 and t_pos + 5 <= len(s):
            try:
                return int(s[t_pos + 1 : t_pos + 5])
            except ValueError:
                pass
        return 0

    return max(candidates, key=_time_key)


def load_temp_log(exp_path: Path):
    """
    Load `*_temp_log.bin` (heating-phase temperature trajectory) from a v05+ experiment folder.

    Format (Lacewing_UI_v05/Lacewing_Thread.py:2887-2894):
        [uint32 N, big-endian]                  # number of heating-phase samples
        [N x float32, big-endian]               # timestamps in seconds (anchored at heating start)
        [N x float32, big-endian]               # ttn_tem_data_lin (firmware-linearised temperature)

    Parameters
    ----------
    exp_path : Path
        Folder containing the experiment .bin files.

    Returns
    -------
    tuple
        (time_s, temp_lin) as 1-D numpy float arrays. Returns (None, None) if no
        ``*_temp_log.bin`` is present, the file is malformed, or N is zero.
    """
    exp_path = Path(exp_path)
    candidates = list(exp_path.glob("*_temp_log.bin"))
    if not candidates:
        return None, None

    # Prefer the file whose timestamp pairs with the most recent readout file. As a fallback
    # (or when there is only one), pick the most recently modified.
    def _ts_key(p: Path):
        m = re.search(r"(?P<date>\d{8})T(?P<time>\d{4,6})", p.name)
        if m:
            return (1, int(m.group("date")), int(m.group("time")[:6]), p.stat().st_mtime)
        return (0, 0, 0, p.stat().st_mtime)

    latest = max(candidates, key=_ts_key)

    try:
        with open(latest, "rb") as fp:
            data = fp.read()
    except Exception:
        return None, None

    if len(data) < 4:
        return None, None
    n = int.from_bytes(data[0:4], byteorder="big", signed=False)
    if n <= 0 or len(data) < 4 + 8 * n:
        return None, None

    time_s = np.frombuffer(data[4 : 4 + 4 * n], dtype=">f4").astype(float)
    temp_lin = np.frombuffer(data[4 + 4 * n : 4 + 8 * n], dtype=">f4").astype(float)
    return time_s, temp_lin


def find_most_recent_vref(exp_path: Path) -> Path:
    """
    Finds the Path of the most recent Vref calibration file in the exp_path folder.

    Lacewing v05+ writes ``*_dvref_multi_payload.bin`` (full TTN dVref multi payload).
    Older runs used ``*_vref_sweep.bin`` (``[len(vref_list)] + vref_list`` as uint16 words).

    Notes
    -----
    - ``*_dvref_multi_payload.bin`` is preferred when both styles are present.
    - If timestamps are present in the filename, they are used to choose the newest file.
    - If no parseable timestamp is found, falls back to filesystem modified time.

    Parameters
    ----------
    exp_path: Path
        Path of the folder containing the experiment data.

    Returns
    -------
    Path
        exp_path_vref, the Path of the most recent Vref calibration file.
    """
    filename_vref_list = [
        *exp_path.glob("*_dvref_multi_payload.bin"),
        *exp_path.glob("*_vref_sweep.bin"),
        *exp_path.glob("*_vref_swepp.bin"),
    ]

    if len(filename_vref_list) == 1:
        return filename_vref_list[0]
    if len(filename_vref_list) == 0:
        raise Exception(
            f"RAISED: No Vref file found in {exp_path} "
            "(expected *_dvref_multi_payload.bin or *_vref_sweep.bin)"
        )

    def _sort_key(p: Path):
        # Prefer timestamp in filename if present:
        # - common pattern: YYYYMMDDT(HHMM|HHMMSS) somewhere in the name
        name = p.name
        m = re.search(r"(?P<date>\d{8})T(?P<time>\d{4,6})", name)
        if m:
            date_i = int(m.group("date"))
            time_i = int(m.group("time")[:6])  # normalize to comparable int
            return (1, date_i, time_i, p.stat().st_mtime)

        # Fallback: if there is a 'T####' token (like the readout convention), use that
        m2 = re.search(r"T(?P<time>\d{4,6})", name)
        if m2:
            time_i = int(m2.group("time")[:6])
            return (1, 0, time_i, p.stat().st_mtime)

        # Final fallback: filesystem modified time only
        return (0, 0, 0, p.stat().st_mtime)

    # Pick the max according to our sort key
    return max(filename_vref_list, key=_sort_key)
