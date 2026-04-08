import argparse
import json
import os
from dataclasses import asdict, dataclass
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCENARIOS = ("v2g", "g2v", "g2b", "b2g")
DEFAULT_HARMONICS = (1, 3, 5, 7)


@dataclass
class CalibrationResult:
    k: float
    b: float
    r2: float
    points: int
    files: int


@dataclass
class ProcessResult:
    signals: np.ndarray  # [fft_cycles, samples_per_cycle]
    label: np.ndarray  # [4]
    channel_source: str
    fft_samples_per_cycle: int


def parse_harmonics(text: str) -> Tuple[int, ...]:
    vals = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    if not vals:
        raise ValueError("harmonics is empty")
    return tuple(vals)


def infer_scenario(folder_name: str) -> Optional[str]:
    prefix = folder_name.split("_")[0].lower()
    return prefix if prefix in SCENARIOS else None


def g2v_folder_keep(folder_name: str, current_filter: str) -> bool:
    if current_filter == "all":
        return True
    if current_filter == "6A":
        return "_6A" in folder_name
    raise ValueError(f"Unsupported g2v current filter: {current_filter}")


def power_folder_keep(folder_name: str, power_filter: str) -> bool:
    if power_filter == "all":
        return True
    if power_filter in ("0.3kw", "0.5kw"):
        return f"_{power_filter}_" in folder_name or folder_name.endswith(f"_{power_filter}")
    raise ValueError(f"Unsupported power filter: {power_filter}")


def read_numeric_columns(csv_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    cols = list(pd.read_csv(csv_path, nrows=0).columns)
    usecols = ["Time", "Channel B"] + (["Channel C"] if "Channel C" in cols else [])
    df = pd.read_csv(csv_path, usecols=usecols)

    time_ms = pd.to_numeric(df["Time"], errors="coerce").to_numpy(dtype=np.float64)
    ch_b = pd.to_numeric(df["Channel B"], errors="coerce").to_numpy(dtype=np.float64)
    ch_c = (
        pd.to_numeric(df["Channel C"], errors="coerce").to_numpy(dtype=np.float64)
        if "Channel C" in df.columns
        else None
    )

    # Remove unit row and any invalid time rows.
    time_mask = np.isfinite(time_ms)
    time_ms = time_ms[time_mask]
    ch_b = ch_b[time_mask]
    if ch_c is not None:
        ch_c = ch_c[time_mask]
    return time_ms, ch_b, ch_c


def fit_channel_c_calibration(
    csv_paths: Sequence[str],
    min_finite_ratio: float = 0.99,
    min_points: int = 1000,
    max_points_per_file: int = 20000,
) -> CalibrationResult:
    all_c = []
    all_b = []
    used_files = 0

    for csv_path in csv_paths:
        cols = list(pd.read_csv(csv_path, nrows=0).columns)
        if "Channel C" not in cols:
            continue

        time_ms, ch_b, ch_c = read_numeric_columns(csv_path)
        if ch_c is None or time_ms.size < 2:
            continue

        finite_ratio_b = np.isfinite(ch_b).mean() if ch_b.size > 0 else 0.0
        if finite_ratio_b < min_finite_ratio:
            continue

        mask = np.isfinite(ch_b) & np.isfinite(ch_c)
        n_pts = int(mask.sum())
        if n_pts < min_points:
            continue

        if n_pts > max_points_per_file:
            idx = np.linspace(0, n_pts - 1, max_points_per_file, dtype=np.int64)
            b_pts = ch_b[mask][idx]
            c_pts = ch_c[mask][idx]
        else:
            b_pts = ch_b[mask]
            c_pts = ch_c[mask]

        all_b.append(b_pts)
        all_c.append(c_pts)
        used_files += 1

    if not all_b:
        return CalibrationResult(k=1.0, b=0.0, r2=float("nan"), points=0, files=0)

    b_cat = np.concatenate(all_b).astype(np.float64)
    c_cat = np.concatenate(all_c).astype(np.float64)
    k, b = np.polyfit(c_cat, b_cat, 1)

    pred = k * c_cat + b
    ss_res = float(np.sum((b_cat - pred) ** 2))
    ss_tot = float(np.sum((b_cat - np.mean(b_cat)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return CalibrationResult(
        k=float(k),
        b=float(b),
        r2=r2,
        points=int(b_cat.size),
        files=int(used_files),
    )


def extract_cycle_segments(
    time_ms: np.ndarray,
    current_a: np.ndarray,
    f0_hz: float,
) -> List[Tuple[float, float, np.ndarray, np.ndarray]]:
    cycle_ms = 1000.0 / f0_hz
    if time_ms.size < 2:
        return []

    t0 = float(time_ms[0])
    n_cycles = int(np.floor((float(time_ms[-1]) - t0) / cycle_ms))

    segments = []
    for idx in range(n_cycles):
        start = t0 + idx * cycle_ms
        end = start + cycle_ms
        mask = (time_ms >= start) & (time_ms < end)
        t_seg = time_ms[mask]
        y_seg = current_a[mask]
        segments.append((start, end, t_seg, y_seg))
    return segments


def cycle_is_valid(seg: Tuple[float, float, np.ndarray, np.ndarray]) -> bool:
    _, _, t_seg, y_seg = seg
    return (t_seg.size >= 2) and np.isfinite(y_seg).all()


def find_first_valid_window(
    segments: Sequence[Tuple[float, float, np.ndarray, np.ndarray]],
    fft_cycles: int,
) -> Optional[List[Tuple[float, float, np.ndarray, np.ndarray]]]:
    if len(segments) < fft_cycles:
        return None
    for start_idx in range(0, len(segments) - fft_cycles + 1):
        window = segments[start_idx : start_idx + fft_cycles]
        if all(cycle_is_valid(seg) for seg in window):
            return list(window)
    return None


def resample_segment(
    seg: Tuple[float, float, np.ndarray, np.ndarray],
    num_points: int,
) -> np.ndarray:
    start, end, t_seg, y_seg = seg
    t_target = np.linspace(start, end, num_points, endpoint=False, dtype=np.float64)
    y_target = np.interp(t_target, t_seg, y_seg)
    return y_target.astype(np.float32)


def estimate_harmonics_from_window(
    window: Sequence[Tuple[float, float, np.ndarray, np.ndarray]],
    harmonics: Sequence[int],
    f0_hz: float,
) -> Tuple[np.ndarray, int]:
    # Build a uniformly sampled 10-cycle signal and use rectangular-window FFT bin amplitude.
    all_dts = []
    for _, _, t_seg, _ in window:
        if t_seg.size >= 2:
            all_dts.append(np.diff(t_seg))
    if not all_dts:
        raise ValueError("Cannot estimate dt from window")

    dt_ms = float(np.median(np.concatenate(all_dts)))
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError(f"Invalid dt_ms={dt_ms}")

    cycle_ms = 1000.0 / f0_hz
    fft_samples_per_cycle = int(round(cycle_ms / dt_ms))
    if fft_samples_per_cycle <= 8:
        raise ValueError(f"Too few FFT samples per cycle: {fft_samples_per_cycle}")

    sig_10cy = []
    for seg in window:
        sig_10cy.append(resample_segment(seg, fft_samples_per_cycle).astype(np.float64))
    x = np.concatenate(sig_10cy, axis=0)
    n = x.size

    fft_vals = np.fft.rfft(x)
    n_cycles = len(window)
    amps = []
    for h in harmonics:
        bin_idx = h * n_cycles
        if bin_idx >= fft_vals.size:
            raise ValueError(f"FFT bin out of range: h={h}, bin={bin_idx}, nfft={fft_vals.size}")
        amp = (2.0 / n) * np.abs(fft_vals[bin_idx])
        amps.append(float(amp))
    return np.asarray(amps, dtype=np.float32), fft_samples_per_cycle


def process_one_csv(
    csv_path: str,
    calibration: CalibrationResult,
    f0_hz: float,
    fft_cycles: int,
    samples_per_cycle: int,
    harmonics: Sequence[int],
) -> Tuple[Optional[ProcessResult], Optional[str]]:
    time_ms, ch_b, ch_c = read_numeric_columns(csv_path)
    if time_ms.size < 2:
        return None, "too_short"

    if ch_c is not None and np.any(~np.isfinite(ch_b)):
        current_a = calibration.k * ch_c + calibration.b
        source = "C_converted"
    else:
        current_a = ch_b
        source = "B"

    segments = extract_cycle_segments(time_ms, current_a, f0_hz=f0_hz)
    if len(segments) < fft_cycles:
        return None, "insufficient_cycles"

    window = find_first_valid_window(segments, fft_cycles=fft_cycles)
    if window is None:
        return None, "no_full_valid_window"

    try:
        label, fft_samples_per_cycle = estimate_harmonics_from_window(
            window=window,
            harmonics=harmonics,
            f0_hz=f0_hz,
        )
    except Exception:
        return None, "fft_failed"

    cycle_signals = []
    for seg in window:
        cycle_signals.append(resample_segment(seg, num_points=samples_per_cycle))
    signals = np.stack(cycle_signals, axis=0).astype(np.float32)

    return ProcessResult(
        signals=signals,
        label=label.astype(np.float32),
        channel_source=source,
        fft_samples_per_cycle=int(fft_samples_per_cycle),
    ), None


def collect_scenario_files(
    input_dir: str,
    g2v_current_filter: str,
    g2b_power_filter: str,
    b2g_power_filter: str,
) -> Dict[str, List[str]]:
    mapping = {sc: [] for sc in SCENARIOS}
    for name in sorted(os.listdir(input_dir)):
        subdir = os.path.join(input_dir, name)
        if not os.path.isdir(subdir):
            continue
        scenario = infer_scenario(name)
        if scenario is None:
            continue
        if scenario == "g2v" and (not g2v_folder_keep(name, g2v_current_filter)):
            continue
        if scenario == "g2b" and (not power_folder_keep(name, g2b_power_filter)):
            continue
        if scenario == "b2g" and (not power_folder_keep(name, b2g_power_filter)):
            continue
        mapping[scenario].extend(sorted(glob(os.path.join(subdir, "*.csv"))))
    return mapping


def save_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Generate realdata harmonic datasets with 10-cycle FFT GT")
    parser.add_argument("--input_dir", type=str, default="dataset/realdata")
    parser.add_argument("--out_dir", type=str, default="dataset")
    parser.add_argument("--samples_per_cycle", type=int, default=64)
    parser.add_argument("--fft_cycles", type=int, default=10)
    parser.add_argument("--f0", type=float, default=50.0)
    parser.add_argument("--harmonics", type=str, default="1,3,5,7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib_min_finite_ratio", type=float, default=0.99)
    parser.add_argument("--calib_min_points", type=int, default=1000)
    parser.add_argument("--calib_max_points_per_file", type=int, default=20000)
    parser.add_argument(
        "--g2v_current_filter",
        type=str,
        default="6A",
        choices=["all", "6A"],
        help="Filter g2v folders by current rating. Default keeps only *_6A.",
    )
    parser.add_argument(
        "--g2b_power_filter",
        type=str,
        default="0.5kw",
        choices=["all", "0.3kw", "0.5kw"],
        help="Filter g2b folders by power tag. Default keeps only *_0.5kw_*.",
    )
    parser.add_argument(
        "--b2g_power_filter",
        type=str,
        default="0.5kw",
        choices=["all", "0.3kw", "0.5kw"],
        help="Filter b2g folders by power tag. Default keeps only *_0.5kw_*.",
    )
    args = parser.parse_args()

    del args.seed  # deterministic processing path; keep CLI for interface stability
    harmonics = parse_harmonics(args.harmonics)

    os.makedirs(args.out_dir, exist_ok=True)

    scenario_files = collect_scenario_files(
        input_dir=args.input_dir,
        g2v_current_filter=args.g2v_current_filter,
        g2b_power_filter=args.g2b_power_filter,
        b2g_power_filter=args.b2g_power_filter,
    )
    all_csv_paths = []
    for paths in scenario_files.values():
        all_csv_paths.extend(paths)

    calibration = fit_channel_c_calibration(
        all_csv_paths,
        min_finite_ratio=args.calib_min_finite_ratio,
        min_points=args.calib_min_points,
        max_points_per_file=args.calib_max_points_per_file,
    )
    print(
        f"[Calibration] I = k*C + b, k={calibration.k:.9f}, b={calibration.b:.9f}, "
        f"R2={calibration.r2:.6f}, files={calibration.files}, points={calibration.points}"
    )

    global_summary = {
        "input_dir": args.input_dir,
        "out_dir": args.out_dir,
        "f0_hz": args.f0,
        "samples_per_cycle": args.samples_per_cycle,
        "fft_cycles": args.fft_cycles,
        "harmonics": list(harmonics),
        "g2v_current_filter": args.g2v_current_filter,
        "g2b_power_filter": args.g2b_power_filter,
        "b2g_power_filter": args.b2g_power_filter,
        "calibration": asdict(calibration),
        "scenarios": {},
    }

    for scenario in SCENARIOS:
        csv_paths = scenario_files.get(scenario, [])
        print(f"\n[Scenario {scenario}] input csv files: {len(csv_paths)}")

        signals_all = []
        labels_all = []
        group_ids_all = []
        sample_cycle_index_all = []
        group_files = []
        group_channel_source = []
        group_fft_samples_per_cycle = []
        dropped = []

        next_group_id = 0

        for csv_path in csv_paths:
            proc, drop_reason = process_one_csv(
                csv_path=csv_path,
                calibration=calibration,
                f0_hz=args.f0,
                fft_cycles=args.fft_cycles,
                samples_per_cycle=args.samples_per_cycle,
                harmonics=harmonics,
            )
            if proc is None:
                dropped.append(
                    {
                        "file": os.path.relpath(csv_path, args.input_dir),
                        "reason": drop_reason,
                    }
                )
                continue

            n_samples = proc.signals.shape[0]
            signals_all.append(proc.signals)
            labels_all.append(np.repeat(proc.label[None, :], n_samples, axis=0))
            group_ids_all.append(np.full((n_samples,), next_group_id, dtype=np.int64))
            sample_cycle_index_all.append(np.arange(n_samples, dtype=np.int64))

            group_files.append(os.path.relpath(csv_path, args.input_dir))
            group_channel_source.append(proc.channel_source)
            group_fft_samples_per_cycle.append(proc.fft_samples_per_cycle)
            next_group_id += 1

        if signals_all:
            signals_np = np.concatenate(signals_all, axis=0).astype(np.float32)
            labels_np = np.concatenate(labels_all, axis=0).astype(np.float32)
            group_ids_np = np.concatenate(group_ids_all, axis=0).astype(np.int64)
            cycle_idx_np = np.concatenate(sample_cycle_index_all, axis=0).astype(np.int64)
        else:
            signals_np = np.zeros((0, args.samples_per_cycle), dtype=np.float32)
            labels_np = np.zeros((0, len(harmonics)), dtype=np.float32)
            group_ids_np = np.zeros((0,), dtype=np.int64)
            cycle_idx_np = np.zeros((0,), dtype=np.int64)

        out_npz = os.path.join(args.out_dir, f"real_{scenario}.npz")
        np.savez_compressed(
            out_npz,
            signals=signals_np,
            labels=labels_np,
            group_ids=group_ids_np,
            sample_cycle_index=cycle_idx_np,
            harmonics=np.asarray(harmonics, dtype=np.int64),
            f0_hz=np.asarray(args.f0, dtype=np.float64),
            samples_per_cycle=np.asarray(args.samples_per_cycle, dtype=np.int64),
            fft_cycles=np.asarray(args.fft_cycles, dtype=np.int64),
            calibration_k=np.asarray(calibration.k, dtype=np.float64),
            calibration_b=np.asarray(calibration.b, dtype=np.float64),
            calibration_r2=np.asarray(calibration.r2, dtype=np.float64),
            group_files=np.asarray(group_files, dtype=object),
            group_channel_source=np.asarray(group_channel_source, dtype=object),
            group_fft_samples_per_cycle=np.asarray(group_fft_samples_per_cycle, dtype=np.int64),
        )

        dropped_reason_count = {}
        for item in dropped:
            dropped_reason_count[item["reason"]] = dropped_reason_count.get(item["reason"], 0) + 1

        meta = {
            "scenario": scenario,
            "npz_path": out_npz,
            "num_input_files": len(csv_paths),
            "num_kept_files": int(next_group_id),
            "num_dropped_files": int(len(dropped)),
            "dropped_reason_count": dropped_reason_count,
            "num_samples": int(signals_np.shape[0]),
            "samples_per_kept_file": args.fft_cycles,
            "num_groups": int(next_group_id),
            "group_channel_source_count": {
                "B": int(sum(src == "B" for src in group_channel_source)),
                "C_converted": int(sum(src == "C_converted" for src in group_channel_source)),
            },
            "calibration": asdict(calibration),
            "dropped_files": dropped,
        }

        meta_path = os.path.join(args.out_dir, f"real_{scenario}_meta.json")
        save_json(meta_path, meta)
        print(
            f"[Scenario {scenario}] kept_files={meta['num_kept_files']}, "
            f"dropped_files={meta['num_dropped_files']}, samples={meta['num_samples']}, "
            f"C_converted_groups={meta['group_channel_source_count']['C_converted']}"
        )
        print(f"[Scenario {scenario}] saved: {out_npz}")
        print(f"[Scenario {scenario}] metadata: {meta_path}")

        global_summary["scenarios"][scenario] = meta

    global_summary_path = os.path.join(args.out_dir, "realdata_preprocess_summary.json")
    save_json(global_summary_path, global_summary)
    print(f"\nSaved global summary to: {global_summary_path}")


if __name__ == "__main__":
    main()
