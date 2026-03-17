from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
VENDOR_DIR = PROJECT_ROOT / ".vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.widgets import Button


def load_ieeg_mat(mat_path: str):
    """Load iEEG data from either `data` or `dataStruct.data`."""
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    data = None
    sampling_rate = None
    channel_indices = None

    if "dataStruct" in mat:
        ds = mat["dataStruct"]
        if hasattr(ds, "data"):
            data = np.asarray(ds.data)
        if hasattr(ds, "iEEGsamplingRate"):
            sampling_rate = float(ds.iEEGsamplingRate)
        if hasattr(ds, "channelIndices"):
            channel_indices = np.asarray(ds.channelIndices)

    if data is None and "data" in mat:
        data = np.asarray(mat["data"])
        if "iEEGsamplingRate" in mat:
            sampling_rate = float(np.asarray(mat["iEEGsamplingRate"]).squeeze())
        if "channelIndices" in mat:
            channel_indices = np.asarray(mat["channelIndices"])

    if data is None:
        raise ValueError(f"Could not find EEG data in {mat_path}")
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D EEG array, got shape {data.shape}")

    if sampling_rate is None:
        sampling_rate = data.shape[0] / 600.0

    if channel_indices is None:
        channel_names = [f"Ch {idx + 1}" for idx in range(data.shape[1])]
    else:
        channel_names = [f"Ch {int(idx)}" for idx in np.ravel(channel_indices)]

    return data, sampling_rate, channel_names


def _segment_scale(segment: np.ndarray) -> tuple[float, str]:
    mad = np.median(np.abs(segment - np.median(segment, axis=0)), axis=0)
    robust_std = 1.4826 * mad
    robust_std[robust_std == 0] = np.nan
    base_scale = np.nanmedian(robust_std)
    if not np.isfinite(base_scale) or base_scale <= 0:
        fallback = float(np.nanstd(segment))
        base_scale = fallback if fallback > 0 else 1.0

    spacing = 6.0 * base_scale
    return spacing, f"Auto scale (spacing~{spacing:.1f})"


def plot_doctor_eeg(
    data: np.ndarray,
    sampling_rate: float,
    channel_names=None,
    start_sec: float = 0.0,
    page_sec: float = 10.0,
    title: str = "EEG (doctor view)",
):
    n_samples, n_channels = data.shape
    if channel_names is None:
        channel_names = [f"Ch {idx + 1}" for idx in range(n_channels)]

    start = int(start_sec * sampling_rate)
    end = min(int((start_sec + page_sec) * sampling_rate), n_samples)
    segment = data[start:end, :]
    times = np.arange(start, end) / sampling_rate

    spacing, scale_label = _segment_scale(segment)
    order = list(range(n_channels))[::-1]
    y_offsets = np.arange(n_channels) * spacing

    plt.figure(figsize=(14, 9))
    axis = plt.gca()

    for row, channel_idx in enumerate(order):
        axis.plot(times, segment[:, channel_idx] + y_offsets[row], linewidth=0.8)
        axis.text(
            times[0] - 0.01 * page_sec,
            y_offsets[row],
            channel_names[channel_idx],
            va="center",
            ha="right",
            fontsize=9,
        )

    axis.set_xlim(times[0], times[-1])
    axis.set_xticks(np.arange(np.floor(times[0]), np.ceil(times[-1]) + 1e-9, 1.0))
    axis.set_yticks(y_offsets)
    axis.grid(True, which="major", axis="x", linewidth=0.6)
    axis.grid(True, which="major", axis="y", linewidth=0.6)
    axis.set_yticklabels([])
    axis.set_xlabel("Time (s)")
    axis.set_title(f"{title} | start={start_sec:.1f}s, page={page_sec:.1f}s | {scale_label}")
    plt.tight_layout()
    plt.show()


def interactive_eeg_viewer(
    data: np.ndarray,
    sampling_rate: float,
    channel_names=None,
    page_sec: float = 10.0,
    title: str = "EEG (doctor view)",
):
    n_samples, n_channels = data.shape
    if channel_names is None:
        channel_names = [f"Ch {idx + 1}" for idx in range(n_channels)]

    total_seconds = n_samples / sampling_rate
    max_start_sec = max(0.0, total_seconds - page_sec)
    state = {"start_sec": 0.0}

    figure = plt.figure(figsize=(14, 10))
    ax_main = figure.add_axes([0.08, 0.12, 0.84, 0.82])
    ax_prev = figure.add_axes([0.35, 0.02, 0.12, 0.05])
    ax_next = figure.add_axes([0.53, 0.02, 0.12, 0.05])

    btn_prev = Button(ax_prev, "Previous")
    btn_next = Button(ax_next, "Next")

    def get_segment():
        start = int(state["start_sec"] * sampling_rate)
        end = min(int((state["start_sec"] + page_sec) * sampling_rate), n_samples)
        return data[start:end, :], np.arange(start, end) / sampling_rate

    def redraw():
        ax_main.clear()
        segment, times = get_segment()
        if segment.size == 0:
            ax_main.text(0.5, 0.5, "No data in this range", transform=ax_main.transAxes, ha="center", va="center")
            figure.canvas.draw_idle()
            return

        spacing, scale_label = _segment_scale(segment)
        order = list(range(n_channels))[::-1]
        y_offsets = np.arange(n_channels) * spacing

        for row, channel_idx in enumerate(order):
            ax_main.plot(times, segment[:, channel_idx] + y_offsets[row], linewidth=0.8)
            ax_main.text(
                times[0] - 0.01 * page_sec,
                y_offsets[row],
                channel_names[channel_idx],
                va="center",
                ha="right",
                fontsize=9,
            )

        ax_main.set_xlim(times[0], times[-1])
        ax_main.set_xticks(np.arange(np.floor(times[0]), np.ceil(times[-1]) + 1e-9, 1.0))
        ax_main.set_yticks(y_offsets)
        ax_main.grid(True, which="major", axis="x", linewidth=0.6)
        ax_main.grid(True, which="major", axis="y", linewidth=0.6)
        ax_main.set_yticklabels([])
        ax_main.set_xlabel("Time (s)")
        ax_main.set_title(
            f"{title} | {state['start_sec']:.1f}s - {state['start_sec'] + page_sec:.1f}s | {scale_label}"
        )
        figure.canvas.draw_idle()

    def on_prev(_):
        state["start_sec"] = max(0.0, state["start_sec"] - page_sec)
        redraw()

    def on_next(_):
        state["start_sec"] = min(max_start_sec, state["start_sec"] + page_sec)
        redraw()

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    redraw()
    plt.show()


if __name__ == "__main__":
    mat_path = str(PROJECT_ROOT / "Pat1Test_1_0.mat")
    data, sampling_rate, channel_names = load_ieeg_mat(mat_path)
    interactive_eeg_viewer(data, sampling_rate, channel_names, page_sec=10)
