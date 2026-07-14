import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

summary_df = pd.read_csv("analysis_output/session_summary.csv")
_valid = summary_df[summary_df["valid_session"]] if "valid_session" in summary_df.columns else summary_df[summary_df["duration_s"] > 30].copy()

# --- PREPROCESSING: Remove Audio-Only Sessions from Video Metrics ---
# If FPS <= 2, it is essentially audio-only/screen-share. We convert the FPS 
# and Jitter values to np.nan so that .mean(skipna=True) completely ignores them!
for direction in ["uplink", "downlink"]:
    fps_col = f"{direction}_fps_m1"
    jitter_col = f"{direction}_jitter_ms"
    video_bitrate_col = f"{direction}_video_bitrate_mbps"
    
    if fps_col in _valid.columns:
        # Find rows where this direction had no real video
        audio_only_mask = _valid[fps_col] <= 2
        
        # Blank out the video metrics for those specific rows
        _valid.loc[audio_only_mask, fps_col] = np.nan
        
        if jitter_col in _valid.columns:
            _valid.loc[audio_only_mask, jitter_col] = np.nan
            
        if video_bitrate_col in _valid.columns:
            _valid.loc[audio_only_mask, video_bitrate_col] = np.nan


PLATFORMS = ["Meet", "Teams"]
PCOLORS = {"Meet": "tab:orange", "Teams": "tab:purple"}

def _panel(ax, title, ylabel, series, kind="grouped"):
    x = np.arange(len(PLATFORMS))
    if kind == "count":
        vals = [int((_valid["platform"] == p).sum()) for p in PLATFORMS]
        ax.bar(x, vals, width=0.55, color=[PCOLORS[p] for p in PLATFORMS])
        ax.set_xticks(x)
        ax.set_xticklabels(PLATFORMS)
    else:
        width = 0.8 / max(len(series), 1)
        for i, (col, label) in enumerate(series):
            vals = [_valid.loc[_valid["platform"] == p, col].mean(skipna=True) if col in _valid.columns else np.nan for p in PLATFORMS]
            ax.bar(x + i * width, vals, width=width, label=label)
        ax.set_xticks(x + width * (len(series) - 1) / 2)
        ax.set_xticklabels(PLATFORMS)
        ax.legend(frameon=False, loc="best")

    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

# Changed to a perfect 4x2 layout for 8 plots!
fig, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)
fig.suptitle("Google Meet vs Microsoft Teams: Aggregate Session Comparison", fontweight="bold", fontsize=16)
ax = axes.ravel()

_panel(ax[0], "Number of Sessions", "Count", [], kind="count")
_panel(ax[1], "Average Session Duration", "Seconds", [("duration_s", "Duration")])
_panel(ax[2], "Average Throughput", "Mbps", [("uplink_throughput_mbps", "Uplink"), ("downlink_throughput_mbps", "Downlink")])
_panel(ax[3], "Average Video Bitrate", "Mbps", [("uplink_video_bitrate_mbps", "Uplink"), ("downlink_video_bitrate_mbps", "Downlink")])
_panel(ax[4], "Average Audio Bitrate", "Mbps", [("uplink_audio_bitrate_mbps", "Uplink"), ("downlink_audio_bitrate_mbps", "Downlink")])
_panel(ax[5], "Average FPS", "Frames/s", [("uplink_fps_m1", "Uplink"), ("downlink_fps_m1", "Downlink")])
_panel(ax[6], "Average Jitter", "ms", [("uplink_jitter_ms", "Uplink"), ("downlink_jitter_ms", "Downlink")])
_panel(ax[7], "Average Packet Loss", "%", [("uplink_loss_pct", "Uplink"), ("downlink_loss_pct", "Downlink")])

save_path = "analysis_output/platform_comparison_plots/meet_vs_teams_overview_ieee.png"
fig.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"✅ Saved perfectly symmetrical 4x2 figure to: {save_path}")
