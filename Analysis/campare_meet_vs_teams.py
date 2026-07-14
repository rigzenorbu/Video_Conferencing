#!/usr/bin/env python3
"""
compare_teams_meet.py — Teams vs Meet QoE Comparison
=====================================================
Replicates the exact metric logic from the Rust capture engine:
  - network_metrics.rs  → Packet Loss %, Throughput (bps)
  - media_metrics.rs    → FPS, Frame Jitter (ms), Video Bitrate (bps)

Usage:
  python3 compare_teams_meet.py

Expects:
  Sessions_18_to_19/Teams/*.csv
  Sessions_18_to_19/Meet/*.csv

Outputs:
  qoe_boxplots.png       — Side-by-side box plots
  qoe_cdf.png            — CDF overlay curves
  qoe_summary.csv        — Summary statistics table
"""

import os
import sys
import csv
import re
import warnings
from datetime import datetime, timezone

# Suppress the expected overflow warning from uint16 wrapping subtraction
# (This is intentional — it replicates Rust's .wrapping_sub() for RTP seq rollover)
warnings.filterwarnings('ignore', message='overflow encountered in scalar subtract')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
#  Constants (matching mod.rs)
# ─────────────────────────────────────────────────────────────────────────────
BIN_SIZE_NS = 5_000_000_000          # 5-second bins
MIN_SESSION_DURATION_SEC = 10        # Skip sessions shorter than 10 seconds
AUDIO_VIDEO_THRESHOLD = 400          # udp_len <= 400 → Audio (media_metrics.rs L46)
MIN_GAP_TICKS = 1_500                # 60 FPS in 90kHz ticks (media_metrics.rs L6)
MAX_GAP_TICKS = 90_000               # 1 FPS in 90kHz ticks (media_metrics.rs L7)

# ─────────────────────────────────────────────────────────────────────────────
#  Server IP prefixes (matching network_metrics.rs L5-L31)
# ─────────────────────────────────────────────────────────────────────────────
TEAMS_SERVER_PREFIXES = [
    "52.112.", "52.113.", "52.114.", "52.115.", "52.122.", "52.123.",
    "2603:1010", "2603:1027", "2603:1037", "2603:1047", "2603:1057", "2620:1ec",
]

MEET_SERVER_PREFIXES = [
    "74.125.250.", "74.125.247.128", "142.250.82.",
    "2001:4860:4864:5:", "2001:4860:4864:4:8000:", "2001:4860:4864:6:",
]

def is_teams_server(ip):
    return any(ip.startswith(p) for p in TEAMS_SERVER_PREFIXES)

def is_meet_server(ip):
    return any(ip.startswith(p) for p in MEET_SERVER_PREFIXES)


# ─────────────────────────────────────────────────────────────────────────────
#  Core: Process a single session CSV → list of per-bin metric dicts
# ─────────────────────────────────────────────────────────────────────────────
def process_session(filepath, platform):
    """
    Reads a session CSV and computes per-5-second-bin metrics.
    Returns a list of dicts, one per bin, or None if session is too short.
    """
    is_server = is_teams_server if platform == "Teams" else is_meet_server

    # ── Read CSV ────────────────────────────────────────────────────────────
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if len(rows) < 2:
        return None

    # ── Parse into numpy-friendly arrays ────────────────────────────────────
    arrival_ns = np.array([int(r['arrival_epoch_ns']) for r in rows], dtype=np.int64)
    src_ips    = [r['src_ip'] for r in rows]
    dst_ips    = [r['dst_ip'] for r in rows]
    udp_lens   = np.array([int(r['udp_len']) for r in rows], dtype=np.int64)
    rtp_ssrcs  = np.array([int(r['rtp_ssrc'], 16) if r['rtp_ssrc'].startswith('0x') 
                           else int(r['rtp_ssrc']) for r in rows], dtype=np.uint32)
    rtp_seqs   = np.array([int(r['rtp_seq']) for r in rows], dtype=np.uint16)
    rtp_ts     = np.array([int(r['rtp_timestamp']) for r in rows], dtype=np.uint32)

    # ── Check session duration ──────────────────────────────────────────────
    duration_ns = arrival_ns[-1] - arrival_ns[0]
    duration_sec = duration_ns / 1e9
    if duration_sec < MIN_SESSION_DURATION_SEC:
        return None

    # ── Classify direction: Uplink (client→server) or Downlink (server→client)
    #    is_downlink[i] = True if src_ip is a server IP (server sending TO client)
    is_downlink = np.array([is_server(src_ips[i]) for i in range(len(rows))], dtype=bool)
    is_uplink   = np.array([is_server(dst_ips[i]) for i in range(len(rows))], dtype=bool)

    # ── Assign packets to 5-second bins ─────────────────────────────────────
    start_ns = arrival_ns[0]
    bin_ids  = ((arrival_ns - start_ns) // BIN_SIZE_NS).astype(np.int64)
    max_bin  = int(bin_ids[-1])

    # ── Per-SSRC sequence state (persists across bins, matching Rust L150-151)
    seq_state = {}  # ssrc -> last_seq (uint16)

    # ── Iterate over each 5-second bin ──────────────────────────────────────
    bin_results = []

    for b in range(max_bin + 1):
        mask = bin_ids == b
        if not np.any(mask):
            continue

        b_arrival  = arrival_ns[mask]
        b_udp_len  = udp_lens[mask]
        b_ssrc     = rtp_ssrcs[mask]
        b_seq      = rtp_seqs[mask]
        b_rtp_ts   = rtp_ts[mask]
        b_downlink = is_downlink[mask]
        b_uplink   = is_uplink[mask]

        # ── NETWORK METRICS (network_metrics.rs) ────────────────────────────
        # Throughput
        down_bytes = int(np.sum(b_udp_len[b_downlink])) if np.any(b_downlink) else 0
        up_bytes   = int(np.sum(b_udp_len[b_uplink]))   if np.any(b_uplink) else 0
        down_bps   = down_bytes * 8.0 / 5.0
        up_bps     = up_bytes * 8.0 / 5.0

        # Packet counts
        down_packets = int(np.sum(b_downlink))
        up_packets   = int(np.sum(b_uplink))

        # Packet Loss — per-SSRC sequence gap detection (network_metrics.rs L72-90)
        down_lost = 0
        up_lost   = 0
        unique_ssrcs = np.unique(b_ssrc)

        for ssrc in unique_ssrcs:
            ssrc_mask = b_ssrc == ssrc
            ssrc_seqs = b_seq[ssrc_mask]
            ssrc_down = b_downlink[ssrc_mask]

            # Determine if this SSRC is downlink or uplink (use first packet)
            ssrc_is_downlink = ssrc_down[0] if len(ssrc_down) > 0 else False

            for seq_val in ssrc_seqs:
                seq_val_u16 = np.uint16(seq_val)
                if ssrc not in seq_state:
                    # First packet for this SSRC — just record it
                    seq_state[ssrc] = int(seq_val_u16)
                else:
                    last = np.uint16(seq_state[ssrc])
                    # Replicate Rust's wrapping_sub: cast to i16
                    diff = np.int16(np.uint16(seq_val_u16) - np.uint16(last))
                    diff = int(diff)
                    if diff > 0:
                        if diff > 1:
                            lost_this = diff - 1
                            if ssrc_is_downlink:
                                down_lost += lost_this
                            else:
                                up_lost += lost_this
                        seq_state[ssrc] = int(seq_val_u16)

        # Loss percentage (network_metrics.rs L110-116)
        down_expected = down_packets + down_lost
        down_loss_pct = (down_lost / down_expected * 100.0) if down_expected > 0 else 0.0

        up_expected = up_packets + up_lost
        up_loss_pct = (up_lost / up_expected * 100.0) if up_expected > 0 else 0.0

        # ── MEDIA METRICS (media_metrics.rs) ────────────────────────────────
        down_mask = b_downlink
        up_mask   = b_uplink
        video_mask = b_udp_len > AUDIO_VIDEO_THRESHOLD

        down_audio_bytes = int(np.sum(b_udp_len[down_mask & ~video_mask])) if np.any(down_mask & ~video_mask) else 0
        down_video_bytes = int(np.sum(b_udp_len[down_mask & video_mask]))  if np.any(down_mask & video_mask) else 0
        up_audio_bytes   = int(np.sum(b_udp_len[up_mask & ~video_mask]))   if np.any(up_mask & ~video_mask) else 0
        up_video_bytes   = int(np.sum(b_udp_len[up_mask & video_mask]))    if np.any(up_mask & video_mask) else 0

        down_audio_bps = down_audio_bytes * 8.0 / 5.0
        down_video_total_bps = down_video_bytes * 8.0 / 5.0
        up_audio_bps = up_audio_bytes * 8.0 / 5.0
        up_video_total_bps = up_video_bytes * 8.0 / 5.0

        # ── Helper: compute FPS, jitter, bitrate for a set of video packets
        def compute_video_metrics(v_ssrc, v_rtp_ts, v_arrival, v_udp_len):
            fps_values = []
            jitter_values = []
            bitrate_values = []

            for ssrc in np.unique(v_ssrc):
                s_mask    = v_ssrc == ssrc
                s_rtp_ts  = v_rtp_ts[s_mask]
                s_arrival = v_arrival[s_mask]
                s_bytes   = int(np.sum(v_udp_len[s_mask]))

                # Map RTP timestamp → first arrival (media_metrics.rs L62-64)
                frame_map = {}
                for i in range(len(s_rtp_ts)):
                    ts_val = int(s_rtp_ts[i])
                    if ts_val not in frame_map:
                        frame_map[ts_val] = int(s_arrival[i])

                if len(frame_map) < 2:
                    continue

                frames = sorted(frame_map.items(), key=lambda x: x[0])

                # Validate video gaps (media_metrics.rs L104-113)
                valid_gap_count = 0
                for i in range(1, len(frames)):
                    gap = int(np.uint32(np.uint32(frames[i][0]) - np.uint32(frames[i-1][0])))
                    if MIN_GAP_TICKS <= gap <= MAX_GAP_TICKS:
                        valid_gap_count += 1

                if valid_gap_count == 0:
                    continue

                span_ticks = int(np.uint32(np.uint32(frames[-1][0]) - np.uint32(frames[0][0])))
                actual_duration = span_ticks / 90000.0
                if actual_duration <= 0:
                    continue

                stream_fps = (len(frames) - 1) / actual_duration

                # Frame Jitter — std dev of inter-frame arrival gaps in ms (L127-140)
                arrival_gaps_ms = []
                for i in range(1, len(frames)):
                    gap_ns = frames[i][1] - frames[i-1][1]
                    if gap_ns < 0:
                        gap_ns = 0
                    arrival_gaps_ms.append(gap_ns / 1_000_000.0)

                if len(arrival_gaps_ms) > 0:
                    mean_gap = np.mean(arrival_gaps_ms)
                    variance = np.mean([(g - mean_gap)**2 for g in arrival_gaps_ms])
                    stream_jitter = np.sqrt(variance)
                else:
                    stream_jitter = 0.0

                stream_bitrate = (s_bytes * 8) / actual_duration

                fps_values.append(stream_fps)
                jitter_values.append(stream_jitter)
                bitrate_values.append(stream_bitrate)

            avg_fps     = np.mean(fps_values) if fps_values else 0.0
            avg_jitter  = np.mean(jitter_values) if jitter_values else 0.0
            avg_bitrate = np.mean(bitrate_values) if bitrate_values else 0.0
            return avg_fps, avg_jitter, avg_bitrate

        # Downlink video metrics
        down_video_mask = down_mask & video_mask
        if np.any(down_video_mask):
            down_fps, down_frame_jitter_ms, down_video_bitrate_bps = compute_video_metrics(
                b_ssrc[down_video_mask], b_rtp_ts[down_video_mask],
                b_arrival[down_video_mask], b_udp_len[down_video_mask])
        else:
            down_fps, down_frame_jitter_ms, down_video_bitrate_bps = 0.0, 0.0, 0.0

        # Uplink video metrics
        up_video_mask = up_mask & video_mask
        if np.any(up_video_mask):
            up_fps, up_frame_jitter_ms, up_video_bitrate_bps = compute_video_metrics(
                b_ssrc[up_video_mask], b_rtp_ts[up_video_mask],
                b_arrival[up_video_mask], b_udp_len[up_video_mask])
        else:
            up_fps, up_frame_jitter_ms, up_video_bitrate_bps = 0.0, 0.0, 0.0

        bin_results.append({
            'down_bps': down_bps,
            'up_bps': up_bps,
            'down_loss_pct': down_loss_pct,
            'up_loss_pct': up_loss_pct,
            'down_fps': down_fps,
            'up_fps': up_fps,
            'down_frame_jitter_ms': down_frame_jitter_ms,
            'up_frame_jitter_ms': up_frame_jitter_ms,
            'down_video_bitrate_bps': down_video_bitrate_bps,
            'up_video_bitrate_bps': up_video_bitrate_bps,
            'down_audio_bps': down_audio_bps,
            'up_audio_bps': up_audio_bps,
            'down_video_total_bps': down_video_total_bps,
            'up_video_total_bps': up_video_total_bps,
        })

    if not bin_results:
        return None

    # ── Extract client_ip from filename (Session_{client_ip}_{timestamp}.csv)
    basename = os.path.basename(filepath)
    # Remove 'Session_' prefix and '.csv' suffix, then split on last '_'
    inner = basename.replace('Session_', '').replace('.csv', '')
    # The last segment is the nanosecond timestamp; everything before is the IP
    parts = inner.rsplit('_', 1)
    client_ip = parts[0] if len(parts) == 2 else inner

    # Convert epoch_ns start/end to human-readable
    start_epoch_ns = int(arrival_ns[0])
    end_epoch_ns   = int(arrival_ns[-1])
    start_time_str = datetime.fromtimestamp(start_epoch_ns / 1e9, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    end_time_str   = datetime.fromtimestamp(end_epoch_ns / 1e9, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    return {
        'client_ip': client_ip,
        'start_time': start_time_str,
        'end_time': end_time_str,
        'duration_sec': duration_sec,
        'num_bins': len(bin_results),
        'bins': bin_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregate per-bin metrics → session-level summary (medians)
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_session(session_data):
    """Takes a session dict with 'bins' and returns session-level medians."""
    bins = session_data['bins']
    df = pd.DataFrame(bins)

    result = {
        'client_ip': session_data['client_ip'],
        'start_time': session_data['start_time'],
        'end_time': session_data['end_time'],
        'duration_sec': session_data['duration_sec'],
        'num_bins': session_data['num_bins'],
        # Network metrics
        'median_down_loss_pct': df['down_loss_pct'].median(),
        'mean_down_loss_pct': df['down_loss_pct'].mean(),
        'median_up_loss_pct': df['up_loss_pct'].median(),
        'mean_up_loss_pct': df['up_loss_pct'].mean(),
        'median_down_bps': df['down_bps'].median(),
        'mean_down_bps': df['down_bps'].mean(),
        'median_up_bps': df['up_bps'].median(),
        'mean_up_bps': df['up_bps'].mean(),
        # Audio bitrate
        'median_down_audio_bps': df['down_audio_bps'].median(),
        'mean_down_audio_bps': df['down_audio_bps'].mean(),
        'median_up_audio_bps': df['up_audio_bps'].median(),
        'mean_up_audio_bps': df['up_audio_bps'].mean(),
    }

    # Downlink video metrics — only bins where video was active
    down_video_bins = df[df['down_fps'] > 0]
    if len(down_video_bins) > 0:
        result['median_down_fps'] = down_video_bins['down_fps'].median()
        result['mean_down_fps'] = down_video_bins['down_fps'].mean()
        result['median_down_jitter_ms'] = down_video_bins['down_frame_jitter_ms'].median()
        result['mean_down_jitter_ms'] = down_video_bins['down_frame_jitter_ms'].mean()
        result['median_down_video_bitrate_bps'] = down_video_bins['down_video_bitrate_bps'].median()
        result['mean_down_video_bitrate_bps'] = down_video_bins['down_video_bitrate_bps'].mean()
    else:
        result['median_down_fps'] = 0.0
        result['mean_down_fps'] = 0.0
        result['median_down_jitter_ms'] = 0.0
        result['mean_down_jitter_ms'] = 0.0
        result['median_down_video_bitrate_bps'] = 0.0
        result['mean_down_video_bitrate_bps'] = 0.0

    # Uplink video metrics — only bins where video was active
    up_video_bins = df[df['up_fps'] > 0]
    if len(up_video_bins) > 0:
        result['median_up_fps'] = up_video_bins['up_fps'].median()
        result['mean_up_fps'] = up_video_bins['up_fps'].mean()
        result['median_up_jitter_ms'] = up_video_bins['up_frame_jitter_ms'].median()
        result['mean_up_jitter_ms'] = up_video_bins['up_frame_jitter_ms'].mean()
        result['median_up_video_bitrate_bps'] = up_video_bins['up_video_bitrate_bps'].median()
        result['mean_up_video_bitrate_bps'] = up_video_bins['up_video_bitrate_bps'].mean()
    else:
        result['median_up_fps'] = 0.0
        result['mean_up_fps'] = 0.0
        result['median_up_jitter_ms'] = 0.0
        result['mean_up_jitter_ms'] = 0.0
        result['median_up_video_bitrate_bps'] = 0.0
        result['mean_up_video_bitrate_bps'] = 0.0

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Walk a directory of session CSVs and process all of them
# ─────────────────────────────────────────────────────────────────────────────
def process_platform(session_dir, platform):
    """Process all CSVs in a platform directory. Returns list of session summaries."""
    summaries = []
    csv_files = sorted([f for f in os.listdir(session_dir) if f.endswith('.csv')])
    total = len(csv_files)

    for i, fname in enumerate(csv_files):
        filepath = os.path.join(session_dir, fname)
        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  [{platform}] Processing {i+1}/{total}: {fname}")

        try:
            session = process_session(filepath, platform)
            if session is not None:
                summary = aggregate_session(session)
                summary['filename'] = fname
                summary['platform'] = platform
                summaries.append(summary)
        except Exception as e:
            print(f"  [WARN] Skipped {fname}: {e}")

    return summaries


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 1: Box Plots (2×2 grid)
# ─────────────────────────────────────────────────────────────────────────────
def plot_boxplots(df):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Teams vs Meet — QoE Comparison (Session-Level Medians)',
                 fontsize=18, fontweight='bold', y=0.98)

    teams = df[df['platform'] == 'Teams']
    meet  = df[df['platform'] == 'Meet']

    metrics = [
        ('median_down_loss_pct', 'Downlink Packet Loss (%)', axes[0, 0]),
        ('median_down_jitter_ms', 'Downlink Frame Jitter (ms)', axes[0, 1]),
        ('median_down_fps', 'Downlink FPS', axes[1, 0]),
        ('median_down_bps', 'Downlink Throughput (Kbps)', axes[1, 1]),
    ]

    colors_teams = '#4FC3F7'  # Light blue
    colors_meet  = '#81C784'  # Light green

    for metric_key, title, ax in metrics:
        t_data = teams[metric_key].dropna().values
        m_data = meet[metric_key].dropna().values

        # Convert bps to Kbps for throughput
        if metric_key == 'median_down_bps':
            t_data = t_data / 1000.0
            m_data = m_data / 1000.0

        bp = ax.boxplot(
            [t_data, m_data],
            labels=['Teams', 'Meet'],
            patch_artist=True,
            widths=0.5,
            showfliers=True,
            flierprops=dict(marker='o', markersize=4, alpha=0.5),
        )
        bp['boxes'][0].set_facecolor(colors_teams)
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(colors_meet)
        bp['boxes'][1].set_alpha(0.7)

        for median_line in bp['medians']:
            median_line.set_color('white')
            median_line.set_linewidth(2)

        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.2)

        # Add sample sizes
        ax.text(1, ax.get_ylim()[1] * 0.95, f'n={len(t_data)}',
                ha='center', fontsize=9, color=colors_teams, fontweight='bold')
        ax.text(2, ax.get_ylim()[1] * 0.95, f'n={len(m_data)}',
                ha='center', fontsize=9, color=colors_meet, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('qoe_boxplots.png', dpi=300, bbox_inches='tight')
    print("\n[SAVED] qoe_boxplots.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 2: CDF Curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_cdf(df):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Teams vs Meet — Cumulative Distribution (CDF)',
                 fontsize=18, fontweight='bold', y=1.02)

    teams = df[df['platform'] == 'Teams']
    meet  = df[df['platform'] == 'Meet']

    cdf_metrics = [
        ('median_down_loss_pct', 'Downlink Packet Loss (%)', axes[0]),
        ('median_down_jitter_ms', 'Downlink Frame Jitter (ms)', axes[1]),
    ]

    for metric_key, xlabel, ax in cdf_metrics:
        for label, data, color, ls in [
            ('Teams', teams[metric_key].dropna().sort_values(), '#4FC3F7', '-'),
            ('Meet',  meet[metric_key].dropna().sort_values(),  '#81C784', '--'),
        ]:
            if len(data) == 0:
                continue
            cdf = np.arange(1, len(data) + 1) / len(data)
            ax.plot(data.values, cdf, label=f'{label} (n={len(data)})',
                    color=color, linewidth=2, linestyle=ls)

        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig('qoe_cdf.png', dpi=300, bbox_inches='tight')
    print("[SAVED] qoe_cdf.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 3: Summary Statistics Table
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(df):
    metrics = [
        ('median_down_loss_pct', 'Down Loss %'),
        ('median_up_loss_pct', 'Up Loss %'),
        ('median_down_jitter_ms', 'Down Frame Jitter ms'),
        ('median_up_jitter_ms', 'Up Frame Jitter ms'),
        ('median_down_fps', 'Down FPS'),
        ('median_up_fps', 'Up FPS'),
        ('median_down_bps', 'Down Throughput bps'),
        ('median_up_bps', 'Up Throughput bps'),
        ('median_down_video_bitrate_bps', 'Down Video Bitrate bps'),
        ('median_up_video_bitrate_bps', 'Up Video Bitrate bps'),
        ('median_down_audio_bps', 'Down Audio bps'),
        ('median_up_audio_bps', 'Up Audio bps'),
    ]

    print("\n" + "=" * 100)
    print(f"{'Metric':<25} | {'Platform':<8} | {'Count':>6} | {'Mean':>10} | {'Median':>10} | {'P5':>10} | {'P25':>10} | {'P75':>10} | {'P95':>10}")
    print("-" * 100)

    summary_rows = []
    for metric_key, metric_name in metrics:
        for plat in ['Teams', 'Meet']:
            data = df[df['platform'] == plat][metric_key].dropna()
            if len(data) == 0:
                continue
            row = {
                'Metric': metric_name,
                'Platform': plat,
                'Count': len(data),
                'Mean': f"{data.mean():.2f}",
                'Median': f"{data.median():.2f}",
                'P5': f"{data.quantile(0.05):.2f}",
                'P25': f"{data.quantile(0.25):.2f}",
                'P75': f"{data.quantile(0.75):.2f}",
                'P95': f"{data.quantile(0.95):.2f}",
            }
            print(f"{row['Metric']:<25} | {row['Platform']:<8} | {row['Count']:>6} | {row['Mean']:>10} | {row['Median']:>10} | {row['P5']:>10} | {row['P25']:>10} | {row['P75']:>10} | {row['P95']:>10}")
            summary_rows.append(row)

    print("=" * 100)

    # Save to CSV
    pd.DataFrame(summary_rows).to_csv('qoe_summary.csv', index=False)
    print("[SAVED] qoe_summary.csv")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    base_dir = "Sessions_18_to_19"
    teams_dir = os.path.join(base_dir, "Teams")
    meet_dir  = os.path.join(base_dir, "Meet")

    if not os.path.isdir(teams_dir):
        print(f"[ERROR] Teams directory not found: {teams_dir}")
        sys.exit(1)
    if not os.path.isdir(meet_dir):
        print(f"[ERROR] Meet directory not found: {meet_dir}")
        sys.exit(1)

    print("=" * 70)
    print(" Teams vs Meet QoE Comparison — Offline Analysis")
    print("=" * 70)

    print(f"\n--- Processing Teams Sessions ---")
    teams_summaries = process_platform(teams_dir, "Teams")
    print(f"  → {len(teams_summaries)} qualifying Teams sessions (duration > {MIN_SESSION_DURATION_SEC}s)")

    print(f"\n--- Processing Meet Sessions ---")
    meet_summaries = process_platform(meet_dir, "Meet")
    print(f"  → {len(meet_summaries)} qualifying Meet sessions (duration > {MIN_SESSION_DURATION_SEC}s)")

    # Combine into a single DataFrame
    all_summaries = teams_summaries + meet_summaries
    df = pd.DataFrame(all_summaries)

    # ── Save detailed per-session CSVs ──────────────────────────────────────
    detail_columns = [
        'filename', 'client_ip', 'start_time', 'end_time', 'duration_sec',
        'median_down_loss_pct', 'mean_down_loss_pct',
        'median_up_loss_pct', 'mean_up_loss_pct',
        'median_down_bps', 'mean_down_bps',
        'median_up_bps', 'mean_up_bps',
        'median_down_fps', 'mean_down_fps',
        'median_up_fps', 'mean_up_fps',
        'median_down_jitter_ms', 'mean_down_jitter_ms',
        'median_up_jitter_ms', 'mean_up_jitter_ms',
        'median_down_video_bitrate_bps', 'mean_down_video_bitrate_bps',
        'median_up_video_bitrate_bps', 'mean_up_video_bitrate_bps',
        'median_down_audio_bps', 'mean_down_audio_bps',
        'median_up_audio_bps', 'mean_up_audio_bps',
    ]

    teams_df = df[df['platform'] == 'Teams'][detail_columns].sort_values('duration_sec', ascending=False)
    meet_df  = df[df['platform'] == 'Meet'][detail_columns].sort_values('duration_sec', ascending=False)

    teams_df.to_csv('teams_session_details.csv', index=False)
    print(f"\n[SAVED] teams_session_details.csv ({len(teams_df)} sessions)")

    meet_df.to_csv('meet_session_details.csv', index=False)
    print(f"[SAVED] meet_session_details.csv ({len(meet_df)} sessions)")

    print_summary(df)

    print(f"\n{'=' * 70}")
    print(f" All Done! Check:")
    print(f"   - teams_session_details.csv  (detailed per-session metrics)")
    print(f"   - meet_session_details.csv   (detailed per-session metrics)")
    print(f"   - qoe_summary.csv            (summary statistics table)")
    print(f"{'=' * 70}")
