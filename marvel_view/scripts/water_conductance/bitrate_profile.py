#!/usr/bin/env python3
import argparse
import csv
import math
import os
import statistics
import subprocess
import sys


def run_ffprobe_duration(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def iter_packets(path: str, stream: str, time_field: str):
    # stream: v:0, a:0, etc.
    # time_field: pts_time or dts_time
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", stream,
        "-show_packets",
        "-show_entries", f"packet=pts_time,dts_time,duration_time,size,flags",
        "-of", "csv=p=0",
        path,
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue

        parts = line.split(",")

        # Expected columns:
        # pts_time,dts_time,duration_time,size,flags
        # But ffprobe can omit N/A fields, so be defensive.
        if len(parts) < 4:
            continue

        pts_s, dts_s, dur_s, size_s = parts[:4]

        def to_float(x):
            try:
                if x == "N/A":
                    return None
                return float(x)
            except Exception:
                return None

        pts = to_float(pts_s)
        dts = to_float(dts_s)
        dur = to_float(dur_s)

        try:
            size = int(size_s)
        except Exception:
            continue

        t = pts if time_field == "pts" else dts
        if t is None:
            t = dts if time_field == "pts" else pts
        if t is None:
            continue

        # DTS can be slightly negative at the beginning with B-frames.
        if t < 0:
            t = 0.0

        yield t, dur, size

    _, err = proc.communicate()
    if proc.returncode != 0:
        print(err, file=sys.stderr)
        raise RuntimeError("ffprobe failed")


def summarize(values):
    values = [v for v in values if v is not None]
    if not values:
        return {}

    values_sorted = sorted(values)

    def percentile(p):
        if len(values_sorted) == 1:
            return values_sorted[0]
        k = (len(values_sorted) - 1) * p / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return values_sorted[int(k)]
        return values_sorted[f] * (c - k) + values_sorted[c] * (k - f)

    return {
        "min": min(values_sorted),
        "mean": statistics.mean(values_sorted),
        "median": statistics.median(values_sorted),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": max(values_sorted),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute local bitrate profile of an MP4 using ffprobe packets."
    )
    parser.add_argument("video", help="Input video file, e.g. /home/rfernandez/Bureau/RC3.mp4")
    parser.add_argument("--bin", type=float, default=1.0, help="Window size in seconds. Default: 1.0")
    parser.add_argument("--stream", default="v:0", help="Stream selector. Default: v:0")
    parser.add_argument("--time", choices=["pts", "dts"], default="pts",
                        help="Use pts for visual timeline, dts for bitstream order. Default: pts")
    parser.add_argument("--top", type=int, default=20, help="Number of highest peaks to print. Default: 20")
    parser.add_argument("--csv", default=None, help="Output CSV path. Default: <video>.bitrate_<bin>s.csv")
    args = parser.parse_args()

    path = args.video
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    duration = run_ffprobe_duration(path)
    bin_s = args.bin
    n_bins = int(math.ceil(duration / bin_s))

    bytes_per_bin = [0] * n_bins

    packet_count = 0
    total_bytes = 0

    for t, dur, size in iter_packets(path, args.stream, args.time):
        idx = int(t // bin_s)
        if 0 <= idx < n_bins:
            bytes_per_bin[idx] += size
            packet_count += 1
            total_bytes += size

    rows = []
    kbps_values = []

    for i, b in enumerate(bytes_per_bin):
        start = i * bin_s
        end = min((i + 1) * bin_s, duration)
        effective_duration = max(end - start, 1e-9)
        kbps = (b * 8.0) / effective_duration / 1000.0
        mbps = kbps / 1000.0
        kbps_values.append(kbps)
        rows.append({
            "bin": i,
            "start_s": start,
            "end_s": end,
            "bytes": b,
            "kbps": kbps,
            "mbps": mbps,
        })

    stats = summarize(kbps_values)

    print()
    print(f"File      : {path}")
    print(f"Stream    : {args.stream}")
    print(f"Duration  : {duration:.3f} s")
    print(f"Bin size  : {bin_s:.3f} s")
    print(f"Packets   : {packet_count}")
    print(f"Data      : {total_bytes / 1024 / 1024:.2f} MiB in selected stream")
    print()

    print("Local bitrate over bins:")
    print(f"  min     : {stats['min']:.1f} kb/s  ({stats['min']/1000:.2f} Mb/s)")
    print(f"  mean    : {stats['mean']:.1f} kb/s  ({stats['mean']/1000:.2f} Mb/s)")
    print(f"  median  : {stats['median']:.1f} kb/s  ({stats['median']/1000:.2f} Mb/s)")
    print(f"  p90     : {stats['p90']:.1f} kb/s  ({stats['p90']/1000:.2f} Mb/s)")
    print(f"  p95     : {stats['p95']:.1f} kb/s  ({stats['p95']/1000:.2f} Mb/s)")
    print(f"  p99     : {stats['p99']:.1f} kb/s  ({stats['p99']/1000:.2f} Mb/s)")
    print(f"  max     : {stats['max']:.1f} kb/s  ({stats['max']/1000:.2f} Mb/s)")
    print()

    print(f"Top {args.top} peaks:")
    peaks = sorted(rows, key=lambda r: r["kbps"], reverse=True)[:args.top]
    for r in peaks:
        print(
            f"  {r['start_s']:8.2f} - {r['end_s']:8.2f} s : "
            f"{r['mbps']:7.2f} Mb/s"
        )

    csv_path = args.csv
    if csv_path is None:
        base = os.path.splitext(path)[0]
        csv_path = f"{base}.bitrate_{bin_s:g}s.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bin", "start_s", "end_s", "bytes", "kbps", "mbps"])
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"CSV written: {csv_path}")


if __name__ == "__main__":
    main()