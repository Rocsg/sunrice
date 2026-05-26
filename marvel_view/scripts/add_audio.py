"""add_audio.py — Add an audio track to a marvel-water-movie MP4.

The video stream is copied without re-encoding.  The audio is looped to fill
the full video duration and cut exactly at the video end.  When ``--vr`` is
given, the Google Spatial-Media spherical XMP metadata (vr360, stereo
left-right) is re-injected after the remux, because ffmpeg silently drops that
custom UUID box during any container operation.

Usage::

    marvel-add-audio-to-movie movie.mp4 music.mp3
    marvel-add-audio-to-movie movie.mp4 music.ogg --output movie_with_audio.mp4
    marvel-add-audio-to-movie vr_movie.mp4 music.mp3 --vr
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_ffmpeg() -> str:
    """Return path to ffmpeg binary (imageio_ffmpeg first, then PATH)."""
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    sys.exit(
        "ffmpeg not found. Install it via your package manager or:\n"
        "  pip install imageio-ffmpeg"
    )


def _inject_vr360_metadata(mp4_path: Path) -> bool:
    """Inject Google Spatial-Media vr360 XMP metadata in-place."""
    try:
        from spatialmedia import metadata_utils  # type: ignore
    except ImportError:
        print(
            "  [warn] spatialmedia not installed — VR metadata not injected.\n"
            "  pip install 'git+https://github.com/google/spatial-media.git'"
        )
        return False

    tagged_path = mp4_path.with_suffix(".tagged.mp4")

    metadata = metadata_utils.Metadata()
    metadata.stereo_mode = "left-right"
    metadata.projection = metadata_utils.generate_spherical_xml(
        stereo="left-right",
    )

    try:
        metadata_utils.inject_metadata(
            str(mp4_path), str(tagged_path), metadata, print,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] spatial-media injection failed: {exc}")
        tagged_path.unlink(missing_ok=True)
        return False

    if not tagged_path.exists() or tagged_path.stat().st_size < 1024:
        print("  [warn] spatial-media produced no output.")
        tagged_path.unlink(missing_ok=True)
        return False

    tagged_path.replace(mp4_path)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="marvel-add-audio-to-movie",
        description=(
            "Add an audio track to a marvel-water-movie MP4 without "
            "re-encoding the video.  The audio loops to fill the video "
            "duration and is cut at the end."
        ),
    )
    parser.add_argument("input_video", help="Input MP4 file.")
    parser.add_argument(
        "audio_file",
        help="Audio file to add (any ffmpeg-supported format: mp3, ogg, wav, …).",
    )
    parser.add_argument(
        "--output", "-o",
        help=(
            "Output MP4 path.  Defaults to <input_stem>_with_audio.mp4 "
            "next to the input file."
        ),
    )
    parser.add_argument(
        "--vr",
        action="store_true",
        help=(
            "Re-inject Google Spatial-Media vr360 stereo metadata after the "
            "remux (required when the source is a VR 360° video, because "
            "ffmpeg strips those metadata during remux)."
        ),
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input_video).resolve()
    audio_path = Path(args.audio_file).resolve()

    if not input_path.exists():
        sys.exit(f"Input video not found: {input_path}")
    if not audio_path.exists():
        sys.exit(f"Audio file not found: {audio_path}")

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = input_path.with_name(
            input_path.stem + "_with_audio" + input_path.suffix
        )

    ffmpeg = _find_ffmpeg()

    # ------------------------------------------------------------------
    # Step 1 — remux: copy video, loop audio, stop at video end
    # ------------------------------------------------------------------
    print(f"Input  : {input_path}")
    print(f"Audio  : {audio_path}")
    print(f"Output : {output_path}")
    print()

    cmd = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-stream_loop", "-1",           # loop audio indefinitely
        "-i", str(audio_path),
        "-c:v", "copy",                 # no video re-encoding
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-map_metadata", "0",           # carry over container metadata
        "-movflags", "+faststart",
        "-shortest",                    # stop at video end
        str(output_path),
    ]

    print("Running ffmpeg…")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(f"ffmpeg failed with exit code {result.returncode}.")

    # ------------------------------------------------------------------
    # Step 2 — re-inject VR metadata if requested
    # ------------------------------------------------------------------
    if args.vr:
        print("\nRe-injecting spatial-media vr360 metadata…")
        ok = _inject_vr360_metadata(output_path)
        if ok:
            print("  VR metadata injected successfully.")
        else:
            print(
                "  VR metadata injection failed — output is still a valid "
                "video but won't be auto-detected as VR by headsets."
            )

    print(f"\nDone: {output_path}")


if __name__ == "__main__":
    main()
