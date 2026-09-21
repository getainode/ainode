"""The ten clips this bench transcribes, and the words they are scored against.

Repo data versioned next to its results, the rule ``bench/decide/items.json`` lives
by, and for a stronger reason here: **word error rate is only comparable over the
same audio**. Two runs of this section can only be read side by side when they heard
the same bytes, so the WAV files are committed under ``bench/speech/clips/`` (1.3 MB
for the set) rather than synthesised per run, and ``CLIPS_VERSION`` is bumped on any
edit to a text, a voice or a file. A record carries both the id and the version.

The manifest below is the ground truth in both directions: it is the text handed to
``say`` to make each clip, so it is exactly what was spoken, and it is the reference
the transcript is scored against. Nothing is transcribed by hand and nothing is
adjusted after a run: a reference edited to match what a model said would make the
error rate a statement about the editor.

How the files were made, and how ``--generate-clips`` remakes them:

    say -v <voice> -o <id>.aiff "<text>"
    afconvert -f WAVE -d LEI16@16000 -c 1 <id>.aiff <id>.wav

Mono 16-bit PCM at 16 kHz because that is the rate Whisper's own front end resamples
to, so the clip carries no resampling the engine did not ask for, and WAV because the
engine decodes it with soundfile and libsndfile reads PCM WAV with no codec.

Six voices across six English locales (US, GB, AU, IE, IN, ZA), because an accent is
the thing a speech model is actually hard on and a set in one voice measures one path
through it. Every line is 11 to 15 words, every line carries a digit, and seven of the
ten carry a place or product name, because those are the two places transcription goes
wrong in a way a user notices: a wrong number and a wrong name.

Regenerating them needs macOS (``say`` and ``afconvert`` ship with it). A run on any
other machine reads the committed files, which is the normal path: the bench does not
need a synthesiser, and a set rebuilt on a different macOS release would be different
audio under the same id, which is what ``CLIPS_VERSION`` exists to stop.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import wave

#: Bumped when a text, a voice or a WAV changes. A record carries it so two runs are
#: never compared across a clip edit.
CLIPS_VERSION = 1
CLIPS_ID = "say-10"

#: Environment override for the clip directory, the same seam
#: ``$AINODE_HARNESS_TASKS`` and ``$AINODE_DECIDE_ITEMS`` give their data.
ENV_CLIPS_DIR = "AINODE_SPEECH_CLIPS"

_REPO_CLIPS = (pathlib.Path(__file__).resolve().parents[3] / "bench" / "speech"
               / "clips")
_HOME_CLIPS = pathlib.Path.home() / ".ainode" / "bench" / "speech" / "clips"

#: The manifest: ``(id, voice, locale, text)``. The text is both what ``say`` was
#: given and the reference the transcript is scored against.
CLIPS = (
    ("clip-01", "Samantha", "en_US",
     "The train from Austin arrives at platform 9 at half past 6."),
    ("clip-02", "Daniel", "en_GB",
     "She bought 14 tickets to the opera in Dallas for Friday evening."),
    ("clip-03", "Karen", "en_AU",
     "A single graphics processor here holds 128 gigabytes of unified memory."),
    ("clip-04", "Moira", "en_IE",
     "Rain is forecast for Houston for 4 days, from Monday through Thursday."),
    ("clip-05", "Rishi", "en_IN",
     "The library in Dublin keeps its 40 oldest maps in a cold room."),
    ("clip-06", "Tessa", "en_ZA",
     "Add 2 cups of flour, a pinch of salt and 3 eggs to the bowl."),
    ("clip-07", "Samantha", "en_US",
     "Nvidia released the driver on September 19 after a long delay."),
    ("clip-08", "Daniel", "en_GB",
     "He learned to sail on a lake in the mountains 60 miles above Denver."),
    ("clip-09", "Rishi", "en_IN",
     "The bridge over the Colorado river was closed for 3 weeks of repairs in October."),
    ("clip-10", "Tessa", "en_ZA",
     "Our cluster has 4 nodes and 487 gigabytes of memory in total right now."),
)


class ClipError(RuntimeError):
    """A run that cannot start: no clip directory, a missing or unreadable WAV.

    A load error rather than a skipped clip, the rule ``bench/decide/items.json``
    loads under: a run that quietly dropped three clips would report an error rate
    over seven of them under a header claiming ten.
    """


def clips_dir(override: str = "") -> pathlib.Path:
    """``--clips`` if given, else ``$AINODE_SPEECH_CLIPS``, else the repo's set,
    else the installed copy under ``~/.ainode``."""
    if override:
        return pathlib.Path(override).expanduser()
    from_env = os.environ.get(ENV_CLIPS_DIR)
    if from_env:
        return pathlib.Path(from_env).expanduser()
    if _REPO_CLIPS.is_dir():
        return _REPO_CLIPS
    return _HOME_CLIPS


def clips_name(path: pathlib.Path) -> str:
    """How a record names the clip directory: repo-relative when it is the repo's.

    A record that named an absolute path would say more about the machine that ran
    the bench than about the audio, so the repo's own set is written
    ``bench/speech/clips``.
    """
    try:
        return str(path.resolve().relative_to(_REPO_CLIPS.parents[2]))
    except ValueError:
        return path.name


def wav_duration_seconds(path: pathlib.Path) -> float:
    """Clip length off the WAV header: frames over the frame rate.

    Read rather than stated in the manifest, because the manifest holds the text and
    the audio holds the duration, and a real-time factor computed from a number
    somebody typed is not a measurement.
    """
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        if not rate:
            raise ClipError(f"{path.name} reports a frame rate of 0")
        return handle.getnframes() / float(rate)


def wav_shape(path: pathlib.Path) -> dict:
    """``channels``, ``sample_rate``, ``sample_width`` and ``seconds`` of one clip."""
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.getnframes()
        shape = {"channels": handle.getnchannels(),
                 "sample_rate": rate,
                 "sample_width_bytes": handle.getsampwidth(),
                 "frames": frames}
    shape["seconds"] = round(frames / float(rate), 3) if rate else None
    return shape


def load_clips(directory=None, manifest=CLIPS) -> list:
    """Every clip as ``{id, voice, locale, text, path, seconds, bytes, shape}``.

    Strict: a manifest entry whose WAV is missing, empty or unreadable raises rather
    than being dropped. The order is the manifest's, so two runs walk the clips in
    the same order and a per-clip table lines up between records.
    """
    directory = pathlib.Path(directory) if directory else clips_dir()
    if not directory.is_dir():
        raise ClipError(
            f"clip directory {directory} not found; set ${ENV_CLIPS_DIR}, pass "
            "--clips, or run from a checkout that has bench/speech/clips/ (rebuild "
            "the set on a Mac with --generate-clips)")
    loaded = []
    for clip_id, voice, locale, text in manifest:
        path = directory / f"{clip_id}.wav"
        if not path.is_file():
            raise ClipError(f"{path} is missing; regenerate the set with "
                            "--generate-clips on a Mac")
        size = path.stat().st_size
        if size <= 44:
            raise ClipError(f"{path} carries no audio ({size} bytes)")
        try:
            shape = wav_shape(path)
        except ClipError:
            raise
        except Exception as exc:
            raise ClipError(f"{path} is not a readable WAV: {exc}") from exc
        loaded.append({"id": clip_id, "voice": voice, "locale": locale,
                       "text": text, "path": path, "bytes": size,
                       "seconds": shape["seconds"], "shape": shape})
    return loaded


def clips_block(loaded: list, directory=None) -> dict:
    """What the record says about the audio, so a reader can tell two runs apart."""
    directory = (pathlib.Path(directory) if directory
                 else (loaded[0]["path"].parent if loaded else clips_dir()))
    seconds = [c["seconds"] for c in loaded if c.get("seconds") is not None]
    rates = sorted({c["shape"]["sample_rate"] for c in loaded if c.get("shape")})
    return {
        "id": CLIPS_ID,
        "version": CLIPS_VERSION,
        "clips": len(loaded),
        "voices": sorted({c["voice"] for c in loaded}),
        "locales": sorted({c["locale"] for c in loaded}),
        "reference_words": sum(len(c["text"].split()) for c in loaded),
        "audio_seconds": round(sum(seconds), 3) if seconds else None,
        "sample_rates": rates,
        "directory": clips_name(directory),
        "source": "ainode/bench/speech/clips.py",
    }


def generate_clips(directory, manifest=CLIPS, say=None) -> list:
    """Rebuild the WAVs from the manifest with ``say`` and ``afconvert``. macOS only.

    Only ``--generate-clips`` calls this, and a bench run never does: regenerating
    audio mid-run would change what the numbers were taken over. Returns the paths
    written. ``say`` is the subprocess runner, injected so a test can pin the exact
    argv without a synthesiser on the machine.
    """
    runner = say or (lambda argv: subprocess.run(argv, check=True))
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for clip_id, voice, _locale, text in manifest:
        aiff = directory / f"{clip_id}.aiff"
        wav = directory / f"{clip_id}.wav"
        runner(["say", "-v", voice, "-o", str(aiff), text])
        runner(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                str(aiff), str(wav)])
        try:
            aiff.unlink()
        except FileNotFoundError:
            pass
        written.append(wav)
    return written


__all__ = ["CLIPS", "CLIPS_ID", "CLIPS_VERSION", "ClipError", "ENV_CLIPS_DIR",
           "clips_block", "clips_dir", "clips_name", "generate_clips",
           "load_clips", "wav_duration_seconds", "wav_shape"]
