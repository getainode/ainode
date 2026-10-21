"""On-disk retention for the figures the collector reports.

Everything the collector knew lived in memory. A restart reset the counters, the
GPU series and the latency percentiles to nothing, the dashboard drew its first
point from whatever the node happened to be doing a second after boot, and
because nothing scraped the Prometheus endpoint there was no copy of the history
anywhere else either. A node that had been serving for a week could not answer
"was it this hot yesterday".

This module keeps a small SQLite file at ``<AINODE_HOME>/metrics.db``:

* ``samples`` holds one row per series per tick, ``(ts, series, value)``, with
  ``value`` NULLABLE on purpose. A figure the node could not measure is written
  as NULL and read back as ``null``. It is never interpolated, never carried
  forward from the previous tick and never written as 0, which is the rule the
  rest of the telemetry already follows (root AGENTS.md, ``optional_float``): a
  zero is a measurement, and a chart draws it as one.
* ``samples_1m`` holds one row per series per minute with the mean, the min, the
  max and how many of that minute's samples carried a value at all. It is built
  forward from a watermark in ``meta``, so a pass costs the minutes since the
  last pass and not a scan of the whole table.

Retention defaults to 48 hours raw and 30 days rolled up, both configurable in
the ``metrics`` block of ``config.json`` (see :class:`MetricsSettings`).

Measured, not estimated, at the defaults (15 series, a sample every 15 seconds,
SQLite 3.53, VACUUMed so the WAL is not double counted): 86400 raw rows a day at
about 72 bytes each is 5.9 MB, and 21600 roll-up rows a day at about 93 bytes
each is 1.9 MB. Steady state is therefore roughly 12 MB of raw plus 58 MB of
roll-ups, about 70 MB per node, and it stops growing there because a pruned page
is reused by the next insert.

Writes never raise at the caller. A read-only or full disk logs one warning, the
store marks itself degraded, and the server keeps serving: ``/api/metrics`` and
``/metrics`` answer exactly what they answered before this module existed.
Losing a metrics sample is not a reason to take a node down, and the degraded
flag is on ``stats()`` and on ``/metrics`` so the loss is visible rather than
silent.

Stdlib ``sqlite3`` only, WAL mode. A node is one process writing a handful of
rows every 15 seconds; WAL lets the history route read while the sampler writes,
and lets an operator open the file with the ``sqlite3`` CLI on a live node
without blocking it.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from ainode.metrics.collector import optional_float

logger = logging.getLogger(__name__)

#: Seconds between samples when nothing says otherwise. 15 s is four points a
#: minute, which is one roll-up row's worth of detail and about 6 MB of raw
#: samples a day across the series below.
DEFAULT_INTERVAL_SECONDS = 15.0

#: How long the raw per-tick samples are kept.
DEFAULT_RETENTION_HOURS = 48

#: How long the one-minute roll-ups are kept.
DEFAULT_RETENTION_DAYS = 30

#: Roll-up bucket width in seconds. Deliberately not configurable: the schema,
#: the watermark and the history route all say "one minute" in their own terms,
#: and a knob here would need all three to agree at read time on a value that
#: may have changed since the rows were written.
BUCKET_SECONDS = 60

#: Most points one history response will return. A caller asking for a month at
#: a 15 second step gets a coarser step and is told which in ``step``, rather
#: than 172800 points no chart can draw.
MAX_HISTORY_POINTS = 5000

#: Default history window when the caller names neither end.
DEFAULT_WINDOW_SECONDS = 3600

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS samples (
        ts     REAL NOT NULL,
        series TEXT NOT NULL,
        value  REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS samples_series_ts ON samples (series, ts)",
    """
    CREATE TABLE IF NOT EXISTS samples_1m (
        series     TEXT    NOT NULL,
        bucket     INTEGER NOT NULL,
        mean_value REAL,
        min_value  REAL,
        max_value  REAL,
        measured   INTEGER NOT NULL,
        missing    INTEGER NOT NULL,
        PRIMARY KEY (series, bucket)
    )
    """,
    "CREATE INDEX IF NOT EXISTS samples_1m_bucket ON samples_1m (bucket)",
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
)

_WATERMARK_KEY = "downsample_watermark"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricsSettings:
    """The ``metrics`` block of ``config.json``, validated.

    ``enabled`` is True by default: about 70 MB at steady state (measured, see
    the module docstring), a write of fifteen rows every fifteen seconds, and a
    node that cannot say what it was doing an hour ago is the problem this exists
    to fix. Turning it off is for a node with a read-only or precious data
    directory.
    """

    enabled: bool = True
    retention_hours: int = DEFAULT_RETENTION_HOURS
    retention_days: int = DEFAULT_RETENTION_DAYS
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS

    @classmethod
    def from_mapping(cls, raw: Any) -> "MetricsSettings":
        """Build settings from a plain mapping, clamping nonsense.

        A config file is hand-edited, so every value here is treated as a
        suggestion: a missing key keeps the default, an unparseable one keeps
        the default, and a retention or interval below one unit is raised to one
        rather than turning the sampler into a busy loop or the retention into
        "delete everything on the next pass".
        """
        if not isinstance(raw, Mapping):
            return cls()

        def _int(key: str, default: int) -> int:
            value = optional_float(raw.get(key))
            if value is None:
                return default
            return max(1, int(value))

        interval = optional_float(raw.get("interval_seconds"))
        if interval is None:
            interval = DEFAULT_INTERVAL_SECONDS
        enabled = raw.get("enabled", True)
        return cls(
            enabled=True if enabled is None else bool(enabled),
            retention_hours=_int("retention_hours", DEFAULT_RETENTION_HOURS),
            retention_days=_int("retention_days", DEFAULT_RETENTION_DAYS),
            interval_seconds=max(1.0, float(interval)),
        )

    @classmethod
    def from_config(cls, config: Any) -> "MetricsSettings":
        """Read the ``metrics`` block off a :class:`NodeConfig`-shaped object."""
        return cls.from_mapping(getattr(config, "metrics", None))

    @property
    def raw_seconds(self) -> float:
        """Raw retention as seconds."""
        return self.retention_hours * 3600.0

    @property
    def rollup_seconds(self) -> float:
        """Roll-up retention as seconds."""
        return self.retention_days * 86400.0


def default_store_path() -> Path:
    """``<AINODE_HOME>/metrics.db``, read at call time.

    Read at call time and not at import, because ``AINODE_HOME`` comes from the
    environment and the test suite and the container set it differently from a
    developer's shell.
    """
    from ainode.core import config as core_config

    return Path(core_config.AINODE_HOME) / "metrics.db"


# ---------------------------------------------------------------------------
# What gets stored
# ---------------------------------------------------------------------------

#: GPU keys taken straight off ``MetricsCollector.get_gpu_metrics()``.
GPU_KEYS = (
    "gpu_count",
    "utilization_percent",
    "memory_used_mb",
    "memory_total_mb",
    "temperature_c",
    "engine_reserved_fraction",
    "system_memory_used_mb",
)

#: Request keys taken straight off the snapshot's ``requests`` block.
REQUEST_KEYS = (
    "total",
    "errors",
    "tokens_generated",
    "tokens_per_second",
)

#: Latency percentiles, stored only once a request has actually been timed.
LATENCY_KEYS = ("p50", "p95", "p99")


def series_from_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Optional[float]]:
    """Flatten one ``get_snapshot()`` into ``{series name: value or None}``.

    A pure function of the snapshot so a test can pin every name and every None
    without a GPU, a clock or a database.

    Two places where None is the honest answer and 0 would be a claim:

    * A GPU block carrying an ``error`` means NVML would not answer. Every GPU
      series for that tick is None, not a row of zeros that a chart draws as an
      idle, empty, cold node. A block marked ``stale`` is the same answer: it is
      the last sample the collector managed, served while a slow driver read is
      still in flight (#238), and storing it as THIS tick's value would be the
      previous tick carried forward, which this file exists to refuse.
    * The latency percentiles read 0 on a collector that has timed nothing,
      because that is the shape ``/api/metrics`` has always returned and this
      module does not change it. Stored, that 0 would be a claim that requests
      are answering instantly. With no request counted, the percentiles are
      None.
    """
    out: dict[str, Optional[float]] = {}

    gpu = snapshot.get("gpu")
    readable = (isinstance(gpu, Mapping) and "error" not in gpu
                and not gpu.get("stale"))
    for key in GPU_KEYS:
        out[f"gpu.{key}"] = optional_float(gpu.get(key)) if readable else None

    requests = snapshot.get("requests")
    requests = requests if isinstance(requests, Mapping) else {}
    for key in REQUEST_KEYS:
        out[f"requests.{key}"] = optional_float(requests.get(key))

    latency = requests.get("latency_ms")
    latency = latency if isinstance(latency, Mapping) else {}
    timed = bool(optional_float(requests.get("total")))
    for key in LATENCY_KEYS:
        out[f"requests.latency_ms.{key}"] = optional_float(latency.get(key)) if timed else None

    # Uptime is what makes a restart visible in the history: the series sawtooths
    # back to near zero at the moment the process came up, so a gap in the other
    # series can be read as a restart rather than as a dead GPU.
    out["uptime_seconds"] = optional_float(snapshot.get("uptime_seconds"))
    return out


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

class MetricsStore:
    """SQLite-backed sample store. Never raises at the caller.

    One connection with ``check_same_thread=False`` behind one lock, because the
    two callers are the sampler thread (writing a few rows every 15 seconds) and
    the API server (reading a window). Per-thread connections would buy
    concurrency this volume cannot use, and would make the roll-up watermark a
    cross-connection race.
    """

    def __init__(
        self,
        path: Any = None,
        settings: Optional[MetricsSettings] = None,
    ) -> None:
        self.settings = settings or MetricsSettings()
        self.path = Path(path) if path is not None else default_store_path()
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._degraded = False
        self._failure_logged = False
        if self.settings.enabled:
            self._open()

    # -- lifecycle ---------------------------------------------------------

    def _open(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # isolation_level=None is autocommit: PRAGMA and DDL run outside a
            # transaction, and every write below opens its own explicit one.
            conn = sqlite3.connect(
                str(self.path),
                check_same_thread=False,
                timeout=5.0,
                isolation_level=None,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            for statement in _SCHEMA:
                conn.execute(statement)
            self._conn = conn
        except Exception:
            self._conn = None
            self._fail("could not open the metrics store")

    def close(self) -> None:
        """Close the connection. Safe to call twice."""
        with self._lock:
            conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    @property
    def available(self) -> bool:
        """True when there is an open database to write to."""
        return self._conn is not None

    @property
    def degraded(self) -> bool:
        """True once any read or write has failed."""
        return self._degraded

    def _fail(self, message: str) -> None:
        """Mark the store degraded, logging the first failure only.

        Once per process, on purpose. A full disk fails on every tick, and four
        warnings a minute for the rest of the node's uptime would bury whatever
        the operator is actually looking for. The continuing state is reported
        by ``stats()["degraded"]`` and by ``ainode_metrics_retention_degraded``
        on ``/metrics``, so it stays visible without being repeated.
        """
        self._degraded = True
        if not self._failure_logged:
            self._failure_logged = True
            logger.warning(
                "metrics retention: %s at %s. Samples are not being kept; "
                "/api/metrics and /metrics are unaffected.",
                message,
                self.path,
                exc_info=True,
            )

    # -- writing -----------------------------------------------------------

    def write(
        self,
        samples: Mapping[str, Optional[float]],
        ts: Optional[float] = None,
    ) -> int:
        """Store one tick. Returns the number of rows written, 0 on failure.

        A None value is stored as SQL NULL. Nothing else in this class will ever
        turn that back into a number.
        """
        if not self.available or not samples:
            return 0
        stamp = time.time() if ts is None else float(ts)
        rows = [
            (stamp, str(name), optional_float(value))
            for name, value in samples.items()
        ]
        try:
            with self._lock:
                conn = self._conn
                if conn is None:
                    return 0
                conn.execute("BEGIN")
                conn.executemany(
                    "INSERT INTO samples (ts, series, value) VALUES (?, ?, ?)", rows
                )
                conn.execute("COMMIT")
            return len(rows)
        except Exception:
            self._rollback()
            self._fail("could not write samples")
            return 0

    def _rollback(self) -> None:
        try:
            with self._lock:
                if self._conn is not None:
                    self._conn.execute("ROLLBACK")
        except Exception:
            pass

    # -- roll-up and retention ---------------------------------------------

    def downsample(self, now: Optional[float] = None) -> int:
        """Roll complete minutes up into ``samples_1m``. Returns rows written.

        Incremental: only the minutes after the stored watermark and before the
        current, still-filling minute are touched, so the cost is the time since
        the last pass. A minute where every sample was None gets NULL mean, min
        and max with ``measured = 0``: the roll-up records that nothing was
        measured, it does not invent a figure for the gap.
        """
        if not self.available:
            return 0
        stamp = time.time() if now is None else float(now)
        # Only complete minutes. The minute in progress is rolled up next pass,
        # otherwise its first sample would be published as the whole minute.
        cutoff = int(stamp // BUCKET_SECONDS) * BUCKET_SECONDS
        try:
            with self._lock:
                conn = self._conn
                if conn is None:
                    return 0
                watermark = self._get_watermark(conn)
                if watermark is None:
                    row = conn.execute("SELECT MIN(ts) FROM samples").fetchone()
                    if row is None or row[0] is None:
                        return 0
                    start = int(float(row[0]) // BUCKET_SECONDS) * BUCKET_SECONDS
                else:
                    start = watermark + BUCKET_SECONDS
                if start >= cutoff:
                    return 0
                conn.execute("BEGIN")
                cursor = conn.execute(
                    """
                    INSERT INTO samples_1m
                        (series, bucket, mean_value, min_value, max_value,
                         measured, missing)
                    SELECT series,
                           CAST(ts / ? AS INTEGER) * ?,
                           AVG(value), MIN(value), MAX(value),
                           COUNT(value), COUNT(*) - COUNT(value)
                      FROM samples
                     WHERE ts >= ? AND ts < ?
                     GROUP BY series, CAST(ts / ? AS INTEGER)
                    ON CONFLICT(series, bucket) DO UPDATE SET
                        mean_value = excluded.mean_value,
                        min_value  = excluded.min_value,
                        max_value  = excluded.max_value,
                        measured   = excluded.measured,
                        missing    = excluded.missing
                    """,
                    (BUCKET_SECONDS, BUCKET_SECONDS, start, cutoff, BUCKET_SECONDS),
                )
                written = cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (_WATERMARK_KEY, str(cutoff - BUCKET_SECONDS)),
                )
                conn.execute("COMMIT")
            return written
        except Exception:
            self._rollback()
            self._fail("could not roll samples up")
            return 0

    def prune(self, now: Optional[float] = None) -> dict[str, int]:
        """Drop samples past their retention. Returns what was deleted."""
        if not self.available:
            return {"samples": 0, "downsampled": 0}
        stamp = time.time() if now is None else float(now)
        raw_floor = stamp - self.settings.raw_seconds
        rollup_floor = stamp - self.settings.rollup_seconds
        try:
            with self._lock:
                conn = self._conn
                if conn is None:
                    return {"samples": 0, "downsampled": 0}
                conn.execute("BEGIN")
                raw = conn.execute(
                    "DELETE FROM samples WHERE ts < ?", (raw_floor,)
                ).rowcount
                rolled = conn.execute(
                    "DELETE FROM samples_1m WHERE bucket < ?", (rollup_floor,)
                ).rowcount
                conn.execute("COMMIT")
            return {
                "samples": max(0, raw or 0),
                "downsampled": max(0, rolled or 0),
            }
        except Exception:
            self._rollback()
            self._fail("could not prune samples")
            return {"samples": 0, "downsampled": 0}

    def maintain(self, now: Optional[float] = None) -> dict[str, int]:
        """Roll up, then prune. In that order, always.

        Pruning first would delete raw rows the roll-up had not read yet, and the
        30 day series would have a 48 hour hole at the front of every restart
        that happened to land after a long sampler outage.
        """
        rolled = self.downsample(now)
        pruned = self.prune(now)
        return {"downsampled": rolled, **{f"pruned_{k}": v for k, v in pruned.items()}}

    # -- reading -----------------------------------------------------------

    def series_names(self) -> list[str]:
        """Every series name the store holds, raw or rolled up."""
        if not self.available:
            return []
        try:
            with self._lock:
                conn = self._conn
                if conn is None:
                    return []
                rows = conn.execute(
                    "SELECT series FROM samples_1m "
                    "UNION SELECT series FROM samples ORDER BY 1"
                ).fetchall()
            return [str(r[0]) for r in rows]
        except Exception:
            self._fail("could not list series")
            return []

    def resolve_window(
        self,
        since: Optional[float] = None,
        until: Optional[float] = None,
        step: Optional[float] = None,
        resolution: Optional[str] = None,
        now: Optional[float] = None,
        max_points: int = MAX_HISTORY_POINTS,
    ) -> tuple[float, float, float, str]:
        """Settle the window, the step and which table answers it.

        Returns ``(since, until, step, resolution)`` with ``since`` aligned down
        to a step boundary, so a chart's x values stay put between polls instead
        of sliding by a fraction of a bucket each time.

        The table is chosen by the range, not by the caller's taste: a window
        reaching back past the raw retention can only be answered by the
        roll-ups, and a step of a minute or more has no use for raw rows.
        """
        stamp = time.time() if now is None else float(now)
        end = stamp if until is None else float(until)
        start = end - DEFAULT_WINDOW_SECONDS if since is None else float(since)
        if start > end:
            start, end = end, start
        span = max(end - start, 1.0)

        chosen = (resolution or "").strip().lower()
        if chosen not in ("raw", "1m"):
            reaches_past_raw = start < stamp - self.settings.raw_seconds
            coarse_step = step is not None and float(step) >= BUCKET_SECONDS
            chosen = "1m" if (reaches_past_raw or coarse_step) else "raw"

        if step is None or float(step) <= 0:
            interval = max(1.0, float(self.settings.interval_seconds))
            resolved = interval if chosen == "raw" else float(BUCKET_SECONDS)
        else:
            resolved = float(step)
        if chosen == "1m":
            resolved = max(resolved, float(BUCKET_SECONDS))

        # Keep the response drawable. Coarsening the step is honest (the
        # response says what step it used); truncating the window silently is
        # not, because the caller would think it had the range it asked for.
        budget = max(1, int(max_points))
        if span / resolved > budget:
            resolved = span / budget
            if chosen == "1m":
                resolved = max(resolved, float(BUCKET_SECONDS))

        aligned = (start // resolved) * resolved
        return aligned, end, resolved, chosen

    def history(
        self,
        series: Optional[Iterable[str]] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        step: Optional[float] = None,
        resolution: Optional[str] = None,
        now: Optional[float] = None,
        max_points: Optional[int] = None,
    ) -> dict[str, Any]:
        """One evenly spaced grid per series, with None where nothing was measured.

        Every series gets the same number of points at the same timestamps, so a
        caller can zip them without reindexing. A slot holding no sample, and a
        slot whose samples were all None, both come back ``value: None``: the
        store does not distinguish "the sampler was down" from "the driver would
        not say" by making one of them a number.

        *max_points* is a budget for the WHOLE response, split across the series
        asked for, because a caller that names no series asks for all fifteen and
        a per-series cap would answer with a payload of 75000 points.
        """
        names = list(series) if series is not None else self.series_names()
        budget = MAX_HISTORY_POINTS if max_points is None else max(1, int(max_points))
        per_series = max(1, budget // max(1, len(names)))
        start, end, resolved, chosen = self.resolve_window(
            since, until, step, resolution, now, max_points=per_series
        )
        slots = max(1, int((end - start) / resolved) + 1)
        payload: dict[str, Any] = {
            "since": start,
            "until": end,
            "step": resolved,
            "resolution": chosen,
            "points": slots,
            "series": {},
        }
        for name in names:
            payload["series"][str(name)] = self._series_grid(
                str(name), start, end, resolved, slots, chosen
            )
        return payload

    def _series_grid(
        self,
        name: str,
        start: float,
        end: float,
        step: float,
        slots: int,
        resolution: str,
    ) -> list[dict[str, Any]]:
        found = self._aggregate(name, start, end, step, resolution)
        grid: list[dict[str, Any]] = []
        for index in range(slots):
            point: dict[str, Any] = {"ts": start + index * step, "value": None}
            row = found.get(index)
            if row is not None:
                point.update(row)
            elif resolution == "1m":
                point["min"] = None
                point["max"] = None
                point["samples"] = 0
            grid.append(point)
        return grid

    def _aggregate(
        self,
        name: str,
        start: float,
        end: float,
        step: float,
        resolution: str,
    ) -> dict[int, dict[str, Any]]:
        """Group one series into step-wide slots. Empty on any failure."""
        if not self.available:
            return {}
        try:
            with self._lock:
                conn = self._conn
                if conn is None:
                    return {}
                if resolution == "1m":
                    rows = conn.execute(
                        """
                        SELECT CAST((bucket - ?) / ? AS INTEGER) AS slot,
                               SUM(mean_value * measured), SUM(measured),
                               MIN(min_value), MAX(max_value)
                          FROM samples_1m
                         WHERE series = ? AND bucket >= ? AND bucket <= ?
                         GROUP BY slot
                        """,
                        (start, step, name, start, end),
                    ).fetchall()
                    out: dict[int, dict[str, Any]] = {}
                    for slot, weighted, measured, low, high in rows:
                        count = int(measured or 0)
                        total = optional_float(weighted)
                        # Weighted by how many raw samples each minute actually
                        # measured, so a minute holding one reading does not
                        # weigh as much as a full one.
                        mean = total / count if count and total is not None else None
                        out[int(slot)] = {
                            "value": mean,
                            "min": optional_float(low),
                            "max": optional_float(high),
                            "samples": count,
                        }
                    return out

                rows = conn.execute(
                    """
                    SELECT CAST((ts - ?) / ? AS INTEGER) AS slot,
                           AVG(value), COUNT(value)
                      FROM samples
                     WHERE series = ? AND ts >= ? AND ts <= ?
                     GROUP BY slot
                    """,
                    (start, step, name, start, end),
                ).fetchall()
                return {
                    int(slot): {
                        "value": optional_float(mean) if int(count or 0) else None
                    }
                    for slot, mean, count in rows
                }
        except Exception:
            self._fail("could not read history")
            return {}

    # -- self-reporting ----------------------------------------------------

    def db_bytes(self) -> int:
        """Bytes on disk, counting the write-ahead log and shared-memory file."""
        total = 0
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix)
            try:
                total += candidate.stat().st_size
            except OSError:
                continue
        return total

    def stats(self) -> dict[str, Any]:
        """What the store itself is costing, for ``/metrics`` and the API.

        ``oldest_sample`` and ``newest_sample`` are None on an empty store. They
        are timestamps, and a 0 there would read as January 1970 on every chart
        and every alert that touched it.
        """
        stats: dict[str, Any] = {
            "enabled": bool(self.settings.enabled),
            "available": self.available,
            "degraded": self._degraded,
            "path": str(self.path),
            "db_bytes": self.db_bytes() if self.available else 0,
            "retention_hours": self.settings.retention_hours,
            "retention_days": self.settings.retention_days,
            "interval_seconds": self.settings.interval_seconds,
            "samples": 0,
            "downsampled": 0,
            "oldest_sample": None,
            "newest_sample": None,
        }
        if not self.available:
            return stats
        try:
            with self._lock:
                conn = self._conn
                if conn is None:
                    return stats
                count, oldest, newest = conn.execute(
                    "SELECT COUNT(*), MIN(ts), MAX(ts) FROM samples"
                ).fetchone()
                rolled = conn.execute("SELECT COUNT(*) FROM samples_1m").fetchone()[0]
            stats["samples"] = int(count or 0)
            stats["downsampled"] = int(rolled or 0)
            stats["oldest_sample"] = optional_float(oldest)
            stats["newest_sample"] = optional_float(newest)
        except Exception:
            self._fail("could not read store stats")
        return stats

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _get_watermark(conn: sqlite3.Connection) -> Optional[int]:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (_WATERMARK_KEY,)
        ).fetchone()
        if row is None:
            return None
        try:
            return int(float(row[0]))
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# The cadence
# ---------------------------------------------------------------------------

class MetricsSampler:
    """Ticks a snapshot into the store on a fixed interval.

    A daemon thread rather than an asyncio task, for two reasons. The snapshot
    calls into NVML and the write calls fsync, and neither belongs on the event
    loop of a server that is also proxying inference. And the collector is
    already a threading-locked object, so this is the same concurrency model it
    has always had rather than a second one.

    Every callback is wrapped: a snapshot provider that raises, or a store that
    raises, costs one tick and one log line. The sampler keeps ticking, and the
    server never sees it.
    """

    def __init__(
        self,
        store: Any,
        snapshot_provider: Callable[[], Mapping[str, Any]],
        interval_seconds: Optional[float] = None,
        maintain_seconds: float = BUCKET_SECONDS,
    ) -> None:
        self._store = store
        self._snapshot_provider = snapshot_provider
        settings = getattr(store, "settings", None)
        default = getattr(settings, "interval_seconds", DEFAULT_INTERVAL_SECONDS)
        self.interval_seconds = max(
            1.0, float(interval_seconds if interval_seconds else default)
        )
        self.maintain_seconds = max(float(BUCKET_SECONDS), float(maintain_seconds))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error_logged = False
        self._last_maintain = 0.0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start ticking. Returns immediately; the first tick is on the thread.

        Deliberately not synchronous: the first tick initialises NVML, and the
        first maintenance pass rolls up whatever accumulated while the node was
        down. Doing either inline would hold up ``on_startup`` and with it the
        port the dashboard is waiting on.
        """
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="ainode-metrics-sampler", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Ask the thread to finish the current tick and stop."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    # -- one tick ----------------------------------------------------------

    def tick(self, ts: Optional[float] = None) -> int:
        """Take one snapshot and store it. Never raises. Returns rows written."""
        try:
            snapshot = self._snapshot_provider() or {}
        except Exception:
            self._log_once("could not take a metrics snapshot")
            return 0
        try:
            return int(self._store.write(series_from_snapshot(snapshot), ts=ts) or 0)
        except Exception:
            self._log_once("could not store a metrics sample")
            return 0

    def maintain(self, now: Optional[float] = None) -> None:
        """Roll up and prune. Never raises."""
        try:
            self._store.maintain(now)
        except Exception:
            self._log_once("could not run metrics retention")

    # -- internals ---------------------------------------------------------

    def _run(self) -> None:
        self.tick()
        self.maintain()
        self._last_maintain = time.time()
        while not self._stop.wait(self.interval_seconds):
            self.tick()
            if time.time() - self._last_maintain >= self.maintain_seconds:
                self._last_maintain = time.time()
                self.maintain()

    def _log_once(self, message: str) -> None:
        if self._error_logged:
            return
        self._error_logged = True
        logger.warning("metrics retention: %s. Sampling continues.", message,
                       exc_info=True)
