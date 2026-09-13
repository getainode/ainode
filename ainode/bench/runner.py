"""Job lifecycle for an in-product bench run.

A run is a long-lived job driven from the browser, so it gets the same shape the
training engine already uses in this codebase: a status, a progress number, a
bounded log deque the UI tails, and a cancel that actually stops work. The
measurement itself is the blocking stdlib code in :mod:`ainode.bench.measure`,
driven through ``asyncio.to_thread`` so a 90-second sustained generation does not
freeze the event loop that is also proxying live inference.

One run at a time per node, and that is a measurement rule rather than a resource
one: two benchmarks against the same engine measure each other's queue depth.

Results are written to ``~/.ainode/bench/results/`` as schema-1 JSON, the same
format ``bench/results/`` holds and the same format ``bench/report.py`` renders.
The run id IS the file stem, so a result stays downloadable after a restart has
forgotten the in-memory job.
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import re
import time
from pathlib import Path

from ainode.bench import report as report_mod
from ainode.bench.measure import (
    Cancelled,
    Reporter,
    SECTION_TITLES,
    Telemetry,
    build_notes,
    build_record,
    measure,
    slug,
)

logger = logging.getLogger(__name__)

SOURCE = "ainode /api/bench (ainode.bench)"


def results_dir() -> Path:
    """Where in-product runs land. Read at call time, not import time, so tests
    can point AINODE_HOME somewhere disposable."""
    from ainode.core.config import AINODE_HOME

    return Path(AINODE_HOME) / "bench" / "results"


def label_slug(label: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "-", (label or "run").strip().lower())
    return re.sub(r"-+", "-", s).strip("-") or "run"


class BenchBusy(RuntimeError):
    """A second run was requested while one was still going."""

    def __init__(self, running_id: str):
        super().__init__(f"a bench run is already in progress on this node ({running_id})")
        self.running_id = running_id


class BenchRun:
    """One benchmark run and everything the UI needs to watch it."""

    LOG_MAX = 2000

    def __init__(self, run_id, opts, target, describe, telemetry_read=None,
                 source=SOURCE, warnings=None, stamp=None, out_dir=None):
        self.run_id = run_id
        self.stamp = stamp or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        self.out_dir = Path(out_dir) if out_dir else results_dir()
        self.opts = opts
        self.target = target
        self._describe = describe
        self._telemetry_read = telemetry_read
        self._source = source
        self.status = "pending"
        self.created_at = time.time()
        self.started_at = None
        self.finished_at = None
        self.error = None
        self.warnings = list(warnings or [])
        self.result = None
        self.result_file = None
        self.logs = collections.deque(maxlen=self.LOG_MAX)
        self._cancel = asyncio.Event()
        self._task = None
        # Progress: which section, and which step inside it.
        self._section = None
        self._section_index = 0
        self._step = 0
        self._step_total = 0
        self._step_label = ""

    # ---------------------------------------------------------------- state

    def log(self, msg: str) -> None:
        self.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def percent(self) -> float:
        total = max(1, len(self.opts.sections))
        frac = self._section_index
        if self._step_total:
            frac += min(1.0, self._step / self._step_total)
        return round(min(100.0, frac / total * 100.0), 1)

    def progress(self) -> dict:
        return {
            "percent": 100.0 if self.status == "completed" else self.percent(),
            "section": self._section,
            "section_label": SECTION_TITLES.get(self._section, self._section or ""),
            "section_index": self._section_index,
            "section_total": len(self.opts.sections),
            "step": self._step,
            "step_total": self._step_total,
            "step_label": self._step_label,
        }

    def get_status(self, log_tail: int = 200, include_result: bool = True) -> dict:
        elapsed = None
        if self.started_at:
            elapsed = round((self.finished_at or time.time()) - self.started_at, 1)
        out = {
            "run_id": self.run_id,
            "status": self.status,
            "model": self.opts.model,
            "label": self.opts.label,
            "node": self.target.node_name if self.target is not None else "",
            "node_id": self.target.node_id if self.target is not None else "",
            "endpoint": self.target.url if self.target is not None else "",
            "sections": list(self.opts.sections),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": elapsed,
            "progress": self.progress(),
            "log": list(self.logs)[-log_tail:] if log_tail else [],
            "warnings": list(self.warnings),
            "error": self.error,
            "result_file": self.result_file,
        }
        if include_result:
            out["result"] = self.result
            out["summary"] = summarize(self.result) if self.result else None
        return out

    # ---------------------------------------------------------------- driving

    async def start(self) -> None:
        if self.status != "pending":
            raise RuntimeError(f"cannot start a run in '{self.status}' state")
        self._task = asyncio.get_event_loop().create_task(self._execute())

    async def cancel(self) -> None:
        """Ask the run to stop. The in-flight HTTP read notices within one SSE
        line, so this lands inside a long generation rather than after it."""
        if self.status in ("completed", "failed", "cancelled"):
            return
        self._cancel.set()
        if self.status == "pending":
            self.status = "cancelled"
            self.finished_at = time.time()
            self.log("Cancelled before start")

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    async def _execute(self) -> None:
        self.status = "running"
        self.started_at = time.time()
        self.log(f"Bench {self.opts.model} on "
                 f"{self.target.node_name or self.target.host}:{self.target.port}")
        self.log(f"label {self.opts.label} - sections {', '.join(self.opts.sections)}")
        for w in self.warnings:
            self.log(f"warn: {w}")
        tel = Telemetry(read=self._telemetry_read)
        try:
            mb, pl, warn = await self._describe()
            self.warnings.extend(w for w in warn if w not in self.warnings)
            for w in warn:
                self.log(f"warn: {w}")
            self.log(f"node {pl.get('node', 'unknown')} {pl.get('gpu', '')} "
                     f"tp={pl.get('tp', '?')} engine={pl.get('engine_image', 'unknown image')}")
            await asyncio.to_thread(tel.start)
            rep = _JobReporter(self)
            results, seconds, cpt = await asyncio.to_thread(measure, self.opts, rep)
            await asyncio.to_thread(tel.stop)
            telemetry = tel.result()
            if telemetry:
                results["telemetry"] = telemetry
                self.log(f"  TELEMETRY  peak GPU {telemetry.get('gpu_util_pct')}%  "
                         f"mem {telemetry.get('gpu_mem_used_gb')}/"
                         f"{telemetry.get('gpu_mem_total_gb')} GB  "
                         f"{telemetry.get('temp_c')} C  ({telemetry.get('samples')} samples)")
            notes = build_notes(self.opts, results, pl, self.warnings, seconds, self._source)
            record = build_record(self.opts, mb, pl, results, cpt, notes,
                                  self.stamp, self._source)
            path = self.out_dir / f"{self.run_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1) + "\n")
            record["_file"] = path.name
            self.result = record
            self.result_file = path.name
            self.status = "completed"
            self.log(f"saved {path}")
        except Cancelled:
            await asyncio.to_thread(tel.stop)
            self.status = "cancelled"
            # Same rule the CLI follows on ^C: a partial run is not a result.
            self.log("Cancelled; nothing was written")
        except Exception as exc:
            await asyncio.to_thread(tel.stop)
            self.status = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
            self.log(f"FAILED {self.error}")
            logger.exception("bench run %s failed", self.run_id)
        finally:
            self.finished_at = time.time()


class _JobReporter(Reporter):
    """Routes measure()'s progress into the job the UI is polling."""

    def __init__(self, run: BenchRun):
        self.run = run

    def section(self, key, title):
        # Sections run in the order the options list them, so the index is the
        # position of this key rather than a counter that a skipped section
        # would desynchronise.
        try:
            self.run._section_index = list(self.run.opts.sections).index(key)
        except ValueError:
            pass
        self.run._section = key
        self.run._step = 0
        self.run._step_total = 0
        self.run._step_label = ""
        self.run.log(title.strip())

    def log(self, msg):
        self.run.log(msg.rstrip())

    def step(self, done, total, label=""):
        self.run._step = done
        self.run._step_total = total
        self.run._step_label = label

    def cancelled(self):
        return self.run.cancelled()


# ---------------------------------------------------------------- summaries

def summarize(record: dict) -> dict:
    """The row the results table shows. Every field is read out of the record, so
    a section that did not run comes back as None and renders "not measured"."""
    if not record:
        return {}
    model = record.get("model") or {}
    pl = record.get("placement") or {}
    streams, agg = report_mod.best_conc(record)
    return {
        "model": model.get("id"),
        "model_name": model.get("name"),
        "label": record.get("label"),
        "stamp": record.get("stamp"),
        "node": pl.get("node"),
        "gpu": pl.get("gpu"),
        "tp": pl.get("tp"),
        "engine_image": pl.get("engine_image"),
        "stacked_with": list(pl.get("stacked_with") or []),
        "single_tok_s": report_mod.single_tps(record),
        "conc_streams": streams,
        "conc_aggregate_tok_s": agg,
        "sections": sorted(k for k in (record.get("results") or {}) if k != "telemetry"),
    }


# ---------------------------------------------------------------- manager

class BenchManager:
    """Owns this node's bench runs, live and on disk.

    Deliberately app-agnostic: the route layer injects the target, the placement
    describer and the telemetry reader, which is what lets the whole lifecycle be
    tested against a fake engine with no cluster.
    """

    def __init__(self, dirpath: Path | None = None):
        self._runs: dict[str, BenchRun] = {}
        self._order: list[str] = []
        self._active_id: str | None = None
        self._dir = Path(dirpath) if dirpath else None

    @property
    def dir(self) -> Path:
        return self._dir if self._dir is not None else results_dir()

    # -- ids ---------------------------------------------------------------

    def _new_id(self, model: str, label: str, stamp: str) -> str:
        base = f"{stamp}-{slug(model)}-{label_slug(label)}"
        rid, n = base, 2
        while rid in self._runs or (self.dir / f"{rid}.json").exists():
            rid = f"{base}-{n}"
            n += 1
        return rid

    # -- lifecycle ---------------------------------------------------------

    @property
    def active_id(self):
        run = self._runs.get(self._active_id or "")
        if run is not None and run.status in ("pending", "running"):
            return run.run_id
        self._active_id = None
        return None

    async def submit(self, opts, target, describe, telemetry_read=None,
                     source=SOURCE, warnings=None) -> BenchRun:
        """Create and start a run. Raises BenchBusy if one is already going."""
        busy = self.active_id
        if busy:
            raise BenchBusy(busy)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        run_id = self._new_id(opts.model, opts.label, stamp)
        run = BenchRun(run_id, opts, target, describe, telemetry_read=telemetry_read,
                       source=source, warnings=warnings, stamp=stamp, out_dir=self.dir)
        self._runs[run_id] = run
        self._order.insert(0, run_id)
        self._active_id = run_id
        await run.start()
        return run

    def get(self, run_id: str):
        return self._runs.get(run_id)

    async def cancel(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        if run is None or run.status in ("completed", "failed", "cancelled"):
            return False
        await run.cancel()
        return True

    def delete(self, run_id: str) -> bool:
        """Forget a run and remove its result file. Refuses a live run: cancel it
        first, so a delete can never leave an orphaned load on the engine."""
        run = self._runs.get(run_id)
        if run is not None and run.status in ("pending", "running"):
            raise BenchBusy(run_id)
        found = False
        if run is not None:
            del self._runs[run_id]
            if run_id in self._order:
                self._order.remove(run_id)
            found = True
        path = self.dir / f"{run_id}.json"
        if path.is_file():
            path.unlink()
            found = True
        return found

    # -- listing -----------------------------------------------------------

    def result_path(self, name: str):
        """Path of a result file by run id, or None if it is not one of ours.

        Rejects anything with a path separator or a dot segment: the name comes
        off the URL, and a results download must not be able to read outside the
        results directory.
        """
        if not name or "/" in name or "\\" in name or name.startswith("."):
            return None
        path = self.dir / f"{name}.json"
        try:
            path.relative_to(self.dir)
        except ValueError:
            return None
        return path if path.is_file() else None

    def list_runs(self, log_tail: int = 0) -> list:
        """Live runs plus every result file on disk, newest first.

        On-disk results are included so the table survives a restart: the runs
        dict is in-memory, the JSON the leaderboard consumes is not.
        """
        rows = []
        seen = set()
        for rid in self._order:
            run = self._runs.get(rid)
            if run is None:
                continue
            row = run.get_status(log_tail=log_tail, include_result=False)
            row["summary"] = summarize(run.result) if run.result else None
            rows.append(row)
            seen.add(rid)
        for path in sorted(self.dir.glob("*.json"), reverse=True) if self.dir.is_dir() else []:
            rid = path.stem
            if rid in seen:
                continue
            try:
                record = json.loads(path.read_text())
            except (ValueError, OSError):
                continue
            summary = summarize(record)
            pl = record.get("placement") or {}
            rows.append({
                "run_id": rid,
                "status": "completed",
                "model": (record.get("model") or {}).get("id"),
                "label": record.get("label"),
                "node": pl.get("node"),
                "node_id": "",
                "endpoint": "",
                "sections": summary.get("sections") or [],
                "created_at": path.stat().st_mtime,
                "started_at": None,
                "finished_at": path.stat().st_mtime,
                "elapsed_seconds": None,
                "progress": {"percent": 100.0, "section": None, "section_label": "",
                             "section_index": 0, "section_total": 0, "step": 0,
                             "step_total": 0, "step_label": ""},
                "log": [],
                "warnings": [],
                "error": None,
                "result_file": path.name,
                "summary": summary,
                "restored": True,
            })
        rows.sort(key=lambda r: (r.get("created_at") or 0), reverse=True)
        return rows
