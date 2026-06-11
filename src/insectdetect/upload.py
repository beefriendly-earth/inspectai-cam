"""Background uploader for captured frames to the Insector API.

OAK camera data uploaded:
  - Detection frames:  POST /api/full-frame/upload  (name, image/jpeg, session_path)
  - Timelapse frames:  POST /api/full-frame/upload  (only if upload_timelapse=True)

Vimba (spectral) camera data uploaded:
  - Detection PNG + JSON pair   → POST /api/spectral/upload  (name, session_path)
  - Timelapse PNG + JSON pair   → POST /api/spectral/upload  (only if upload_timelapse=True)

No server-side DB records are created for this device. Both endpoints are file-only;
POST /api/record and POST /api/record/upload-image are not used.

Uses an in-memory queue so captures are never blocked by network I/O.
A per-session upload manifest (uploaded.csv) tracks successfully uploaded files.
On the next session start, call enqueue_leftover_sessions() to re-enqueue any
files from previous sessions that are absent from their manifest.
Files that fail permanently after max_retries remain on disk and will be retried
in the next session.
"""

from __future__ import annotations

import csv
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import psutil
import requests

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "uploaded.csv"
_MANIFEST_COLUMNS = ("path", "kind")


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _load_manifest(session_dir: Path) -> set[str]:
    manifest_path = session_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        return set()
    uploaded: set[str] = set()
    try:
        with manifest_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "path" in row:
                    uploaded.add(row["path"])
    except Exception as exc:
        logger.warning("Could not read upload manifest %s: %s", manifest_path, exc)
    return uploaded


def _append_manifest(session_dir: Path, relative_path: str, kind: str) -> None:
    """Append one successfully uploaded entry to the session manifest.

    Creates the manifest with a header row if it does not yet exist.
    Uses line buffering (buffering=1) so each row is flushed immediately.
    """
    manifest_path = session_dir / _MANIFEST_FILENAME
    write_header = not manifest_path.exists()
    try:
        with manifest_path.open("a", newline="", encoding="utf-8", buffering=1) as f:
            writer = csv.DictWriter(f, fieldnames=_MANIFEST_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow({"path": relative_path, "kind": kind})
    except Exception as exc:
        logger.warning("Could not write upload manifest %s: %s", manifest_path, exc)


# ---------------------------------------------------------------------------
# Session path helper
# ---------------------------------------------------------------------------

def session_path_for(file_path: Path, data_path: Path, content_subdir: str = "") -> str:
    """Return the device-relative path to send as session_path to the upload server.

    Strips the content-type subdirectory (e.g. 'spectral') so the server organizes
    files by date/session without repeating the implied content subdir.

    Examples:
        session_dir/<stem>.jpg                      → '2026-06-09/2026-06-09_18-23-45'
        session_dir/timelapse/<stem>_timelapse.jpg  → '2026-06-09/2026-06-09_18-23-45/timelapse'
        session_dir/spectral/<stem>_spectral.png    → '2026-06-09/2026-06-09_18-23-45'
    """
    rel = file_path.parent.relative_to(data_path)
    parts = list(rel.parts)
    if content_subdir and content_subdir in parts:
        parts.remove(content_subdir)
    return "/".join(parts)


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

class InsectorUploadError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


def _raise_for_status(resp: requests.Response) -> None:
    if 200 <= resp.status_code < 300:
        return
    msg = "Request failed"
    try:
        j = resp.json()
        # FastAPI/Pydantic validation errors come back as {"detail": [...]}.
        detail = j.get("detail") or j.get("error") or j.get("message")
        if isinstance(detail, list):
            msg = "; ".join(
                f"{'.'.join(str(x) for x in e.get('loc', []))}: {e.get('msg', '')}"
                for e in detail
            )
        elif detail:
            msg = str(detail)
        else:
            msg = str(j)
    except Exception:
        msg = resp.text[:200] if resp.text else "Request failed"
    raise InsectorUploadError(resp.status_code, msg)


@dataclass(frozen=True)
class UploadConfig:
    """Connection and behaviour settings for the uploader.

    Field names intentionally mirror UploadServerConfig in config.py so that
    the dataclass can be constructed directly from the Pydantic model fields:

        cfg = UploadConfig(**config.storage.upload_server.model_dump())
    """
    base_url: str
    api_key: str
    timeout_s: float = 30.0
    max_retries: int = 3
    retry_delay_s: float = 5.0
    upload_timelapse: bool = False
    upload_spectral: bool = True
    enabled: bool = True


class _InsectorClient:
    """Minimal API client covering the upload endpoints."""

    def __init__(self, cfg: UploadConfig) -> None:
        self._base = cfg.base_url.rstrip("/")
        self._key = cfg.api_key
        self._timeout = cfg.timeout_s
        self._upload_session = requests.Session()
        self._health_session = requests.Session()
        self._upload_lock = threading.Lock()

    def _auth(self) -> dict[str, str]:
        return {"X-API-Key": self._key}

    def upload_full_frame(self, file_stem: str, jpg_path: Path, session_path: str = "") -> bool:
        """POST /api/full-frame/upload — upload an OAK RGB JPEG.

        Returns True on success, False if storage is not configured (HTTP 501).
        """
        url = f"{self._base}/api/full-frame/upload"
        with self._upload_lock:
            with jpg_path.open("rb") as f:
                data: dict[str, str] = {"name": file_stem}
                if session_path:
                    data["session_path"] = session_path
                resp = self._upload_session.post(
                    url,
                    headers=self._auth(),
                    files={"image": (jpg_path.name, f, "image/jpeg")},
                    data=data,
                    timeout=self._timeout,
                )
        if resp.status_code == 501:
            return False
        _raise_for_status(resp)
        return True

    def upload_spectral_pair(
        self, file_stem: str, png_path: Path, json_path: Path, session_path: str = ""
    ) -> bool:
        """POST /api/spectral/upload — upload a Vimba PNG + JSON sidecar pair.

        Returns True on success, False if storage is not configured (HTTP 501).
        """
        url = f"{self._base}/api/spectral/upload"
        with self._upload_lock:
            with png_path.open("rb") as png_f, json_path.open("rb") as json_f:
                data: dict[str, str] = {"name": file_stem}
                if session_path:
                    data["session_path"] = session_path
                resp = self._upload_session.post(
                    url,
                    headers=self._auth(),
                    files={
                        "image": (png_path.name, png_f, "image/png"),
                        "metadata": (json_path.name, json_f, "application/json"),
                    },
                    data=data,
                    timeout=self._timeout,
                )
        if resp.status_code == 501:
            return False
        _raise_for_status(resp)
        return True

    def submit_health_report(self, report: dict[str, object]) -> None:
        """POST /api/health — None values are stripped before sending."""
        url = f"{self._base}/api/health"
        payload = {k: v for k, v in report.items() if v is not None}
        resp = self._health_session.post(
            url,
            headers={**self._auth(), "Content-Type": "application/json"},
            json=payload,
            timeout=15.0,
        )
        _raise_for_status(resp)


# ---------------------------------------------------------------------------
# Queue item
# ---------------------------------------------------------------------------

# full_frame   — POST /api/full-frame/upload: OAK JPEG (detection or timelapse)
# spectral_png — POST /api/spectral/upload: Vimba PNG + JSON sidecar pair
FileKind = Literal["full_frame", "spectral_png"]


@dataclass
class _UploadItem:
    kind: FileKind
    identifier: str       # file_stem sent as `name` in the upload request
    session_dir: Path     # used to write manifest entry and compute relative_path
    primary_path: Path    # JPEG (full_frame) or PNG (spectral_png)
    session_path: str     # device-relative path sent to server for organization
    sidecar_path: Path | None = None  # JSON sidecar for spectral_png uploads
    retries: int = 0
    retry_after: float = field(default_factory=time.monotonic)

    @property
    def manifest_key(self) -> str:
        return str(self.primary_path.relative_to(self.session_dir))

    @property
    def log_name(self) -> str:
        return self.primary_path.name


# ---------------------------------------------------------------------------
# Uploader
# ---------------------------------------------------------------------------

class CaptureUploader:
    """Background uploader that drains a thread-safe in-memory queue of captured files.

    OAK RGB frames go to POST /api/full-frame/upload.
    Vimba spectral pairs go to POST /api/spectral/upload.
    No server-side DB records are created for this device.

    Usage::

        uploader = CaptureUploader(cfg, session_dir)
        uploader.start()
        uploader.enqueue_leftover_sessions(data_path)

        # After save_encoded_frame() completes:
        uploader.enqueue_oak_jpeg(jpg_path, session_path)

        # After save_vimba_frame() completes:
        uploader.enqueue_spectral_pair(file_stem, png_path, json_path, session_path)

        uploader.stop(wait=True)
    """

    def __init__(self, cfg: UploadConfig, session_dir: Path) -> None:
        self._cfg = cfg
        self._session_dir = session_dir
        self._client = _InsectorClient(cfg)
        self._queue: list[_UploadItem] = []
        self._queue_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="CaptureUploader", daemon=False)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread.start()
        logger.info(
            "CaptureUploader started (upload_timelapse=%s, upload_spectral=%s)",
            self._cfg.upload_timelapse, self._cfg.upload_spectral,
        )

    def stop(self, wait: bool = True, timeout: float = 60.0) -> None:
        """Signal the upload thread to drain the remaining queue and exit."""
        self._stop.set()
        if wait:
            self._thread.join(timeout=timeout)
            remaining = self.pending()
            if remaining > 0:
                logger.warning(
                    "CaptureUploader stopped with %d item(s) still pending — "
                    "they will be retried next session via uploaded.csv",
                    remaining,
                )

    def enqueue_oak_jpeg(
        self, jpg_path: Path, session_path: str, is_timelapse: bool = False
    ) -> None:
        """Enqueue an OAK JPEG for upload to POST /api/full-frame/upload.

        Timelapse frames are silently skipped unless upload_timelapse is enabled.

        Args:
            jpg_path:     Absolute path to the saved JPEG file.
            session_path: Device-relative session path for server-side organization.
            is_timelapse: True if this is a timelapse frame.
        """
        if is_timelapse and not self._cfg.upload_timelapse:
            return
        self._enqueue(_UploadItem(
            kind="full_frame",
            identifier=jpg_path.stem,
            session_dir=self._session_dir,
            primary_path=jpg_path,
            session_path=session_path,
        ))

    def enqueue_spectral_pair(
        self,
        file_stem: str,
        png_path: Path,
        json_path: Path,
        session_path: str,
        is_timelapse: bool = False,
    ) -> None:
        """Enqueue a Vimba spectral PNG + JSON sidecar pair for upload.

        Timelapse spectral pairs are skipped unless upload_timelapse is enabled.

        Args:
            file_stem:    Bare filename stem without _spectral/_spectral_timelapse suffix.
            png_path:     Absolute path to the saved PNG file.
            json_path:    Absolute path to the matching JSON sidecar file.
            session_path: Device-relative session path for server-side organization.
            is_timelapse: True if this pair was captured on a timelapse trigger.
        """
        if is_timelapse and not self._cfg.upload_timelapse:
            return
        if not self._cfg.upload_spectral:
            return
        self._enqueue(_UploadItem(
            kind="spectral_png",
            identifier=file_stem,
            session_dir=self._session_dir,
            primary_path=png_path,
            sidecar_path=json_path,
            session_path=session_path,
        ))

    def pending(self) -> int:
        """Return the number of items still in the upload queue."""
        with self._queue_lock:
            return len(self._queue)

    def send_health_report(
        self,
        session_start: float,
        rpi_metrics: dict[str, object] | None,
        power_info: dict[str, object] | None,
    ) -> None:
        """Build and POST a health report to /api/health.

        Best-effort — failures are logged at DEBUG level and never affect the capture loop.

        Args:
            session_start: Unix timestamp (time.time()) of when the session started.
            rpi_metrics:   Dict returned by get_rpi_metrics(), or None.
            power_info:    Dict returned by pwr.get_power_info(), or None.
        """
        try:
            disk = psutil.disk_usage("/")
            report: dict[str, object] = {
                "timestamp": time.time(),
                "minutes_recorded": round((time.time() - session_start) / 60, 1),
                "records_waiting_upload": self.pending(),
                "disk": {
                    "total_gb":     round(disk.total / 1_073_741_824, 2),
                    "used_gb":      round(disk.used  / 1_073_741_824, 2),
                    "free_gb":      round(disk.free  / 1_073_741_824, 2),
                    "percent_used": disk.percent,
                },
            }
            if rpi_metrics:
                report["temperature"] = rpi_metrics.get("rpi_cpu_temp")
            if power_info:
                charge = power_info.get("charge_level")
                if charge == "USB_C_IN":
                    report["battery_level"] = 100.0
                elif isinstance(charge, (int, float)):
                    report["battery_level"] = float(charge)
            self._client.submit_health_report(report)
            logger.debug("Health report sent")
        except Exception as exc:
            logger.debug("Health report failed (non-critical): %s", exc)

    # ------------------------------------------------------------------
    # Leftover scanning
    # ------------------------------------------------------------------

    def enqueue_leftover_sessions(self, data_path: Path) -> int:
        """Scan previous session directories and enqueue files absent from their manifest.

        Respects upload_timelapse: timelapse files are skipped if it is False,
        consistent with how they are handled during live capture.

        Args:
            data_path: Root data directory (DATA_PATH) containing dated subdirectories.

        Returns:
            Number of items enqueued.
        """
        total = 0
        for date_dir in sorted(data_path.iterdir()):
            if not date_dir.is_dir():
                continue
            for session_dir in sorted(date_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                if session_dir.resolve() == self._session_dir.resolve():
                    continue
                total += self._enqueue_session_leftovers(session_dir, data_path)

        if total:
            logger.info("CaptureUploader: enqueued %d leftover item(s) from previous sessions", total)
        return total

    def _enqueue_session_leftovers(self, session_dir: Path, data_path: Path) -> int:
        uploaded = _load_manifest(session_dir)
        enqueued = 0

        def _maybe(item: _UploadItem) -> None:
            nonlocal enqueued
            if item.manifest_key not in uploaded:
                self._enqueue(item)
                enqueued += 1

        # OAK detection JPEGs
        for jpg_path in session_dir.glob("*.jpg"):
            sp = session_path_for(jpg_path, data_path)
            _maybe(_UploadItem(
                kind="full_frame",
                identifier=jpg_path.stem,
                session_dir=session_dir,
                primary_path=jpg_path,
                session_path=sp,
            ))

        # OAK timelapse JPEGs (skipped if upload_timelapse is disabled)
        if self._cfg.upload_timelapse:
            timelapse_dir = session_dir / "timelapse"
            if timelapse_dir.is_dir():
                for jpg_path in timelapse_dir.glob("*.jpg"):
                    sp = session_path_for(jpg_path, data_path)
                    _maybe(_UploadItem(
                        kind="full_frame",
                        identifier=jpg_path.stem,
                        session_dir=session_dir,
                        primary_path=jpg_path,
                        session_path=sp,
                    ))

        # Vimba spectral detection pairs (skipped if upload_spectral is disabled)
        if self._cfg.upload_spectral:
            spectral_dir = session_dir / "spectral"
            if spectral_dir.is_dir():
                enqueued += self._enqueue_spectral_leftovers(
                    spectral_dir, session_dir, data_path, uploaded, suffix="_spectral", is_timelapse=False
                )
                # Vimba spectral timelapse pairs (skipped if upload_timelapse is disabled)
                if self._cfg.upload_timelapse:
                    spectral_tl_dir = spectral_dir / "timelapse"
                    if spectral_tl_dir.is_dir():
                        enqueued += self._enqueue_spectral_leftovers(
                            spectral_tl_dir, session_dir, data_path, uploaded,
                            suffix="_spectral_timelapse", is_timelapse=True
                        )

        return enqueued

    def _enqueue_spectral_leftovers(
        self,
        directory: Path,
        session_dir: Path,
        data_path: Path,
        uploaded: set[str],
        suffix: str,
        is_timelapse: bool,
    ) -> int:
        if is_timelapse and not self._cfg.upload_timelapse:
            return 0
        enqueued = 0
        for png_path in directory.glob("*.png"):
            if str(png_path.relative_to(session_dir)) in uploaded:
                continue
            json_path = png_path.with_suffix(".json")
            if not json_path.exists():
                logger.warning(
                    "Skipping leftover spectral PNG (missing JSON sidecar): %s", png_path
                )
                continue
            file_stem = png_path.stem.removesuffix(suffix)
            sp = session_path_for(png_path, data_path, content_subdir="spectral")
            self._enqueue(_UploadItem(
                kind="spectral_png",
                identifier=file_stem,
                session_dir=session_dir,
                primary_path=png_path,
                sidecar_path=json_path,
                session_path=sp,
            ))
            enqueued += 1
        return enqueued

    # ------------------------------------------------------------------
    # Internal queue management
    # ------------------------------------------------------------------

    def _enqueue(self, item: _UploadItem) -> None:
        with self._queue_lock:
            self._queue.append(item)

    def _pop_ready(self) -> _UploadItem | None:
        now = time.monotonic()
        with self._queue_lock:
            for i, item in enumerate(self._queue):
                if item.retry_after <= now:
                    return self._queue.pop(i)
        return None

    def _requeue(self, item: _UploadItem) -> None:
        item.retry_after = time.monotonic() + self._cfg.retry_delay_s
        with self._queue_lock:
            self._queue.append(item)

    def _run(self) -> None:
        logger.debug("CaptureUploader thread running")
        while not self._stop.is_set():
            item = self._pop_ready()
            if item is None:
                time.sleep(0.2)
                continue
            self._process(item)
        # Drain remaining ready items once (best-effort flush before exit)
        drained = 0
        deadline = time.monotonic() + 10.0  # hard cap on drain time
        while time.monotonic() < deadline:
            item = self._pop_ready()
            if item is None:
                break
            self._process(item)
            drained += 1
        remaining = self.pending()
        if remaining:
            logger.warning(
                "CaptureUploader exiting with %d item(s) still pending — will retry next session",
                remaining,
            )
        logger.info("CaptureUploader thread finished (drained %d item(s))", drained)

    def _process(self, item: _UploadItem) -> None:
        try:
            ok = self._dispatch(item)
        except InsectorUploadError as exc:
            if exc.status_code < 500:
                logger.warning(
                    "Dropping %s (HTTP %s, no retry): %s",
                    item.log_name, exc.status_code, exc,
                )
                return
            ok = False
        except Exception as exc:
            logger.debug("Upload attempt failed for %s: %s", item.log_name, exc)
            ok = False

        if ok:
            logger.debug("Uploaded %s (%s)", item.log_name, item.kind)
            _append_manifest(item.session_dir, item.manifest_key, item.kind)
            return

        item.retries += 1
        if item.retries >= self._cfg.max_retries:
            logger.warning(
                "Giving up on %s after %d retries — will retry next session",
                item.log_name, item.retries,
            )
            return

        logger.debug(
            "Requeueing %s (attempt %d/%d) after %.0f s",
            item.log_name, item.retries, self._cfg.max_retries, self._cfg.retry_delay_s,
        )
        self._requeue(item)

    def _dispatch(self, item: _UploadItem) -> bool:
        if not item.primary_path.exists():
            logger.warning("Upload skipped — file no longer exists: %s", item.primary_path)
            return True

        if item.kind == "full_frame":
            return self._client.upload_full_frame(
                item.identifier, item.primary_path, item.session_path
            )

        if item.kind == "spectral_png":
            if item.sidecar_path is None or not item.sidecar_path.exists():
                logger.warning(
                    "Spectral JSON sidecar missing for %s — skipping", item.primary_path.name
                )
                return True
            return self._client.upload_spectral_pair(
                item.identifier, item.primary_path, item.sidecar_path, item.session_path
            )

        logger.error("Unknown upload kind: %s", item.kind)
        return True
