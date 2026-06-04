"""Background uploader for captured frames and metadata to the Insector API.

OAK camera data uploaded:
  - Detection frames (if enabled):
      1. POST /api/record              — create detection record (client_uuid, timestamp,
                                         label, confidence, track_id)
      2. POST /api/record/upload-image — attach JPEG to that record
  - Timelapse frames (if upload_timelapse=True):
      POST /api/record/upload-image    — image only, no prior record needed

Vimba (spectral) camera data uploaded:
  - Detection PNG + JSON pair   → POST /api/spectral/upload  (per pair, client_uuid = file_stem)
  - Timelapse PNG + JSON pair   → POST /api/spectral/upload  (only if upload_timelapse=True)

Uses an in-memory queue so captures are never blocked by network I/O.
A per-session upload manifest (uploaded.csv) tracks successfully uploaded files/records.
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

import requests

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "uploaded.csv"
_MANIFEST_COLUMNS = ("path", "kind")


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _load_manifest(session_dir: Path) -> set[str]:
    """Return the set of relative paths already recorded in the upload manifest.

    Args:
        session_dir: Session directory containing the manifest file.

    Returns:
        Set of path strings relative to session_dir, or empty set if none exists.
    """
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
    Uses line buffering (buffering=1) so each row is flushed immediately —
    the manifest stays consistent even if the process is killed mid-session.

    Args:
        session_dir:   Session directory where the manifest is written.
        relative_path: File path relative to session_dir (used as stable key).
        kind:          Upload kind string.
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
        msg = j.get("error") or j.get("message") or msg
    except Exception:
        pass
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
    # consumed by CaptureUploader.start() log message only; not forwarded to _InsectorClient
    enabled: bool = True
    # Name of the active OAK detection model. Stamped onto each detection record
    # so analysts can attribute predictions to a model version.
    model_name: str = "insect_detect"


@dataclass(frozen=True)
class DetectionRecord:
    """Detection metadata sent to POST /api/record before the image upload."""
    client_uuid: str    # = bare file_stem, correlates record with image upload
    timestamp: str      # ISO 8601 string (server accepts ISO or float epoch)
    label: str          # detection label, sent as classifications[0].name
    confidence: float   # model confidence, sent as classifications[0].probability
    track_id: int       # 0 = "no track"; omitted from payload in that case
    model_name: str     # detection model identifier; sent as classifications[0].model_name


class _InsectorClient:
    """Minimal API client covering the upload endpoints."""

    def __init__(self, cfg: UploadConfig) -> None:
        self._base = cfg.base_url.rstrip("/")
        self._key = cfg.api_key
        self._timeout = cfg.timeout_s
        self._session = requests.Session()
        self._lock = threading.RLock()

    def _auth(self) -> dict[str, str]:
        return {"X-API-Key": self._key}

    def create_record(self, record: DetectionRecord) -> None:
        """POST /api/record — must be called before upload_oak_image() for detection frames.

        Args:
            record: Detection metadata. client_uuid must match the one used in
                    the subsequent upload_oak_image() call.
        """
        url = f"{self._base}/api/record"
        payload: dict[str, object] = {
            "client_uuid": record.client_uuid,
            "timestamp": record.timestamp,
            "classifications": [
                {
                    "name": record.label,
                    "probability": record.confidence,
                    "model_name": record.model_name,
                }
            ],
        }
        # track_id == 0 is the "no active track" sentinel; omit so the server
        # stores NULL instead of confusing it with tracklet ID 0.
        if record.track_id:
            payload["track_id"] = str(record.track_id)
        with self._lock:
            resp = self._session.post(
                url,
                headers={**self._auth(), "Content-Type": "application/json"},
                json=payload,
                timeout=self._timeout,
            )
        _raise_for_status(resp)

    def upload_oak_image(self, file_stem: str, jpg_path: Path) -> bool:
        """POST /api/record/upload-image

        For detection frames, create_record() must be called first.
        For timelapse frames, no prior record is needed.

        Args:
            file_stem: Bare filename stem used as client_uuid.
            jpg_path:  Path to the JPEG file to upload.

        Returns:
            True on success, False if storage is not configured (HTTP 501).
        """
        url = f"{self._base}/api/record/upload-image"
        with self._lock:
            with jpg_path.open("rb") as f:
                files = {"image": (jpg_path.name, f, "image/jpeg")}
                data = {"client_uuid": file_stem}
                resp = self._session.post(
                    url, headers=self._auth(), files=files, data=data, timeout=self._timeout
                )
        if resp.status_code == 501:
            return False
        _raise_for_status(resp)
        return True

    def upload_spectral_pair(self, file_stem: str, png_path: Path, json_path: Path) -> bool:
        """POST /api/spectral/upload

        Args:
            file_stem: Bare filename stem used as client_uuid.
            png_path:  Path to the PNG file.
            json_path: Path to the matching JSON sidecar file.

        Returns:
            True on success, False if storage is not configured (HTTP 501).
        """
        url = f"{self._base}/api/spectral/upload"
        with self._lock:
            with png_path.open("rb") as png_f, json_path.open("rb") as json_f:
                files = {
                    "image": (png_path.name, png_f, "image/png"),
                    "metadata": (json_path.name, json_f, "application/json"),
                }
                # Server contract uses `name` (basename for the pair), not `client_uuid`.
                data = {"name": file_stem}
                resp = self._session.post(
                    url, headers=self._auth(), files=files, data=data, timeout=self._timeout
                )
        if resp.status_code == 501:
            return False
        _raise_for_status(resp)
        return True


# ---------------------------------------------------------------------------
# Queue item
# ---------------------------------------------------------------------------

# record       — POST /api/record: create detection record before image upload
# jpeg         — POST /api/record/upload-image: OAK JPEG (detection or timelapse)
# spectral_png — POST /api/spectral/upload: Vimba PNG + JSON sidecar pair
FileKind = Literal["record", "jpeg", "spectral_png"]


@dataclass
class _UploadItem:
    kind: FileKind
    identifier: str     # client_uuid for jpeg/spectral_png/record
    session_dir: Path   # used to write manifest entry and compute relative_path
    # Primary file path (JPEG or PNG); None for 'record' items
    primary_path: Path | None = None
    # Spectral PNG uploads also carry the matching JSON sidecar
    sidecar_path: Path | None = None
    # Detection metadata payload for 'record' items
    detection_record: DetectionRecord | None = None
    retries: int = 0
    retry_after: float = field(default_factory=time.monotonic)

    @property
    def manifest_key(self) -> str:
        """Stable string used as the manifest key.

        File-backed items use their path relative to session_dir.
        Record items (no file) use a synthetic key so they are also tracked.
        """
        if self.primary_path is not None:
            return str(self.primary_path.relative_to(self.session_dir))
        return f"record:{self.identifier}"

    @property
    def log_name(self) -> str:
        """Short name for log messages."""
        if self.primary_path is not None:
            return self.primary_path.name
        return self.manifest_key


# ---------------------------------------------------------------------------
# Uploader
# ---------------------------------------------------------------------------

class CaptureUploader:
    """Background uploader that drains a thread-safe in-memory queue of captured files.

    Detection frames require two sequential queue items per frame:
      1. kind='record' — POST /api/record (creates the server-side record)
      2. kind='jpeg'   — POST /api/record/upload-image (attaches the image)
    enqueue_oak_detection() always enqueues them in this order.

    Timelapse frames (enqueue_oak_timelapse) and spectral pairs (enqueue_spectral_pair)
    are only enqueued when UploadConfig.upload_timelapse is True.

    Usage::

        uploader = CaptureUploader(cfg, session_dir)
        uploader.start()
        uploader.enqueue_leftover_sessions(data_path)

        # After save_encoded_frame() completes for a detection frame:
        uploader.enqueue_oak_detection(record, jpg_path)

        # After save_encoded_frame() completes for a timelapse frame (only if enabled):
        uploader.enqueue_oak_timelapse(file_stem, jpg_path)

        # After save_vimba_frame() completes (only if enabled):
        uploader.enqueue_spectral_pair(file_stem, png_path, json_path)

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
        logger.info("CaptureUploader started (upload_timelapse=%s)", self._cfg.upload_timelapse)

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

    def enqueue_oak_detection(self, record: DetectionRecord, jpg_path: Path) -> None:
        """Enqueue a detection record + JPEG pair for upload.

        Enqueues the record item first so the server-side record is created
        before the image upload is attempted. Both items share the same
        client_uuid (record.client_uuid = jpg_path bare stem).

        Args:
            record:   Detection metadata. record.client_uuid must be the bare
                      file_stem (no extension, no suffix).
            jpg_path: Absolute path to the saved detection JPEG.
        """
        self._enqueue(_UploadItem(
            kind="record",
            identifier=record.client_uuid,
            session_dir=self._session_dir,
            detection_record=record,
        ))
        self._enqueue(_UploadItem(
            kind="jpeg",
            identifier=record.client_uuid,
            session_dir=self._session_dir,
            primary_path=jpg_path,
        ))

    def enqueue_oak_timelapse(self, file_stem: str, jpg_path: Path) -> None:
        """Enqueue an OAK timelapse JPEG for upload (only if upload_timelapse is enabled).

        Timelapse frames have no detection record — only the image is uploaded.

        Args:
            file_stem: Bare filename stem without _timelapse suffix. Used as client_uuid.
            jpg_path:  Absolute path to the saved timelapse JPEG.
        """
        if not self._cfg.upload_timelapse:
            return
        self._enqueue(_UploadItem(
            kind="jpeg",
            identifier=file_stem,
            session_dir=self._session_dir,
            primary_path=jpg_path,
        ))

    def enqueue_spectral_pair(
        self,
        file_stem: str,
        png_path: Path,
        json_path: Path,
        is_timelapse: bool = False,
    ) -> None:
        """Enqueue a Vimba spectral PNG + JSON sidecar pair for upload.

        Timelapse spectral pairs are skipped unless upload_timelapse is enabled.

        Args:
            file_stem:    Bare filename stem without _spectral/_spectral_timelapse suffix.
            png_path:     Absolute path to the saved PNG file.
            json_path:    Absolute path to the matching JSON sidecar file.
            is_timelapse: True if this pair was captured on a timelapse trigger.
        """
        if is_timelapse and not self._cfg.upload_timelapse:
            return
        self._enqueue(_UploadItem(
            kind="spectral_png",
            identifier=file_stem,
            session_dir=self._session_dir,
            primary_path=png_path,
            sidecar_path=json_path,
        ))

    def pending(self) -> int:
        """Return the number of items still in the upload queue."""
        with self._queue_lock:
            return len(self._queue)

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
                total += self._enqueue_session_leftovers(session_dir)

        if total:
            logger.info("CaptureUploader: enqueued %d leftover item(s) from previous sessions", total)
        return total

    def _enqueue_session_leftovers(self, session_dir: Path) -> int:
        uploaded = _load_manifest(session_dir)
        stem_to_row = _load_metadata_rows(session_dir)
        enqueued = 0

        def _maybe(item: _UploadItem) -> None:
            nonlocal enqueued
            if item.manifest_key not in uploaded:
                self._enqueue(item)
                enqueued += 1

        # OAK detection JPEGs + records
        for jpg_path in session_dir.glob("*.jpg"):
            file_stem = jpg_path.stem
            record_key = f"record:{file_stem}"
            jpeg_key = str(jpg_path.relative_to(session_dir))

            if record_key not in uploaded:
                row = stem_to_row.get(file_stem)
                _maybe(_UploadItem(
                    kind="record",
                    identifier=file_stem,
                    session_dir=session_dir,
                    detection_record=_row_to_detection_record(
                        file_stem, row, self._cfg.model_name
                    ),
                ))
            if jpeg_key not in uploaded:
                _maybe(_UploadItem(
                    kind="jpeg",
                    identifier=file_stem,
                    session_dir=session_dir,
                    primary_path=jpg_path,
                ))

        # OAK timelapse JPEGs (skipped if upload_timelapse is disabled)
        if self._cfg.upload_timelapse:
            timelapse_dir = session_dir / "timelapse"
            if timelapse_dir.is_dir():
                for jpg_path in timelapse_dir.glob("*.jpg"):
                    file_stem = jpg_path.stem.removesuffix("_timelapse")
                    _maybe(_UploadItem(
                        kind="jpeg",
                        identifier=file_stem,
                        session_dir=session_dir,
                        primary_path=jpg_path,
                    ))

        # Vimba spectral detection pairs
        spectral_dir = session_dir / "spectral"
        if spectral_dir.is_dir():
            enqueued += self._enqueue_spectral_leftovers(
                spectral_dir, session_dir, uploaded, suffix="_spectral", is_timelapse=False
            )
            # Vimba spectral timelapse pairs (skipped if upload_timelapse is disabled)
            if self._cfg.upload_timelapse:
                spectral_tl_dir = spectral_dir / "timelapse"
                if spectral_tl_dir.is_dir():
                    enqueued += self._enqueue_spectral_leftovers(
                        spectral_tl_dir, session_dir, uploaded,
                        suffix="_spectral_timelapse", is_timelapse=True
                    )

        return enqueued

    def _enqueue_spectral_leftovers(
        self,
        directory: Path,
        session_dir: Path,
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
            self._enqueue(_UploadItem(
                kind="spectral_png",
                identifier=file_stem,
                session_dir=session_dir,
                primary_path=png_path,
                sidecar_path=json_path,
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
        while not self._stop.is_set() or self.pending() > 0:
            item = self._pop_ready()
            if item is None:
                time.sleep(0.2)
                continue
            self._process(item)
        logger.debug("CaptureUploader thread finished")

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
        if item.kind == "record":
            if item.detection_record is None:
                logger.error("Record item missing detection_record: %s", item.identifier)
                return True
            self._client.create_record(item.detection_record)
            return True

        if item.primary_path is None or not item.primary_path.exists():
            logger.warning("Upload skipped — file no longer exists: %s", item.primary_path)
            return True

        if item.kind == "jpeg":
            return self._client.upload_oak_image(item.identifier, item.primary_path)

        if item.kind == "spectral_png":
            if item.sidecar_path is None or not item.sidecar_path.exists():
                logger.warning(
                    "Spectral JSON sidecar missing for %s — skipping", item.primary_path.name
                )
                return True
            return self._client.upload_spectral_pair(
                item.identifier, item.primary_path, item.sidecar_path
            )

        logger.error("Unknown upload kind: %s", item.kind)
        return True


# ---------------------------------------------------------------------------
# Helpers for leftover scanning
# ---------------------------------------------------------------------------

def _load_metadata_rows(session_dir: Path) -> dict[str, dict[str, str]]:
    """Load the session metadata CSV into a dict keyed by bare file_stem.

    The CSV 'filename' column contains e.g. 'hostname_2026-05-31_....jpg'.
    Stripping the extension gives the bare file_stem used as client_uuid.

    Args:
        session_dir: Session directory containing the *_metadata.csv file.

    Returns:
        Dict mapping bare file_stem → CSV row dict, or empty dict on any error.

    Note:
        Column names (filename, timestamp, label, confidence, track_id) must
        match the actual columns written by save_metadata() in data.py.
        Verify before deploying.
    """
    rows: dict[str, dict[str, str]] = {}
    for csv_path in session_dir.glob("*_metadata.csv"):
        try:
            with csv_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    filename = row.get("filename", "")
                    stem = Path(filename).stem if filename else ""
                    if stem:
                        rows[stem] = row
        except Exception as exc:
            logger.warning("Could not read metadata CSV %s: %s", csv_path, exc)
    return rows


def _row_to_detection_record(
    file_stem: str, row: dict[str, str] | None, model_name: str
) -> DetectionRecord:
    """Build a DetectionRecord from a metadata CSV row.

    Falls back to safe defaults if the row is missing or incomplete.

    Args:
        file_stem:  Bare file stem used as client_uuid.
        row:        CSV row dict, or None if the metadata CSV was not found.
        model_name: Active detection model identifier from UploadConfig.
    """
    if row is None:
        return DetectionRecord(
            client_uuid=file_stem,
            timestamp="",
            label="unknown",
            confidence=0.0,
            track_id=0,
            model_name=model_name,
        )
    return DetectionRecord(
        client_uuid=file_stem,
        timestamp=row.get("timestamp", ""),
        label=row.get("label", "unknown"),
        confidence=float(row.get("confidence") or 0.0),
        track_id=int(row.get("track_id") or 0),
        model_name=model_name,
    )
