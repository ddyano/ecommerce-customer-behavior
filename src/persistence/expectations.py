"""Independent expectations registry for verified Parquet artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pyarrow.parquet as pq

from src.persistence.codec import VerificationResult, verify_parquet
from src.persistence.hashing import (
    FINGERPRINT_VERSION,
    canonical_json_bytes,
    fingerprint_dataframe,
    sha256_hex,
)
from src.persistence.memory_preflight import (
    DEFAULT_MAX_FOOTER_BYTES,
    MemoryLimits,
    estimate_registration,
    estimate_verification,
    inspect_parquet_footer,
    require_capacity,
)
from src.persistence.schema import (
    ANNOTATED_COLUMNS,
    SCHEMA_VERSION,
    schema_digest,
    to_persistence_view,
    validate_annotated_view,
)
from src.persistence.text_exceptions import MANIFEST_FIELDS, encode_descriptions

EXPECTATIONS_VERSION = 1
CODEC_VERSION = 1
DEFAULT_MAX_REGISTRY_BYTES = 1024 * 1024
DEFAULT_MAX_SNAPSHOT_BYTES = 1024 * 1024 * 1024
COPY_CHUNK_BYTES = 1024 * 1024
_CODEC_METADATA_KEY = b"ecommerce.persistence.codec.v1"
_KEY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_RECORD_FIELDS = frozenset(
    {
        "expectations_version",
        "codec_version",
        "schema_version",
        "schema_digest",
        "row_count",
        "columns",
        "fingerprint",
        "restoration_digest",
        "manifest_count",
        "manifest_digest",
    }
)
_CODEC_FIELDS = frozenset(
    {
        "codec_version",
        "schema_version",
        "schema_digest",
        "original_dtypes",
        "dynamic_categories",
        "dynamic_nulls",
        "description_dtype",
        "description_null_kinds",
        "exception_manifest",
        "fingerprint",
    }
)


class ExpectationsError(RuntimeError):
    """An expectation record or independently verified artifact is invalid."""


class ExpectationConflictError(ExpectationsError):
    """A different expectation is already registered for the requested key."""


@dataclass(frozen=True)
class ArtifactExpectation:
    expectations_version: int
    codec_version: int
    schema_version: int
    schema_digest: str
    row_count: int
    columns: tuple[str, ...]
    fingerprint: dict[str, Any]
    restoration_digest: str
    manifest_count: int
    manifest_digest: str

    def to_record(self) -> dict[str, Any]:
        return {
            "expectations_version": self.expectations_version,
            "codec_version": self.codec_version,
            "schema_version": self.schema_version,
            "schema_digest": self.schema_digest,
            "row_count": self.row_count,
            "columns": list(self.columns),
            "fingerprint": self.fingerprint,
            "restoration_digest": self.restoration_digest,
            "manifest_count": self.manifest_count,
            "manifest_digest": self.manifest_digest,
        }


def _safe_error(message: str) -> None:
    raise ExpectationsError(message) from None


def _registry_path(registry: Path | str, key: str) -> Path:
    if type(key) is not str or _KEY_PATTERN.fullmatch(key) is None:
        _safe_error("Expectation key is invalid.")
    root = Path(registry)
    if not root.is_dir():
        _safe_error("Expectations registry directory is unavailable.")
    filename = hashlib.sha256(key.encode("ascii")).hexdigest() + ".json"
    return root / filename


def _manifest_records(manifest: pd.DataFrame) -> list[dict[str, Any]]:
    if tuple(manifest.columns) != MANIFEST_FIELDS:
        _safe_error("Description exception manifest fields are invalid.")
    records: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False, name=None):
        row_id, original_type, text, source_sheet, stock_code = row
        records.append(
            {
                "row_id": int(row_id),
                "original_type": str(original_type),
                "text": str(text),
                "source_sheet": str(source_sheet),
                "stock_code": None if pd.isna(stock_code) else str(stock_code),
            }
        )
    records.sort(key=lambda item: item["row_id"])
    if len({item["row_id"] for item in records}) != len(records):
        _safe_error("Description exception manifest has duplicate row mappings.")
    return records


def _restoration_payload(persistence, encoded) -> dict[str, Any]:
    return {
        "original_dtypes": persistence.original_dtypes,
        "dynamic_categories": persistence.dynamic_categories,
        "dynamic_nulls": persistence.dynamic_nulls,
        "description_dtype": dict(persistence.original_dtypes)["Description"],
        "description_null_kinds": encoded.null_kinds,
    }


def build_expectation(
    trusted_frame: pd.DataFrame,
    *,
    limits: MemoryLimits = MemoryLimits(),
    available_bytes: int | None = None,
) -> ArtifactExpectation:
    """Derive an independent expectation from a trusted annotated frame."""
    estimate = estimate_registration(trusted_frame)
    require_capacity(estimate, limits, available_bytes=available_bytes)
    try:
        validate_annotated_view(trusted_frame)
        persistence = to_persistence_view(trusted_frame)
        encoded = encode_descriptions(trusted_frame)
        fingerprint = fingerprint_dataframe(trusted_frame).to_manifest_dict()
        manifest = _manifest_records(encoded.manifest)
        restoration_digest = sha256_hex(
            canonical_json_bytes(_restoration_payload(persistence, encoded))
        )
        manifest_digest = sha256_hex(canonical_json_bytes(manifest))
    except ExpectationsError:
        raise
    except Exception:
        _safe_error("Trusted frame cannot produce a valid expectation.")
    return ArtifactExpectation(
        EXPECTATIONS_VERSION,
        CODEC_VERSION,
        SCHEMA_VERSION,
        schema_digest(),
        len(trusted_frame),
        ANNOTATED_COLUMNS,
        fingerprint,
        restoration_digest,
        len(manifest),
        manifest_digest,
    )


def _parse_canonical_json(data: bytes, label: str) -> dict[str, Any]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    try:
        parsed = json.loads(data.decode("utf-8"), object_pairs_hook=unique_object)
        if not isinstance(parsed, dict) or canonical_json_bytes(parsed) != data:
            raise ValueError
        return parsed
    except Exception:
        _safe_error(f"{label} is malformed or noncanonical.")


def _expectation_from_record(record: Any) -> ArtifactExpectation:
    if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
        _safe_error("Expectation record fields are malformed.")
    fingerprint = record.get("fingerprint")
    if (
        type(record.get("expectations_version")) is not int
        or record["expectations_version"] != EXPECTATIONS_VERSION
        or type(record.get("codec_version")) is not int
        or record["codec_version"] != CODEC_VERSION
        or type(record.get("schema_version")) is not int
        or record["schema_version"] != SCHEMA_VERSION
        or type(record.get("schema_digest")) is not str
        or record["schema_digest"] != schema_digest()
        or type(record.get("row_count")) is not int
        or record["row_count"] < 0
        or not isinstance(record.get("columns"), list)
        or record["columns"] != list(ANNOTATED_COLUMNS)
        or not isinstance(fingerprint, dict)
        or type(record.get("restoration_digest")) is not str
        or type(record.get("manifest_count")) is not int
        or record["manifest_count"] < 0
        or type(record.get("manifest_digest")) is not str
    ):
        _safe_error("Expectation record values are malformed or unsupported.")
    for field in ("schema_digest", "restoration_digest", "manifest_digest"):
        value = record[field]
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            _safe_error("Expectation digest is malformed.")
    if (
        set(fingerprint) != {"fingerprint_version", "digest", "rows", "columns"}
        or type(fingerprint.get("fingerprint_version")) is not int
        or fingerprint["fingerprint_version"] != FINGERPRINT_VERSION
        or type(fingerprint.get("digest")) is not str
        or len(fingerprint["digest"]) != 64
        or fingerprint.get("rows") != record["row_count"]
        or not isinstance(fingerprint.get("columns"), list)
    ):
        _safe_error("Expectation fingerprint is malformed.")
    fingerprint_columns = fingerprint["columns"]
    if (
        len(fingerprint_columns) != len(record["columns"])
        or any(
            not isinstance(item, list)
            or len(item) != 2
            or item[0] != name
            or type(item[1]) is not str
            for name, item in zip(record["columns"], fingerprint_columns)
        )
    ):
        _safe_error("Expectation fingerprint columns are malformed.")
    if any(char not in "0123456789abcdef" for char in fingerprint["digest"]):
        _safe_error("Expectation fingerprint digest is malformed.")
    return ArtifactExpectation(
        record["expectations_version"],
        record["codec_version"],
        record["schema_version"],
        record["schema_digest"],
        record["row_count"],
        tuple(record["columns"]),
        fingerprint,
        record["restoration_digest"],
        record["manifest_count"],
        record["manifest_digest"],
    )


def _envelope_bytes(expectation: ArtifactExpectation) -> bytes:
    record = expectation.to_record()
    return canonical_json_bytes(
        {
            "record": record,
            "record_digest": sha256_hex(canonical_json_bytes(record)),
        }
    )


def lookup_expectation(
    registry: Path | str,
    key: str,
    *,
    max_record_bytes: int = DEFAULT_MAX_REGISTRY_BYTES,
) -> ArtifactExpectation:
    path = _registry_path(registry, key)
    if type(max_record_bytes) is not int or max_record_bytes <= 0:
        _safe_error("Expectation record size limit is invalid.")
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            data = stream.read(max_record_bytes + 1)
            after = os.fstat(stream.fileno())
    except ExpectationsError:
        raise
    except FileNotFoundError:
        _safe_error("Expectation record is missing.")
    except OSError:
        _safe_error("Expectation record cannot be read safely.")
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        _safe_error("Expectation record changed or was truncated while it was read.")
    if before.st_size <= 0 or before.st_size > max_record_bytes:
        _safe_error("Expectation record size is invalid or exceeds its limit.")
    if len(data) != before.st_size:
        _safe_error("Expectation record changed or was truncated while it was read.")
    envelope = _parse_canonical_json(data, "Expectation record")
    if set(envelope) != {"record", "record_digest"}:
        _safe_error("Expectation record envelope is malformed.")
    record = envelope["record"]
    digest = envelope["record_digest"]
    if not isinstance(record, dict) or type(digest) is not str:
        _safe_error("Expectation record envelope is malformed.")
    try:
        expected_digest = sha256_hex(canonical_json_bytes(record))
    except Exception:
        _safe_error("Expectation record contents are malformed.")
    if digest != expected_digest:
        _safe_error("Expectation record checksum does not match.")
    return _expectation_from_record(record)


def register_expectation(
    registry: Path | str,
    key: str,
    trusted_frame: pd.DataFrame,
    *,
    limits: MemoryLimits = MemoryLimits(),
    available_bytes: int | None = None,
) -> ArtifactExpectation:
    """Atomically publish a create-only expectation record."""
    destination = _registry_path(registry, key)
    expectation = build_expectation(
        trusted_frame, limits=limits, available_bytes=available_bytes
    )
    data = _envelope_bytes(expectation)
    try:
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    except OSError:
        _safe_error("Expectation record could not be written safely.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = lookup_expectation(registry, key)
            if existing != expectation:
                raise ExpectationConflictError(
                    "A conflicting expectation is already registered."
                ) from None
        except OSError:
            _safe_error("Expectation record could not be published atomically.")
        return expectation
    except ExpectationsError:
        raise
    except OSError:
        _safe_error("Expectation record could not be written safely.")
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


@contextmanager
def _artifact_snapshot(
    path: Path | str,
    *,
    max_snapshot_bytes: int = DEFAULT_MAX_SNAPSHOT_BYTES,
) -> Iterator[Path]:
    source_path = Path(path)
    if not source_path.is_file():
        _safe_error("Artifact is missing or is not a regular file.")
    if type(max_snapshot_bytes) is not int or max_snapshot_bytes <= 0:
        _safe_error("Artifact snapshot size limit is invalid.")
    handle = -1
    snapshot: Path | None = None
    try:
        with source_path.open("rb") as source:
            before = os.fstat(source.fileno())
            if before.st_size > max_snapshot_bytes:
                _safe_error("Artifact exceeds the configured snapshot size limit.")
            handle, name = tempfile.mkstemp(
                prefix=f".{source_path.name}.",
                suffix=".verify",
                dir=source_path.parent,
            )
            snapshot = Path(name)
            copied = 0
            with os.fdopen(handle, "wb") as target:
                handle = -1
                while True:
                    block = source.read(COPY_CHUNK_BYTES)
                    if not block:
                        break
                    if copied + len(block) > max_snapshot_bytes:
                        _safe_error("Artifact exceeds the configured snapshot size limit.")
                    target.write(block)
                    copied += len(block)
                target.flush()
                os.fsync(target.fileno())
            after = os.fstat(source.fileno())
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after or copied != before.st_size:
            _safe_error("Artifact changed while its verification snapshot was created.")
        yield snapshot
    except ExpectationsError:
        raise
    except OSError:
        _safe_error("Artifact verification snapshot could not be created safely.")
    finally:
        if handle >= 0:
            try:
                os.close(handle)
            except OSError:
                pass
        if snapshot is not None:
            try:
                snapshot.unlink(missing_ok=True)
            except OSError:
                pass


def _read_codec_metadata(snapshot: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    parquet = None
    try:
        parquet = pq.ParquetFile(snapshot)
        names = tuple(parquet.schema_arrow.names)
        raw = (parquet.schema_arrow.metadata or {}).get(_CODEC_METADATA_KEY)
    except Exception:
        _safe_error("Artifact footer metadata cannot be read safely.")
    finally:
        if parquet is not None:
            parquet.close()
    if raw is None:
        _safe_error("Artifact codec metadata is missing.")
    payload = _parse_canonical_json(raw, "Artifact codec metadata")
    if set(payload) != _CODEC_FIELDS:
        _safe_error("Artifact codec metadata fields are malformed.")
    return payload, names


def _artifact_expectation(payload: dict[str, Any], names: tuple[str, ...]) -> ArtifactExpectation:
    restoration = {
        "original_dtypes": payload.get("original_dtypes"),
        "dynamic_categories": payload.get("dynamic_categories"),
        "dynamic_nulls": payload.get("dynamic_nulls"),
        "description_dtype": payload.get("description_dtype"),
        "description_null_kinds": payload.get("description_null_kinds"),
    }
    manifest = payload.get("exception_manifest")
    if not isinstance(manifest, list):
        _safe_error("Artifact exception manifest is malformed.")
    checked: list[dict[str, Any]] = []
    for record in manifest:
        if not isinstance(record, dict) or set(record) != set(MANIFEST_FIELDS):
            _safe_error("Artifact exception manifest fields are malformed.")
        if (
            type(record.get("row_id")) is not int
            or any(
                type(record.get(field)) is not str
                for field in ("original_type", "text", "source_sheet")
            )
            or (
                record.get("stock_code") is not None
                and type(record.get("stock_code")) is not str
            )
        ):
            _safe_error("Artifact exception manifest values are malformed.")
        checked.append(record)
    checked.sort(key=lambda item: item["row_id"])
    if len({item["row_id"] for item in checked}) != len(checked):
        _safe_error("Artifact exception manifest has duplicate row mappings.")
    try:
        record = {
            "expectations_version": EXPECTATIONS_VERSION,
            "codec_version": payload.get("codec_version"),
            "schema_version": payload.get("schema_version"),
            "schema_digest": payload.get("schema_digest"),
            "row_count": payload.get("fingerprint", {}).get("rows")
            if isinstance(payload.get("fingerprint"), dict)
            else None,
            "columns": list(names),
            "fingerprint": payload.get("fingerprint"),
            "restoration_digest": sha256_hex(canonical_json_bytes(restoration)),
            "manifest_count": len(checked),
            "manifest_digest": sha256_hex(canonical_json_bytes(checked)),
        }
    except Exception:
        _safe_error("Artifact restoration metadata is malformed.")
    return _expectation_from_record(record)


def verify_expected_artifact(
    registry: Path | str,
    key: str,
    artifact: Path | str,
    *,
    limits: MemoryLimits = MemoryLimits(),
    available_bytes: int | None = None,
    max_footer_bytes: int = DEFAULT_MAX_FOOTER_BYTES,
    max_snapshot_bytes: int = DEFAULT_MAX_SNAPSHOT_BYTES,
) -> VerificationResult:
    """Verify independent expectations and then run full codec verification.

    Snapshots default to a conservative one-GiB maximum and are copied in
    bounded chunks. Callers may lower or explicitly raise that positive limit.
    """
    expected = lookup_expectation(registry, key)
    with _artifact_snapshot(
        artifact, max_snapshot_bytes=max_snapshot_bytes
    ) as snapshot:
        footer = inspect_parquet_footer(
            snapshot, max_footer_bytes=max_footer_bytes
        )
        estimate = estimate_verification(footer)
        require_capacity(estimate, limits, available_bytes=available_bytes)
        payload, names = _read_codec_metadata(snapshot)
        reported = _artifact_expectation(payload, names)
        if reported != expected:
            _safe_error("Artifact metadata does not match its independent expectation.")
        try:
            result = verify_parquet(snapshot)
        except Exception:
            _safe_error("Artifact failed full codec verification.")
        if (
            result.rows != expected.row_count
            or result.columns != len(expected.columns)
            or result.schema_version != expected.schema_version
            or result.fingerprint != expected.fingerprint["digest"]
        ):
            _safe_error("Verified artifact does not match its independent expectation.")
        return result
