from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg import sql

from .spec import SPECS, FileSpec

DB_DSN = os.environ.get("TOOLOCR_DB_DSN", "")


class ImportFailure(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def timestamp_from_name(name: str) -> datetime | None:
    match = re.search(r"(20\d{12})", name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def locate_member(zf: zipfile.ZipFile, spec: FileSpec) -> str:
    pattern = re.compile(rf"(^|/){re.escape(spec.prefix)}_ALL_INS_\d{{14}}\.txt$", re.IGNORECASE)
    matches = [n for n in zf.namelist() if pattern.search(n)]
    if len(matches) != 1:
        raise ImportFailure(f"Expected exactly one {spec.prefix}_ALL_INS_*.txt, found {matches!r}")
    return matches[0]


def read_header(zf: zipfile.ZipFile, member: str) -> tuple[str, ...]:
    with zf.open(member, "r") as raw:
        wrapper = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        reader = csv.reader(wrapper, delimiter=";")
        try:
            return tuple(next(reader))
        except StopIteration:
            raise ImportFailure(f"File {member} is empty")


def ensure_archive_contract(zf: zipfile.ZipFile) -> dict[str, str]:
    members: dict[str, str] = {}
    for spec in SPECS:
        member = locate_member(zf, spec)
        header = read_header(zf, member)
        if header != spec.source_columns:
            raise ImportFailure(
                f"Unexpected header for {member}:\n"
                f"  got:      {header}\n"
                f"  expected: {spec.source_columns}"
            )
        members[spec.prefix] = member
    return members


def create_dataset(conn: psycopg.Connection, archive: Path, checksum: str) -> int:
    source_ts = timestamp_from_name(archive.name)
    with conn.cursor() as cur:
        cur.execute("SELECT dataset_id, status FROM toolocr.dataset WHERE source_sha256 = %s", (checksum,))
        existing = cur.fetchone()
        if existing:
            if existing[1] == "failed":
                # A transient failure (disk full, interrupted COPY, etc.) must not permanently
                # block retrying the same customer snapshot. CASCADE removes partial rows.
                cur.execute("DELETE FROM toolocr.dataset WHERE dataset_id = %s", (existing[0],))
            else:
                raise ImportFailure(f"This archive is already registered as dataset_id={existing[0]} status={existing[1]}")
        cur.execute(
            """
            INSERT INTO toolocr.dataset(source_filename, source_timestamp, source_sha256, status)
            VALUES (%s, %s, %s, 'loading')
            RETURNING dataset_id
            """,
            (archive.name, source_ts, checksum),
        )
        dataset_id = cur.fetchone()[0]
    conn.commit()
    return dataset_id


def copy_member(conn: psycopg.Connection, zf: zipfile.ZipFile, member: str, spec: FileSpec, dataset_id: int) -> int:
    columns = ("dataset_id",) + spec.db_columns
    copy_stmt = sql.SQL(
        "COPY toolocr.{} ({}) FROM STDIN WITH (FORMAT CSV, DELIMITER ';', QUOTE '\"', ESCAPE '\"', NULL '')"
    ).format(
        sql.Identifier(spec.table),
        sql.SQL(", ").join(map(sql.Identifier, columns)),
    )

    row_count = 0
    with zf.open(member, "r") as raw, conn.cursor() as cur:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        # Consume exactly one header line; contract was already validated with csv.reader.
        text.readline()
        with cur.copy(copy_stmt) as cp:
            prefix = f"{dataset_id};"
            batch: list[str] = []
            batch_size = 10_000
            for line in text:
                if not line.strip():
                    continue
                batch.append(prefix + line)
                row_count += 1
                if len(batch) >= batch_size:
                    cp.write("".join(batch))
                    batch.clear()
            if batch:
                cp.write("".join(batch))
    conn.commit()
    return row_count


def validate_dataset(conn: psycopg.Connection, dataset_id: int, row_counts: dict[str, int]) -> dict:
    errors: list[str] = []
    checks: dict[str, object] = {}

    for spec in SPECS:
        count = row_counts.get(spec.prefix, 0)
        checks[f"rows_{spec.prefix}"] = count
        if count < spec.min_rows:
            errors.append(f"{spec.prefix}: row count {count} below safety threshold {spec.min_rows}")

    with conn.cursor() as cur:
        # One pass through address_range; all reference tables are indexed by (dataset_id, id).
        cur.execute(
            """
            SELECT count(*)
            FROM toolocr.address_range ar
            LEFT JOIN toolocr.postal_code pc
              ON pc.dataset_id = ar.dataset_id AND pc.postal_code = ar.postal_code
            LEFT JOIN toolocr.federal_subject fs
              ON fs.dataset_id = ar.dataset_id AND fs.id_subject = ar.id_subject
            LEFT JOIN toolocr.district dt
              ON dt.dataset_id = ar.dataset_id AND dt.id_district = ar.id_district
            LEFT JOIN toolocr.main_city mc
              ON mc.dataset_id = ar.dataset_id AND mc.id_main_city = ar.id_main_city
            LEFT JOIN toolocr.city ct
              ON ct.dataset_id = ar.dataset_id AND ct.id_city = ar.id_city
            LEFT JOIN toolocr.street sr
              ON sr.dataset_id = ar.dataset_id AND sr.id_street = ar.id_street
            WHERE ar.dataset_id = %s
              AND (
                (ar.postal_code IS NOT NULL AND pc.postal_code IS NULL) OR
                (ar.id_subject IS NOT NULL AND fs.id_subject IS NULL) OR
                (ar.id_district IS NOT NULL AND dt.id_district IS NULL) OR
                (ar.id_main_city IS NOT NULL AND mc.id_main_city IS NULL) OR
                (ar.id_city IS NOT NULL AND ct.id_city IS NULL) OR
                (ar.id_street IS NOT NULL AND sr.id_street IS NULL)
              )
            """,
            (dataset_id,),
        )
        ar_orphans = cur.fetchone()[0]
        checks["address_reference_orphans"] = ar_orphans
        if ar_orphans:
            errors.append(f"address_range contains {ar_orphans} rows with broken references")

        cur.execute(
            """
            SELECT count(*)
            FROM toolocr.district d
            LEFT JOIN toolocr.federal_subject s
              ON s.dataset_id=d.dataset_id AND s.id_subject=d.id_subject
            WHERE d.dataset_id=%s AND d.id_subject IS NOT NULL AND s.id_subject IS NULL
            """,
            (dataset_id,),
        )
        district_orphans = cur.fetchone()[0]
        checks["district_subject_orphans"] = district_orphans
        if district_orphans:
            errors.append(f"district contains {district_orphans} broken subject references")

        cur.execute(
            """
            SELECT count(*)
            FROM toolocr.main_city m
            LEFT JOIN toolocr.federal_subject s
              ON s.dataset_id=m.dataset_id AND s.id_subject=m.id_subject
            LEFT JOIN toolocr.district d
              ON d.dataset_id=m.dataset_id AND d.id_district=m.id_district
            WHERE m.dataset_id=%s AND (
                (m.id_subject IS NOT NULL AND s.id_subject IS NULL) OR
                (m.id_district IS NOT NULL AND d.id_district IS NULL)
            )
            """,
            (dataset_id,),
        )
        main_city_orphans = cur.fetchone()[0]
        checks["main_city_reference_orphans"] = main_city_orphans
        if main_city_orphans:
            errors.append(f"main_city contains {main_city_orphans} broken references")

        cur.execute(
            """
            SELECT count(*)
            FROM toolocr.city c
            LEFT JOIN toolocr.main_city m
              ON m.dataset_id=c.dataset_id AND m.id_main_city=c.id_main_city
            WHERE c.dataset_id=%s AND c.id_main_city IS NOT NULL AND m.id_main_city IS NULL
            """,
            (dataset_id,),
        )
        city_orphans = cur.fetchone()[0]
        checks["city_main_city_orphans"] = city_orphans
        if city_orphans:
            errors.append(f"city contains {city_orphans} broken main_city references")

    checks["ok"] = not errors
    checks["errors"] = errors
    return checks


def save_validation(conn: psycopg.Connection, dataset_id: int, row_counts: dict[str, int], validation: dict) -> None:
    status = "ready" if validation["ok"] else "failed"
    error_text = None if validation["ok"] else "\n".join(validation["errors"])
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE toolocr.dataset
               SET status=%s,
                   row_counts_json=%s::jsonb,
                   validation_json=%s::jsonb,
                   error_text=%s
             WHERE dataset_id=%s
            """,
            (status, json.dumps(row_counts), json.dumps(validation), error_text, dataset_id),
        )
    conn.commit()


def activate_dataset(conn: psycopg.Connection, dataset_id: int) -> None:
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM toolocr.dataset WHERE dataset_id=%s FOR UPDATE", (dataset_id,))
            row = cur.fetchone()
            if not row:
                raise ImportFailure(f"dataset_id={dataset_id} does not exist")
            if row[0] not in ("ready", "active", "retired"):
                raise ImportFailure(f"dataset_id={dataset_id} cannot be activated from status={row[0]}")

            cur.execute("SELECT active_dataset_id FROM toolocr.runtime_state WHERE singleton=true FOR UPDATE")
            old_id = cur.fetchone()[0]
            if old_id == dataset_id:
                return
            if old_id is not None:
                cur.execute(
                    "UPDATE toolocr.dataset SET status='retired' WHERE dataset_id=%s AND status='active'",
                    (old_id,),
                )
            cur.execute(
                "UPDATE toolocr.dataset SET status='active', activated_at=now(), error_text=NULL WHERE dataset_id=%s",
                (dataset_id,),
            )
            cur.execute(
                "UPDATE toolocr.runtime_state SET active_dataset_id=%s, updated_at=now() WHERE singleton=true",
                (dataset_id,),
            )


def mark_failed(conn: psycopg.Connection, dataset_id: int, error: Exception) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE toolocr.dataset SET status='failed', error_text=%s WHERE dataset_id=%s",
                (str(error)[:10000], dataset_id),
            )
        conn.commit()
    except Exception:
        conn.rollback()


def import_archive(path: Path, activate: bool) -> int:
    if not path.is_file():
        raise ImportFailure(f"Archive not found: {path}")

    checksum = sha256_file(path)
    print(f"Archive: {path}")
    print(f"SHA256:  {checksum}")

    with zipfile.ZipFile(path) as zf:
        members = ensure_archive_contract(zf)
        print("Archive contract: OK")

        with psycopg.connect(DB_DSN, autocommit=False) as conn:
            # Only one full snapshot loader at a time. The lock is session-scoped and
            # automatically released even if the importer container dies.
            conn.execute("SELECT pg_advisory_lock(hashtext('toolocr.address_snapshot_import'))")
            conn.commit()
            dataset_id = create_dataset(conn, path, checksum)
            print(f"dataset_id={dataset_id} created")
            try:
                row_counts: dict[str, int] = {}
                for spec in SPECS:
                    print(f"Loading {spec.prefix} -> toolocr.{spec.table} ...", flush=True)
                    count = copy_member(conn, zf, members[spec.prefix], spec, dataset_id)
                    row_counts[spec.prefix] = count
                    print(f"  {count:,} rows")

                print("Validating references and safety thresholds ...", flush=True)
                validation = validate_dataset(conn, dataset_id, row_counts)
                save_validation(conn, dataset_id, row_counts, validation)
                print(json.dumps(validation, ensure_ascii=False, indent=2))
                if not validation["ok"]:
                    raise ImportFailure("Dataset validation failed; active dataset was not changed")

                # Refresh optimizer statistics for the newly loaded snapshot.
                conn.execute("ANALYZE toolocr.federal_subject")
                conn.execute("ANALYZE toolocr.district")
                conn.execute("ANALYZE toolocr.postal_code")
                conn.execute("ANALYZE toolocr.main_city")
                conn.execute("ANALYZE toolocr.city")
                conn.execute("ANALYZE toolocr.street")
                conn.execute("ANALYZE toolocr.address_range")
                conn.commit()

                if activate:
                    activate_dataset(conn, dataset_id)
                    print(f"dataset_id={dataset_id} is now ACTIVE")
                else:
                    print(f"dataset_id={dataset_id} is READY (not activated)")
                return dataset_id
            except Exception as exc:
                conn.rollback()
                mark_failed(conn, dataset_id, exc)
                raise


def list_datasets() -> None:
    with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT dataset_id, source_filename, source_timestamp, status,
                   imported_at, activated_at, row_counts_json, error_text
            FROM toolocr.dataset
            ORDER BY dataset_id DESC
            """
        )
        for row in cur.fetchall():
            print(json.dumps({
                "dataset_id": row[0],
                "source_filename": row[1],
                "source_timestamp": row[2].isoformat() if row[2] else None,
                "status": row[3],
                "imported_at": row[4].isoformat() if row[4] else None,
                "activated_at": row[5].isoformat() if row[5] else None,
                "row_counts": row[6],
                "error": row[7],
            }, ensure_ascii=False, default=str))


def prune(keep: int) -> None:
    if keep < 1:
        raise ImportFailure("--keep must be >= 1")
    with psycopg.connect(DB_DSN, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT active_dataset_id FROM toolocr.runtime_state WHERE singleton=true")
            active_id = cur.fetchone()[0]
            cur.execute(
                """
                SELECT dataset_id
                FROM toolocr.dataset
                WHERE status IN ('active', 'retired', 'ready')
                ORDER BY dataset_id DESC
                """
            )
            candidates = [r[0] for r in cur.fetchall()]
            protected = set(candidates[:keep])
            if active_id is not None:
                protected.add(active_id)
            to_delete = [x for x in candidates if x not in protected]
            if to_delete:
                cur.execute("DELETE FROM toolocr.dataset WHERE dataset_id = ANY(%s)", (to_delete,))
        conn.commit()
    print(f"Pruned datasets: {to_delete if to_delete else 'none'}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="toolocr-importer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import", help="Load and validate a full ADDRESS_*.zip snapshot")
    p_import.add_argument("archive", type=Path)
    p_import.add_argument("--activate", action="store_true", help="Atomically make the validated dataset active")

    p_activate = sub.add_parser("activate", help="Atomically switch the active dataset (also provides rollback)")
    p_activate.add_argument("dataset_id", type=int)

    sub.add_parser("list", help="List imported datasets")

    p_prune = sub.add_parser("prune", help="Delete old ready/retired datasets; active is always protected")
    p_prune.add_argument("--keep", type=int, default=2)

    args = parser.parse_args()
    try:
        if args.command == "import":
            import_archive(args.archive, args.activate)
        elif args.command == "activate":
            with psycopg.connect(DB_DSN, autocommit=False) as conn:
                activate_dataset(conn, args.dataset_id)
            print(f"dataset_id={args.dataset_id} is ACTIVE")
        elif args.command == "list":
            list_datasets()
        elif args.command == "prune":
            prune(args.keep)
        return 0
    except (ImportFailure, zipfile.BadZipFile, psycopg.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
