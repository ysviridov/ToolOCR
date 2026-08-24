import csv
import io
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "importer"))
from toolocr_importer.spec import SPECS  # noqa: E402


def test_archive_contract(archive: Path):
    with zipfile.ZipFile(archive) as zf:
        for spec in SPECS:
            rx = re.compile(rf"(^|/){spec.prefix}_ALL_INS_\d{{14}}\.txt$", re.I)
            names = [n for n in zf.namelist() if rx.search(n)]
            assert len(names) == 1, (spec.prefix, names)
            with zf.open(names[0]) as raw:
                r = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig"), delimiter=";")
                assert tuple(next(r)) == spec.source_columns


if __name__ == "__main__":
    test_archive_contract(Path(sys.argv[1]))
    print("archive contract: OK")
