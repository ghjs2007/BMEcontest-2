from pathlib import Path

import numpy as np


def test_legacy_and_canonical_reader_are_equal(tmp_path: Path):
    from src.data.loader import load_session_tsv
    from src.pipeline.io.raw_session import RawSessionSource, load_raw_session

    raw = tmp_path / "collect_data1_2_3.txt"
    header = "ACC_TIME\tPPG_TIME\tGYRO_TIME\t" + "\t".join(f"v{i}" for i in range(50)) + "\n"
    raw.write_text(header + "100\t200\t100\t" + "\t".join(["1"] * 50) + "\n", encoding="utf-8")
    old = load_session_tsv(raw)
    new = load_raw_session(RawSessionSource(raw, "fixture", None))
    for name in ("acc", "gyro", "ppg", "t_acc", "t_ppg", "imu_valid", "ppg_valid"):
        np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
    assert old.meta["row_rate"] == new.meta["row_rate"]
