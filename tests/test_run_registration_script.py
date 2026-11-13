"""scripts/run_registration.sh: argument checks, dry run and resumption, with stub tools."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_registration.sh"
TOOLS = ("visium_hd_tiff_decimate", "visium_hd_capture_grid", "visium_hd_register",
         "visium_hd_count_register")


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in TOOLS:
        tool = bin_dir / name
        tool.write_text("#!/bin/sh\nexit 0\n")
        tool.chmod(0o755)
    counts = tmp_path / "SpatialGex.out"
    for size in ("002", "008", "016"):
        folder = counts / "hard" / f"square_{size}um"
        folder.mkdir(parents=True)
        (folder / "matrix.mtx").write_text("")
    he = tmp_path / "he.btf"
    he.write_bytes(b"")
    cytassist = tmp_path / "cytassist.tif"
    cytassist.write_bytes(b"")
    layout = tmp_path / "H1-TEST.vlf"
    layout.write_text('{"slide_uid": "H1-TEST", "capture_areas": {"A1": [], "D1": []}}')
    return {"bin": bin_dir, "counts": counts, "he": he, "cytassist": cytassist,
            "layout": layout, "out": tmp_path / "out"}


def argv(inputs: dict[str, Path], *extra: str) -> list[str]:
    return [str(SCRIPT), "--he", str(inputs["he"]), "--cytassist", str(inputs["cytassist"]),
            "--layout", str(inputs["layout"]), "--area", "D1", "--counts", str(inputs["counts"]),
            "--out-dir", str(inputs["out"]), "--bin-dir", str(inputs["bin"]), *extra]


def test_dry_run_prints_every_step_and_writes_nothing(inputs):
    result = subprocess.run(argv(inputs, "--dry-run", "--pixel-size-um", "0.2738"),
                            capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    for expected in ("visium_hd_tiff_decimate", "visium_hd_capture_grid", "visium_hd_register",
                     "segment_hd_he_cellpose.py", "visium_hd_count_register project",
                     "visium_hd_count_register correct"):
        assert any(expected in line for line in lines), expected
    assert "--profile independent_v1" in result.stdout
    assert "--model-sha256 e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2" in result.stdout
    assert "--source-pixel-size-um 0.2738" in result.stdout
    register = next(line for line in lines if "visium_hd_register" in line)
    assert register.rstrip().endswith("--threads 1")
    assert not inputs["out"].exists()


def test_a_finished_segmentation_is_used_as_given(inputs, tmp_path):
    segmentation = tmp_path / "cellpose"
    segmentation.mkdir()
    (segmentation / "summary.json").write_text("{}")
    result = subprocess.run(argv(inputs, "--dry-run", "--segmentation", str(segmentation)),
                            capture_output=True, text=True, check=True)
    assert "segment_hd_he_cellpose.py" not in result.stdout
    assert f"--segmentation {segmentation}" in result.stdout


def test_missing_counts_are_rejected(inputs):
    (inputs["counts"] / "hard" / "square_016um" / "matrix.mtx").unlink()
    result = subprocess.run(argv(inputs, "--dry-run"), capture_output=True, text=True)
    assert result.returncode == 2
    assert "square_016um" in result.stderr


def test_unknown_capture_area_is_rejected(inputs):
    command = argv(inputs, "--dry-run")
    command[command.index("D1")] = "B1"
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert "no capture area B1" in result.stderr


def test_an_incomplete_step_is_not_overwritten(inputs, tmp_path):
    (inputs["out"] / "inputs").mkdir(parents=True)
    (inputs["out"] / "inputs" / "he_ds16.tif").write_bytes(b"")
    (inputs["out"] / "capture_grid").mkdir()
    result = subprocess.run(argv(inputs, "--segmentation", str(tmp_path)), capture_output=True,
                            text=True)
    assert result.returncode == 2
    assert "must hold a finished segmentation" in result.stderr
    (tmp_path / "summary.json").write_text("{}")
    result = subprocess.run(argv(inputs, "--segmentation", str(tmp_path)), capture_output=True,
                            text=True)
    assert result.returncode == 2
    # Once the output folder exists, messages go to its log and to standard output.
    assert "capture_grid is incomplete" in result.stdout
    assert "capture_grid is incomplete" in (inputs["out"] / "run_registration.log").read_text()
