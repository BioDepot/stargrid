#!/usr/bin/env bash
# Register the microscope (H&E) image of a Visium HD slide to its capture grid,
# as STARgrid does, from the image files and STARgrid's count matrices.
#
# First stage (images only):
#   visium_hd_tiff_decimate   the H&E image reduced 16-fold for matching
#   visium_hd_capture_grid    fiducials of the CytAssist image -> capture grid
#   visium_hd_register        H&E image -> CytAssist image (SIFT, RANSAC)
# Nuclei:
#   scripts/publication/segment_hd_he_cellpose.py, Cellpose-SAM (cpsam)
# Second stage (nuclei and counts):
#   visium_hd_count_register project   nuclei onto the capture grid
#   visium_hd_count_register correct   the shift from the profile of counts
#                                      across the nucleus edge; the corrected
#                                      positions of every bin
#
# Every step is skipped when its output is already complete, so a run can be
# resumed. The settings are those of the STARgrid manuscript's runs.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
    cat <<'EOF'
usage: run_registration.sh --he FILE --cytassist FILE --layout FILE --area NAME
                           --counts DIR --out-dir DIR [options]

Required:
  --he FILE             microscope (H&E) image of the slide, TIFF or BigTIFF
  --cytassist FILE      CytAssist image of the same capture area
  --layout FILE         layout file of the slide (<serial>.vlf), from 10x Genomics
  --area NAME           capture area on the slide, for example A1 or D1
  --counts DIR          STARgrid's SpatialGex.out or SpatialFlex.out; needs
                        hard/square_002um, square_008um and square_016um
  --out-dir DIR         output folder

Nuclei (Cellpose-SAM):
  --segmentation DIR    use this finished segmentation instead of running Cellpose
  --pixel-size-um F     micrometres per pixel of --he (default: from the image's
                        OME-TIFF metadata). It sets only the cell outlines, which
                        the registration does not use.
  --cellpose-model FILE Cellpose-SAM model file (default ~/.cellpose/models/cpsam,
                        where Cellpose saves it); it must be the manuscript's
                        version (SHA-256 e1440429...)
  --python PATH         Python with Cellpose 4.1.1, tifffile and zarr (default python3)
  --container IMAGE     run Cellpose in this Docker image instead, for example one
                        built from containers/cellpose-paper/Dockerfile; extra
                        docker run options can be given in DOCKER_RUN_OPTS
  --cpu                 run Cellpose on the CPU (much slower)

Options:
  --threads N           threads (default 16)
  --bin-dir DIR         folder with the C++ tools (default: build/ in this
                        repository, then PATH)
  --dry-run             print the commands without running them
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

he=
cytassist=
layout=
area=
counts=
out_dir=
segmentation=
pixel_size=
cellpose_model=
python=python3
container=
cpu=0
threads=16
bin_dir=
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --he) he=$2; shift 2 ;;
        --cytassist) cytassist=$2; shift 2 ;;
        --layout) layout=$2; shift 2 ;;
        --area) area=$2; shift 2 ;;
        --counts) counts=$2; shift 2 ;;
        --out-dir) out_dir=$2; shift 2 ;;
        --segmentation) segmentation=$2; shift 2 ;;
        --pixel-size-um) pixel_size=$2; shift 2 ;;
        --cellpose-model) cellpose_model=$2; shift 2 ;;
        --python) python=$2; shift 2 ;;
        --container) container=$2; shift 2 ;;
        --cpu) cpu=1; shift ;;
        --threads) threads=$2; shift 2 ;;
        --bin-dir) bin_dir=$2; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown option: $1" ;;
    esac
done

[[ -n "$he" && -f "$he" ]] || die "--he must be an existing image file"
[[ -n "$cytassist" && -f "$cytassist" ]] || die "--cytassist must be an existing image file"
[[ -n "$layout" && -f "$layout" ]] || die "--layout must be the slide's layout file"
[[ "$area" =~ ^[A-Z][0-9]$ ]] || die "--area must name a capture area, for example D1"
grep -q "\"$area\"" "$layout" || die "$layout has no capture area $area"
[[ -n "$counts" ]] || die "--counts is required"
for size in 002 008 016; do
    [[ -f "$counts/hard/square_${size}um/matrix.mtx" || -f "$counts/hard/square_${size}um/matrix.mtx.gz" ]] \
        || die "$counts lacks hard/square_${size}um; give STARgrid's SpatialGex.out or SpatialFlex.out"
done
[[ -n "$out_dir" ]] || die "--out-dir is required"
[[ "$threads" =~ ^[1-9][0-9]*$ ]] || die "--threads must be a positive integer"
[[ -z "$pixel_size" || "$pixel_size" =~ ^[0-9]*\.?[0-9]+$ ]] || die "--pixel-size-um must be a positive number"
[[ -z "$segmentation" || -f "$segmentation/summary.json" ]] || die "--segmentation must hold a finished segmentation (summary.json)"

# The C++ tools: --bin-dir, this repository's build/, then PATH.
tools=(visium_hd_tiff_decimate visium_hd_capture_grid visium_hd_register visium_hd_count_register)
if [[ -z "$bin_dir" && -x "$repo_root/build/visium_hd_register" ]]; then
    bin_dir=$repo_root/build
fi
declare -A tool
for name in "${tools[@]}"; do
    if [[ -n "$bin_dir" ]]; then
        tool[$name]=$bin_dir/$name
    else
        tool[$name]=$(command -v "$name" || true)
    fi
    [[ -n "${tool[$name]}" && -x "${tool[$name]}" ]] \
        || die "$name not found; build the C++ tools (see README) or give --bin-dir"
done

absolute() {
    (cd "$(dirname "$1")" && echo "$(pwd)/$(basename "$1")")
}
he=$(absolute "$he")
if [[ $dry_run -eq 0 ]]; then
    mkdir -p "$out_dir"
    out_dir=$(cd "$out_dir" && pwd)
    # Keep a log beside the outputs.
    exec > >(tee -a "$out_dir/run_registration.log") 2>&1
fi

run() {
    printf '%q ' "$@"
    echo
    if [[ $dry_run -eq 0 ]]; then
        "$@"
    fi
}

# A step runs unless its output is complete; an incomplete output is not overwritten.
pending() {
    local done_file=$1 dir=$2
    [[ -f "$done_file" ]] && { echo "# done: $dir"; return 1; }
    [[ -e "$dir" && $dry_run -eq 0 ]] && die "$dir is incomplete; remove it and run again"
    return 0
}

# Python for Cellpose, locally or in a container with the same paths.
seg_python() {
    if [[ -n "$container" ]]; then
        local mounts=(-v "$repo_root:$repo_root:ro" -v "$(dirname "$he"):$(dirname "$he"):ro" -v "$out_dir:$out_dir")
        [[ -n "$cellpose_model" ]] && mounts+=(-v "$(dirname "$cellpose_model"):$(dirname "$cellpose_model"):ro")
        mkdir -p "$HOME/.cellpose"
        mounts+=(-v "$HOME/.cellpose:$HOME/.cellpose")
        local gpu=(--gpus all) extra=()
        [[ $cpu -eq 1 ]] && gpu=()
        read -r -a extra <<< "${DOCKER_RUN_OPTS:-}"
        docker run --rm "${gpu[@]}" "${extra[@]}" --user "$(id -u):$(id -g)" -e HOME="$HOME" \
            -e PYTHONDONTWRITEBYTECODE=1 -e MPLCONFIGDIR=/tmp/matplotlib "${mounts[@]}" -w "$repo_root" \
            --entrypoint python "$container" "$@"
    else
        "$python" "$@"
    fi
}

# The Cellpose-SAM weights used for the manuscript.
cpsam_sha256=e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2

ds16=$out_dir/inputs/he_ds16.tif
capture=$out_dir/capture_grid
registration=$out_dir/registration
projection=$out_dir/projection
registered=$out_dir/registered
[[ -n "$segmentation" ]] || segmentation=$out_dir/segmentation

[[ ! -f "$registered/COMPLETE.json" ]] || die "$registered is already complete; use a new --out-dir"

# First stage: H&E -> CytAssist -> capture grid. No counts, nuclei or Space
# Ranger output enter it. On one thread the feature matching is deterministic.
if [[ $dry_run -eq 1 || ! -f "$ds16" ]]; then
    [[ $dry_run -eq 0 ]] && mkdir -p "$out_dir/inputs"
    run "${tool[visium_hd_tiff_decimate]}" --input "$he" --output "$ds16" \
        --summary "$out_dir/inputs/he_ds16.json" --factor 16 --threads "$threads"
fi
if pending "$capture/capture_grid.json" "$capture"; then
    run "${tool[visium_hd_capture_grid]}" --cytassist "$cytassist" --vlf "$layout" --area "$area" \
        --threads "$threads" --out-dir "$capture"
fi
if pending "$registration/registration.json" "$registration"; then
    run "${tool[visium_hd_register]}" --moving "$ds16" --fixed "$cytassist" \
        --out-dir "$registration" --threads 1
fi

# Nuclei: Cellpose-SAM with the manuscript's settings (profile independent_v1).
if pending "$segmentation/summary.json" "$segmentation"; then
    if [[ -z "$container" && $dry_run -eq 0 ]]; then
        "$python" -c "import cellpose, tifffile, zarr" 2>/dev/null \
            || die "$python lacks Cellpose 4.1.1, tifffile or zarr; give --python, --container or --segmentation"
    fi
    [[ -n "$cellpose_model" ]] || cellpose_model=$HOME/.cellpose/models/cpsam
    if [[ -f "$cellpose_model" ]]; then
        cellpose_model=$(absolute "$cellpose_model")
    elif [[ $dry_run -eq 0 ]]; then
        die "no Cellpose-SAM model at $cellpose_model; give --cellpose-model, or let Cellpose download it with: python -c \"from cellpose import models; models.CellposeModel(pretrained_model='cpsam')\""
    fi
    if [[ -z "$pixel_size" ]]; then
        if [[ $dry_run -eq 1 ]]; then
            pixel_size=PIXEL_SIZE_FROM_OME_METADATA
        else
            pixel_size=$(seg_python -c '
import re, sys, tifffile
with tifffile.TiffFile(sys.argv[1]) as tif:
    found = re.search(r"PhysicalSizeX=\"([0-9.eE+-]+)\"", tif.pages[0].description or "")
    unit = re.search(r"PhysicalSizeXUnit=\"([^\"]+)\"", tif.pages[0].description or "")
if not found or (unit and unit.group(1) not in ("um", "µm", "μm")):
    sys.exit(1)
print(found.group(1))
' "$he") || die "$he gives no pixel size in micrometres; give --pixel-size-um"
            echo "pixel size of the H&E image: $pixel_size µm (OME-TIFF metadata)"
        fi
    fi
    model=(--model cpsam --model-file "$cellpose_model" --model-sha256 "$cpsam_sha256")
    device=()
    [[ $cpu -eq 1 ]] && device=(--cpu)
    seg_command=("$repo_root/scripts/publication/segment_hd_he_cellpose.py"
        --image "$he" --out-dir "$segmentation" --source-pixel-size-um "$pixel_size"
        --profile independent_v1 --downsample 2 "${model[@]}" --tile-size 2048 --overlap 128
        --nucleus-expansion-um 2 --flow-threshold 0.4 --cellprob-threshold 0 --min-size 15
        --max-size-fraction 0.4 "${device[@]}")
    if [[ -n "$container" ]]; then
        echo "# in container $container:"
    fi
    printf '%q ' python "${seg_command[@]}"
    echo
    [[ $dry_run -eq 1 ]] || seg_python "${seg_command[@]}"
fi

# Second stage: nuclei onto the grid, then the shift from the counts.
if pending "$projection/COMPLETE.json" "$projection"; then
    run "${tool[visium_hd_count_register]}" project --segmentation "$segmentation" \
        --registration "$registration/registration.json" --capture "$capture/capture_grid.json" \
        --out "$projection" --threads "$threads"
fi
pending "$registered/COMPLETE.json" "$registered"
run "${tool[visium_hd_count_register]}" correct --segmentation "$segmentation" \
    --projection "$projection" --mex "$counts" --out "$registered" --threads "$threads"

if [[ $dry_run -eq 0 ]]; then
    python3 - "$registered" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
whole = json.loads((root / "check_before.json").read_text())["whole"]
fit = json.loads((root / "fit.json").read_text())["fit"]
correction = json.loads((root / "registered_geometry.json").read_text())["correction"]
if whole.get("usable"):
    print(f"whole slide: the counts place the nuclei {whole['dy']} rows and {whole['dx']} columns "
          f"({whole['dy_um']} and {whole['dx_um']} micrometres) from the first stage")
else:
    print(f"whole slide: no stable shift ({whole.get('reason')})")
print(f"regional fit: {fit.get('held_out_pass', 0)} of {fit.get('held_out_total', 0)} held-out regions "
      f"within one capture location; {'accepted' if fit.get('accepted') else 'not accepted'}")
print(f"correction applied: {correction.get('kind')}")
EOF
    echo "positions: $registered/positions_{002,008,016}um.parquet"
fi
