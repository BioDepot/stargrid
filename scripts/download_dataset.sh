#!/usr/bin/env bash
# Download a public Visium HD dataset of the STARgrid manuscript, with the
# reference it needs, and check every file against the checksums we used.
#
#   ovarian      Visium HD 3' human ovarian cancer, fresh frozen (H1-YQJBZ7X/D1),
#                RNA-seq quantitation against GRCh38 2024-A
#   colorectal   Visium HD human colorectal cancer, FFPE, 6.5 mm (H1-GMHFWPH/D1),
#                Flex quantitation with the probe set published with it
#
# Both come from 10x Genomics' dataset pages under CC BY 4.0 (docs/DATASETS.md).
# The slide layout file (<serial>.vlf) is not downloaded here; see the message
# at the end.
set -euo pipefail

usage() {
    cat <<'EOF'
usage: download_dataset.sh ovarian|colorectal OUT_DIR [--keep-archives]

Downloads the FASTQ files (extracted from their archive), the H&E and CytAssist
images and, for colorectal, the probe set; then the 10x Genomics human
reference GRCh38 2024-A (genome FASTA and GTF). Files already present and
correct are not downloaded again. Needs curl, md5sum and tar.

Space while unpacking: about 70 GB for ovarian (36 GB afterwards), 42 GB for
colorectal (25 GB afterwards) and 15 GB for the reference (3 GB afterwards).
--keep-archives keeps the FASTQ and reference archives, which are otherwise
deleted once unpacked.
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

[[ $# -ge 2 ]] || { usage >&2; exit 2; }
dataset=$1
out_dir=$2
shift 2
keep=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep-archives) keep=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown option: $1" ;;
    esac
done
for command in curl md5sum tar; do
    command -v "$command" >/dev/null 2>&1 || die "$command is required"
done

# name, md5 (as recorded when we downloaded them for the manuscript)
case "$dataset" in
    ovarian)
        base=https://cf.10xgenomics.com/samples/spatial-exp/4.0.1/Visium_HD_3prime_Human_Ovarian_Cancer_FF_Min_Depth
        prefix=Visium_HD_3prime_Human_Ovarian_Cancer_FF_Min_Depth
        serial=H1-YQJBZ7X
        area=D1
        layout_sha256=1b48b8365a42cae37072250528cd372e0e516bfe5294263dd8ad07dea7dd4f20
        files=(
            "${prefix}_fastqs.tar 12fa0cff0565bf6b62ba835d6b18ef6b"
            "${prefix}_tissue_image.btf eb0838ec91d559e0b2110435c35e9c43"
            "${prefix}_image.tif a9d4d73ee7fd20ef7571b5c65c684b9f"
        )
        ;;
    colorectal)
        base=https://cf.10xgenomics.com/samples/spatial-exp/4.1.0/Visium_HD_6p5mm_Human_Colon_Cancer
        prefix=Visium_HD_6p5mm_Human_Colon_Cancer
        serial=H1-GMHFWPH
        area=D1
        layout_sha256=ce2ec21b24a19a5eaac11d842a51528476af018e05e7edac367e809016482a63
        files=(
            "${prefix}_fastqs.tar baf3600bb3d88cfcd9b16c546a17e1d9"
            "${prefix}_tissue_image.btf b3abeb5464f070262f691c5595440b1b"
            "${prefix}_image.tif 4fb17fd7024b2b104d96115b2ad7baa9"
            "${prefix}_probe_set.csv a74f1583689de3c9f95164b2fc090194"
        )
        ;;
    *) usage >&2; die "the dataset must be ovarian or colorectal" ;;
esac
reference_url=https://cf.10xgenomics.com/supp/cell-exp/refdata-gex-GRCh38-2024-A.tar.gz
reference=refdata-gex-GRCh38-2024-A
reference_fasta_md5=46c8022ace600b277a05a94b0cb2f4fa
reference_gtf_md5=7eea40d4fb344eed2c255e2935aa2f95

mkdir -p "$out_dir"
cd "$out_dir"

fetch() {
    local name=$1 url=$2 md5=$3
    if [[ -f "$name" ]] && echo "$md5  $name" | md5sum -c --quiet - 2>/dev/null; then
        echo "present: $name"
        return
    fi
    echo "downloading $name"
    curl -fL --retry 8 --retry-delay 5 --retry-all-errors --continue-at - -o "$name" "$url"
    echo "$md5  $name" | md5sum -c --quiet - || die "$name does not match its checksum; delete it and run again"
}

for entry in "${files[@]}"; do
    read -r name md5 <<< "$entry"
    fastq_dir=${prefix}_fastqs
    if [[ "$name" == *_fastqs.tar && -f "$fastq_dir/.complete" ]]; then
        echo "present: $fastq_dir/"
        continue
    fi
    fetch "$name" "$base/$name" "$md5"
    if [[ "$name" == *_fastqs.tar ]]; then
        echo "unpacking $name"
        tar -xf "$name"
        touch "$fastq_dir/.complete"
        [[ $keep -eq 1 ]] || rm -f "$name"
    fi
done

if [[ -f "$reference/.complete" ]]; then
    echo "present: $reference/"
else
    if [[ ! -f "$reference.tar.gz" ]]; then
        echo "downloading $reference.tar.gz"
        curl -fL --retry 8 --retry-delay 5 --retry-all-errors --continue-at - -o "$reference.tar.gz" "$reference_url"
    fi
    echo "unpacking $reference.tar.gz"
    tar -xzf "$reference.tar.gz" "$reference/fasta/genome.fa" "$reference/genes/genes.gtf.gz"
    echo "$reference_fasta_md5  $reference/fasta/genome.fa" | md5sum -c --quiet - \
        || die "$reference/fasta/genome.fa differs from the manuscript's reference"
    echo "$reference_gtf_md5  $reference/genes/genes.gtf.gz" | md5sum -c --quiet - \
        || die "$reference/genes/genes.gtf.gz differs from the manuscript's reference"
    touch "$reference/.complete"
    [[ $keep -eq 1 ]] || rm -f "$reference.tar.gz"
fi

cat <<EOF

Downloaded to $(pwd):
  FASTQ files       ${prefix}_fastqs/
  H&E image         ${prefix}_tissue_image.btf
  CytAssist image   ${prefix}_image.tif
  reference         $reference/ (genome FASTA and GTF)
EOF
[[ "$dataset" == colorectal ]] && echo "  probe set         ${prefix}_probe_set.csv"
cat <<EOF

Still needed for the registration: the layout file of slide $serial (capture
area $area). Download it from 10x Genomics' slide layout file page by entering
the serial number $serial, and save it here as $serial.vlf. The file we used
has SHA-256 $layout_sha256.

Next steps: README.md, section "Reproduce a slide".
EOF
