# Datasets

Every dataset analysed is public. Raw sequencing data, vendor outputs and
whole-slide images are not redistributed here; download them from the sources
below.

## Visium HD Flex — human colorectal cancer

Primary full-slide probe-based analysis, and slide 1 of the independent
Cellpose morphology benchmark.

| Field | Value |
|---|---|
| Source | <https://www.10xgenomics.com/datasets/visium-hd-cytassist-6p5mm-human-colon-cancer> |
| Slide / area | `H1-GMHFWPH` / `D1` |
| Assay | Visium HD Spatial Gene Expression, 6.5 mm capture area, FFPE |
| Probe set | Visium Human Transcriptome Probe Set v2.0 |
| Depth | 212,554,625 read pairs (43 bp R1, 50 bp R2, 10 bp i7, 10 bp i5) |
| Space Ranger | 4.1.0 |
| License | CC BY 4.0 |

## Visium HD 3′ — human ovarian cancer

Cross-assay replication on untargeted poly(A) chemistry with genomic alignment.
The Space Ranger comparator is our own Space Ranger 4.1.0 run of the same
FASTQs with introns included, so that both pipelines count the same features.

| Field | Value |
|---|---|
| Source | <https://www.10xgenomics.com/datasets/visium-hd-three-prime-ovarian-cancer-discovery-fresh-frozen> |
| Resource | `Visium_HD_3prime_Human_Ovarian_Cancer_FF_Min_Depth` |
| Slide / area | `H1-YQJBZ7X` / `D1` |
| Assay | Visium HD 3′ poly(A), fresh frozen |
| Depth | 474,131,092 read pairs (43 bp R1, 75 bp R2) |
| Space Ranger (published outputs) | 4.0.1 |
| Adjacent section | Xenium Prime 5K, below |

## Xenium Prime 5K — human ovarian cancer, adjacent section

Adjacent-section reference for the ovarian 3′ slide. The Visium HD dataset page
states that the Xenium v1, Xenium Prime 5K and Visium HD datasets "were
generated from adjacent tissue sections".

| Field | Value |
|---|---|
| Source | <https://www.10xgenomics.com/datasets/xenium-comparison-fresh-frozen-human-ovarian-cancer> ("Cross-Platform Comparison: FF Human Ovarian Cancer with Xenium v1 and Xenium Prime 5K") |
| Resource | `Xenium_Prime_Human_Ovary_Cancer_FF` (Xenium Onboard Analysis 4.0.0; 200,900 cells) |
| Files, sizes, MD5 | `paper_results/xenium_ovarian_adjacent_20251007/source_manifest.tsv` |
| License | CC BY 4.0 |

## Visium HD — human colorectal cancer (decoder taxonomy slide)

Used only for the audit of Space Ranger's recorded coordinates: Space Ranger
4.0.1 was run from the published FASTQs with a BAM, and each read's raw (`CR`)
and corrected (`CB`) barcodes were compared with the closest squares.
This is a **different slide** from `H1-GMHFWPH/D1`; coordinates must not be
transferred between them.

| Field | Value |
|---|---|
| Source | <https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc-v4> |
| Slide / area | `H1-VM2JXXK` / `A1` |
| Independent annotations | <https://doi.org/10.5281/zenodo.11402686> (CC0 pathologist annotations and nuclei) |

## SPATCH — human ovarian Visium HD FFPE with adjacent CODEX

Slide 2 of the independent Cellpose morphology benchmark. This is an
**independent third-party cohort**, not a vendor demonstration dataset, which is
why it carries most of the weight in the replication argument.

| Field | Value |
|---|---|
| Project | SPATCH — <https://github.com/zenglab-pku/SPATCH> |
| Assay | Visium HD Spatial Gene Expression, FFPE, with adjacent CODEX |
| Comparator used | published Space Ranger **filtered** `h5ad`; 629,942 barcodes, 18,085 features |
| H&E image | `hd_OV_HE.tif`, SHA256 `8411c6af9b23704244a655e5bc3ce2a4d90efaa30923c7e3bf5473d81b896b71` |
| CODEX | adjacent-section 16-plex CODEX from the deposition, compared on 100–500 µm grids |

> **Comparator surfaces differ.** The colorectal comparator is a *raw* Space
> Ranger matrix; the SPATCH comparator is a *published filtered* matrix. These
> are different eligibility surfaces, so the two slides replicate in
> **direction** but their magnitudes are not poolable. Do not average the two
> percentages.

Accessions for the SPATCH deposition span the project repository, the Genome
Sequence Archive and the BioImage Archive; consult the SPATCH project for the
current canonical identifiers and redistribution terms.

## Image masks

All cell and nucleus masks are Cellpose segmentations of the raw H&E images,
generated for this study with no vendor segmentation, expression or
registration product in their ancestry: the two probe slides with the primary
profile (flow threshold 0.4, nuclei expanded by 2 µm), and the ovarian 3′ slide
with the masks of its original intron-matched analysis (flow threshold 0.8,
nuclei expanded by 8 µm).
