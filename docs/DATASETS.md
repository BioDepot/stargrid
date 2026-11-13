# Datasets

Every dataset used in the STARgrid manuscript is public. Reads, vendor outputs and images are not
redistributed here; get them from the sources below. Supplemental Table S3 of the manuscript lists
the same datasets.

`scripts/download_dataset.sh ovarian|colorectal DIR` downloads the first and third slides below,
with the reference they need, and checks each file against the checksums of the copies we used.
The README, section "Reproduce a slide", describes the rest.

## Visium HD slides compared with Space Ranger

| Dataset | Slide | Assay; reference | Source | Used for |
|---|---|---|---|---|
| Colorectal cancer, FFPE | H1-GMHFWPH/D1 | Visium HD, Flex; the probe set published with the dataset (Visium Human Transcriptome Probe Set v2.0, GRCh38 2020-A) | 10x Genomics: [Visium HD Flex human colorectal cancer](https://www.10xgenomics.com/datasets/visium-hd-cytassist-6p5mm-human-colon-cancer) (CC BY 4.0) | comparison, timing, H&E tests, registration |
| SPATCH ovarian cancer, FFPE (Ren et al. 2025) | H1-WP2FTZD/D1 | Visium HD, Flex; the same probe set as the slide above | SPATCH resource: reads in the Genome Sequence Archive for Human ([HRA011129](https://ngdc.cncb.ac.cn/gsa-human/browse/HRA011129)), images in the BioImage Archive ([S-BIAD1900](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1900)), filtered count matrix from <https://spatch.pku-genomics.org> | comparison, timing, H&E tests, registration |
| Ovarian cancer, fresh frozen | H1-YQJBZ7X/D1 | Visium HD 3′, RNA-seq; GRCh38 2024-A (10x Genomics reference `refdata-gex-GRCh38-2024-A`) | 10x Genomics: [Visium HD 3′ human ovarian cancer, fresh frozen](https://www.10xgenomics.com/datasets/visium-hd-three-prime-ovarian-cancer-discovery-fresh-frozen), minimum-depth library (CC BY 4.0) | comparison, timing, H&E and Xenium tests, read-by-read comparison, registration |
| Colorectal cancer, FFPE | H1-VM2JXXK/A1 | Visium HD, Flex; Visium Human Transcriptome Probe Set v2.1.0 (GRCh38 2024-A) | 10x Genomics: [Visium HD human colorectal cancer](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc-v4); Space Ranger output also in GEO ([GSE280315](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280315), sample GSM8594568) | comparison, read-by-read comparison, registration |

The two colorectal slides are different tissue sections; coordinates must not be carried from one
to the other.

## Xenium

| Dataset | Slide | Assay | Source | Used for |
|---|---|---|---|---|
| Ovarian cancer, fresh frozen, sections adjacent to the Visium HD slide | — | Xenium Prime 5K | 10x Genomics: [Cross-Platform Comparison: FF Human Ovarian Cancer with Xenium v1 and Xenium Prime 5K](https://www.10xgenomics.com/datasets/xenium-comparison-fresh-frozen-human-ovarian-cancer), resource `Xenium_Prime_Human_Ovary_Cancer_FF` (CC BY 4.0); files, sizes and MD5 in `paper_results/xenium_ovarian_adjacent_20251007/source_manifest.tsv` | Xenium tests of the added counts |
| Lung tissue cores (Vannan et al. 2025) | H1-K3WKZHN/D1 | Xenium, then Visium HD on the same section | GEO: Visium HD with Space Ranger's counts and registration in [GSE276934](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE276934) (sample GSM8509590); Xenium in [GSE250346](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250346) | registration test |
| Lung adenocarcinoma, whole section, experiment 1 | H1-9QZXXNP/D1 | Xenium v1, then Visium HD on the same section | 10x Genomics, with Space Ranger 3.0.0's counts and registration: [Visium HD post-Xenium human lung cancer](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-human-lung-cancer-post-xenium-expt) and [Xenium human lung cancer](https://www.10xgenomics.com/datasets/xenium-human-lung-cancer-post-xenium-technote) (CC BY 4.0) | registration test |
| Lung adenocarcinoma, whole section, experiment 2 | H1-JGHRVB7/D1 | Xenium Prime 5K, then Visium HD on the same section | as experiment 1 | registration test |

## Other tissues

Published by 10x Genomics with Space Ranger 4.0.1's output and nucleus segmentation (CC BY 4.0);
used to test the registration in other tissues.

| Dataset | Assay | Source |
|---|---|---|
| Human lymph node, FFPE | Visium HD, Flex | [human lymph node](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-human-lymph-node-v4) |
| Human prostate cancer, FFPE | Visium HD, Flex | [human prostate cancer](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-human-prostate-cancer-ffpe) |
| Human kidney, FFPE | Visium HD, Flex | [human kidney](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-human-kidney-ffpe-v4) |
| Mouse brain | Visium HD, Flex | [mouse brain](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-brain-he-v4) |

## Files from 10x Genomics that are not data

- The barcode layout of Visium HD slides is read from `visium_hd_v1.slide`, a reference file
  distributed with Space Ranger (`lib/python/cellranger/barcodes/`; the same file in Space Ranger
  3.0.0, 4.0.1 and 4.1.0). Obtain it with Space Ranger, under its licence.
  `scripts/make_visium_hd_contract.py` in STAR Suite turns it into STAR's barcode inputs.
- The registration reads the layout file of each slide (`<serial>.vlf`), which 10x Genomics
  provides for download by slide serial number.

## Nucleus masks

The nuclei are segmented from the H&E images with Cellpose-SAM (Cellpose 4.1.1, model `cpsam`) by
`scripts/publication/segment_hd_he_cellpose.py` with the profile `independent_v1`: the image
reduced twofold, hematoxylin optical density, flow threshold 0.4, cell-probability threshold 0,
cells drawn 2 µm around each nucleus. No vendor segmentation, expression or registration output
enters the masks. StarDist masks (`scripts/segment_hd_he_stardist.py`) were the control.
