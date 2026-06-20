# Hard-Car Failure Analysis

## History Summary
| Arch | Vehicle | Val | Test | Gap | Best Epoch |
|---|---|---:|---:|---:|---:|
| E0 | CY02C | 77.89% | 64.23% | 13.7pp | 182 |
| E0 | M6 | 77.54% | 62.84% | 14.7pp | 192 |
| E2 | CY02C | 84.27% | 59.10% | 25.2pp | 107 |
| E2 | M6 | 84.85% | 61.33% | 23.5pp | 195 |
| E3 | CY02C | 84.92% | 58.60% | 26.3pp | 167 |
| E3 | M6 | 84.86% | 60.67% | 24.2pp | 194 |
| E4 | CY02C | 84.43% | 63.39% | 21.0pp | 195 |
| E4 | M6 | 86.58% | 61.81% | 24.8pp | 189 |

## Vehicle Union Scale Diagnostics
| Vehicle | Group | Samples | Union Diag | vs Normal Mean | Union X | Union Y | Union Z |
|---|---|---:|---:|---:|---:|---:|---:|
| C201 | normal | 222 | 2905 | 1.05x | 2001 | 1980 | 719 |
| CY02C | hard | 224 | 2767 | 1.00x | 1846 | 1798 | 1009 |
| EP32 | normal | 217 | 2806 | 1.01x | 1917 | 1878 | 819 |
| FX11 | normal | 221 | 2714 | 0.98x | 1844 | 1818 | 810 |
| JX65 | normal | 172 | 2584 | 0.93x | 1648 | 1890 | 623 |
| M6 | hard | 242 | 2793 | 1.01x | 1821 | 1913 | 910 |
| S50EVK | normal | 220 | 2866 | 1.03x | 1989 | 1914 | 771 |

## Local Sample Scale Diagnostics
| Vehicle | Group | Count | BBox Diag | vs Normal Mean | BBox X | BBox Y | BBox Z |
|---|---|---:|---:|---:|---:|---:|---:|
| C201 | normal | 222 | 898 | 1.01x | 585 | 578 | 355 |
| CY02C | hard | 224 | 871 | 0.98x | 557 | 547 | 368 |
| EP32 | normal | 217 | 898 | 1.01x | 578 | 568 | 382 |
| FX11 | normal | 221 | 880 | 0.99x | 568 | 553 | 372 |
| JX65 | normal | 172 | 876 | 0.99x | 575 | 577 | 310 |
| M6 | hard | 242 | 877 | 0.99x | 556 | 555 | 380 |
| S50EVK | normal | 220 | 892 | 1.00x | 581 | 564 | 367 |

## Largest Hard-vs-Normal Feature Shifts
Values below are normalized feature means unless a raw-unit row is shown later.

| Feature | Normal Mean (z) | Hard Mean (z) | Diff (z) | Cohen d |
|---|---:|---:|---:|---:|
| mat_02_mean | 0.213 | -0.02051 | -0.2335 | -1.09 |
| mat_08_mean | 0.05927 | 0.3153 | 0.2561 | 0.67 |
| mat_07_mean | 0.07398 | 0.2886 | 0.2146 | 0.63 |
| mat_06_mean | 0.07287 | 0.2858 | 0.2129 | 0.63 |
| mat_05_mean | 0.07522 | 0.2879 | 0.2127 | 0.62 |
| mat_04_mean | 0.08175 | 0.2978 | 0.2161 | 0.62 |
| mat_03_mean | 0.1126 | 0.3533 | 0.2407 | 0.60 |
| mat_11_mean | 0.1152 | 0.3073 | 0.1921 | 0.56 |
| mat_02_std | 0.3933 | 0.5149 | 0.1217 | 0.48 |
| mat_00_std | 0.754 | 0.6716 | -0.08234 | -0.44 |
| mat_01_std | 0.7803 | 0.7135 | -0.06671 | -0.37 |
| mat_12_mean | -0.136 | -0.1262 | 0.009727 | 0.30 |

## Material Feature Raw-Unit Check
Material channels are Z-Score normalized at runtime. Physical interpretation should use these raw-unit estimates, not the z-score means directly.

| Feature | Label | Normal Raw | Hard Raw | Diff Raw | Cohen d |
|---|---|---:|---:|---:|---:|
| mat_02_mean | poisson_ratio | 0.3153 | 0.2893 | -0.02601 | -1.09 |
| mat_08_mean | stress_curve_0.001_0.5 | 291.4 | 369.1 | 77.68 | 0.67 |
| mat_07_mean | stress_curve_0.001_0.2 | 268.7 | 327.4 | 58.68 | 0.63 |
| mat_06_mean | stress_curve_0.001_0.15 | 256.8 | 312.7 | 55.91 | 0.63 |
| mat_05_mean | stress_curve_0.001_0.1 | 242.1 | 295 | 52.87 | 0.62 |
| mat_04_mean | stress_curve_0.001_0.05 | 222.8 | 272.7 | 49.95 | 0.62 |
| mat_03_mean | stress_curve_0.001_0 | 183.6 | 231.9 | 48.28 | 0.60 |
| mat_11_mean | stress_curve_1_0.1 | 222.9 | 267.2 | 44.27 | 0.56 |
| mat_12_mean | stress_curve_1_0.15 | 5.979 | 7.047 | 1.067 | 0.30 |
| mat_09_mean | stress_curve_1_0 | 4.193 | 4.941 | 0.7484 | 0.30 |
| mat_10_mean | stress_curve_1_0.05 | 5.152 | 6.071 | 0.9195 | 0.30 |
| mat_14_mean | stress_curve_1_0.5 | 6.927 | 8.163 | 1.236 | 0.30 |

## Material Lookup Coverage
ID-level overlap is 6/28 hard IDs, while rounded vector-level overlap is 14/18 hard material vectors.

| Vehicle | IDs | Unique Vectors | ID Overlap vs Normal | ID Only vs Normal | Vector Overlap vs Normal | Vector Only vs Normal |
|---|---:|---:|---:|---:|---:|---:|
| C201 | 20 | 17 | 20 | 0 | 17 | 0 |
| EP32 | 16 | 16 | 16 | 0 | 16 | 0 |
| JX65 | 18 | 18 | 18 | 0 | 18 | 0 |
| CY02C | 19 | 15 | 4 | 15 | 11 | 4 |
| M6 | 9 | 8 | 2 | 7 | 8 | 0 |
| S50EVK | 12 | 11 | 12 | 0 | 11 | 0 |
| FX11 | 20 | 17 | 20 | 0 | 17 | 0 |

## Inference Summary
| Arch | Vehicle | Count | Mean Acc | Mean MSE | Neg Rate |
|---|---|---:|---:|---:|---:|
| E0 | CY02C | 224 | 64.28% | 936495 | 0.0000 |
| E0 | M6 | 242 | 62.58% | 84153512 | 0.0000 |
| E2 | CY02C | 224 | 58.68% | 2175283 | 0.0000 |
| E2 | M6 | 242 | 61.00% | 73873431 | 0.0372 |
| E3 | CY02C | 224 | 58.78% | 2698211 | 0.0000 |
| E3 | M6 | 242 | 61.23% | 70083390 | 0.0000 |
| E4 | CY02C | 224 | 63.69% | 1681776 | 0.0000 |
| E4 | M6 | 242 | 62.07% | 74921945 | 0.0083 |

## Worst Samples
| Rank | Arch | Vehicle | Acc | HIC True | HIC Pred | Age | Bucket | Sample |
|---:|---|---|---:|---:|---:|---|---|---|
| 1 | E0 | M6 | 1.35% | 140176 | 1887 | Adult | >2k | A_13__8 |
| 2 | E4 | M6 | 1.99% | 410 | -8 | Adult | <500 | A_11__2 |
| 3 | E2 | CY02C | 2.59% | 259 | 9999 | Adult | <500 | A_15__8 |
| 4 | E2 | CY02C | 2.65% | 259 | 9770 | Adult | <500 | A_15_8 |
| 5 | E3 | CY02C | 2.80% | 259 | 9244 | Adult | <500 | A_15_8 |
| 6 | E4 | M6 | 3.18% | 450 | 14 | Adult | <500 | A_11__1 |
| 7 | E4 | CY02C | 3.48% | 259 | 7443 | Adult | <500 | A_15_8 |
| 8 | E4 | CY02C | 3.63% | 259 | 7144 | Adult | <500 | A_15__8 |
| 9 | E3 | CY02C | 3.67% | 259 | 7057 | Adult | <500 | A_15__8 |
| 10 | E4 | M6 | 4.18% | 447 | 19 | Adult | <500 | A_11_4 |
| 11 | E2 | M6 | 4.38% | 8938 | 392 | Adult | >2k | A_11__7 |
| 12 | E4 | M6 | 5.42% | 140176 | 7599 | Adult | >2k | A_13__8 |
| 13 | E4 | M6 | 5.59% | 346 | -19 | Adult | <500 | A_11_3 |
| 14 | E4 | M6 | 6.57% | 8938 | 587 | Adult | >2k | A_11__7 |
| 15 | E2 | M6 | 6.69% | 140176 | 9374 | Adult | >2k | A_13__8 |
| 16 | E2 | M6 | 7.24% | 6301 | 456 | Adult | >2k | A_14_7 |
| 17 | E0 | CY02C | 7.25% | 259 | 3573 | Adult | <500 | A_15__8 |
| 18 | E2 | M6 | 7.25% | 5054 | 366 | Adult | >2k | A_12_7 |
| 19 | E0 | CY02C | 7.29% | 259 | 3554 | Adult | <500 | A_15_8 |
| 20 | E4 | CY02C | 7.70% | 805 | 10454 | Adult | 500-1k | A_14__8 |

## E0 vs E3 Per-Sample Delta
| Rank | Vehicle | E0 Acc | E3 Acc | E3-E0 | HIC True | Age | Bucket | Sample |
|---:|---|---:|---:|---:|---:|---|---|---|
| 1 | CY02C | 88.01% | 24.85% | -63.2pp | 1229 | Adult | 1k-1.5k | A_10_8 |
| 2 | CY02C | 95.11% | 32.96% | -62.2pp | 2167 | Adult | >2k | A_15_7 |
| 3 | M6 | 96.12% | 36.21% | -59.9pp | 384 | Adult | <500 | A_13__5 |
| 4 | M6 | 93.35% | 36.16% | -57.2pp | 355 | Adult | <500 | A_13__2 |
| 5 | M6 | 67.49% | 10.33% | -57.2pp | 1461 | Adult | 1k-1.5k | A_10__9 |
| 6 | M6 | 96.69% | 39.60% | -57.1pp | 328 | Adult | <500 | A_13__1 |
| 7 | M6 | 87.92% | 31.02% | -56.9pp | 809 | Adult | 500-1k | A_13__6 |
| 8 | CY02C | 82.77% | 26.41% | -56.4pp | 399 | Adult | <500 | A_13_0 |
| 9 | CY02C | 72.08% | 16.13% | -56.0pp | 443 | Adult | <500 | A_13__2 |
| 10 | M6 | 91.34% | 37.40% | -53.9pp | 305 | Adult | <500 | A_13__4 |
| 11 | CY02C | 97.90% | 44.77% | -53.1pp | 1278 | Adult | 1k-1.5k | A_9__8 |
| 12 | CY02C | 73.22% | 21.33% | -51.9pp | 434 | Adult | <500 | A_13_2 |
| 13 | M6 | 89.16% | 38.78% | -50.4pp | 875 | Adult | 500-1k | A_11_6 |
| 14 | M6 | 99.98% | 50.08% | -49.9pp | 1965 | Adult | 1.5k-2k | A_10_6 |
| 15 | CY02C | 76.07% | 27.45% | -48.6pp | 438 | Adult | <500 | A_13_1 |
| 16 | M6 | 73.25% | 25.23% | -48.0pp | 500 | Adult | 500-1k | A_14__5 |
| 17 | M6 | 88.22% | 42.12% | -46.1pp | 317 | Adult | <500 | A_13_0 |
| 18 | M6 | 87.62% | 41.56% | -46.1pp | 2760 | Adult | >2k | A_10_8 |
| 19 | M6 | 97.92% | 52.82% | -45.1pp | 320 | Adult | <500 | A_13_3 |
| 20 | M6 | 65.63% | 20.56% | -45.1pp | 1566 | Adult | 1.5k-2k | A_10_9 |

## Hard-Only Material Exposure vs E0-E3 Delta
| Vehicle | Count | Exposed | Mean Delta | Exposed Delta | Unexposed Delta | r(frac, delta) | r(top32, delta) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CY02C | 223 | 143 (64.1%) | -5.71pp | -5.82pp | -5.51pp | -0.314 | -0.245 |
| M6 | 242 | 0 (0.0%) | -1.35pp | nanpp | -1.35pp | nan | nan |
