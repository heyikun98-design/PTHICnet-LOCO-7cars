# Error Analysis Report

Checkpoint: `experiments/pt_hicnet_es_seed42_film-global/checkpoints/best_acc_model.pth` (epoch 21)
Samples: 172

## Overall
- Mean Accuracy: 83.88%
- Mean MSE: 83837
- Neg Rate: 0.0000

## By HIC Bucket
| Bucket | Count | Mean Acc | Mean MSE | Mean HIC |
|---|---:|---:|---:|---:|
| <500 | 0 | — | — | — |
| 500-1k | 110 | 85.61% | 27295 | 764 |
| 1k-1.5k | 40 | 85.32% | 53730 | 1215 |
| 1.5k-2k | 18 | 74.45% | 227533 | 1670 |
| >2k | 4 | 64.37% | 1293168 | 2531 |

## By Age Group
| Age | Count | Mean Acc | Mean MSE |
|---|---:|---:|---:|
| Child | 104 | 85.93% | 71254 |
| Adult | 68 | 80.75% | 103080 |

## By HIC × Age
| Bucket | Age | Count | Mean Acc | Mean MSE |
|---|---:|---:|---:|
| 500-1k | Child | 65 | 89.32% | 23873 |
| 500-1k | Adult | 45 | 80.25% | 32238 |
| 1k-1.5k | Child | 23 | 86.81% | 40294 |
| 1k-1.5k | Adult | 17 | 83.29% | 71908 |
| 1.5k-2k | Child | 14 | 72.57% | 254077 |
| 1.5k-2k | Adult | 4 | 81.03% | 134630 |
| >2k | Child | 2 | 58.79% | 687433 |
| >2k | Adult | 2 | 69.96% | 1898904 |

## Worst 20 Samples (lowest accuracy_ratio)
| Rank | Accuracy | HIC True | HIC Pred | Age |
|---|---:|---:|---:|---:|
| 1 | 49.08% | 659 | 324 | Adult |
| 2 | 50.41% | 958 | 1901 | Child |
| 3 | 50.76% | 589 | 299 | Adult |
| 4 | 53.52% | 988 | 529 | Adult |
| 5 | 56.32% | 1712 | 964 | Child |
| 6 | 57.23% | 2011 | 1151 | Child |
| 7 | 59.51% | 825 | 491 | Adult |
| 8 | 60.12% | 1712 | 1029 | Child |
| 9 | 60.35% | 2011 | 1213 | Child |
| 10 | 60.98% | 1696 | 1034 | Child |
| 11 | 62.21% | 1123 | 698 | Adult |
| 12 | 62.45% | 574 | 358 | Adult |
| 13 | 62.47% | 1696 | 1060 | Child |
| 14 | 64.67% | 3062 | 4734 | Adult |
| 15 | 66.26% | 1736 | 1150 | Adult |
| 16 | 67.48% | 1742 | 1176 | Child |
| 17 | 68.18% | 1085 | 1591 | Adult |
| 18 | 68.25% | 1556 | 1062 | Child |
| 19 | 68.31% | 695 | 1018 | Adult |
| 20 | 68.38% | 1377 | 942 | Child |

## Best 20 Samples (highest accuracy_ratio)
| Rank | Accuracy | HIC True | HIC Pred | Age |
|---|---:|---:|---:|---:|
| 1 | 99.97% | 1051 | 1051 | Adult |
| 2 | 99.94% | 977 | 977 | Adult |
| 3 | 99.94% | 698 | 699 | Child |
| 4 | 99.89% | 628 | 629 | Child |
| 5 | 99.72% | 1030 | 1033 | Child |
| 6 | 99.70% | 908 | 911 | Child |
| 7 | 99.29% | 760 | 755 | Child |
| 8 | 99.28% | 819 | 813 | Child |
| 9 | 99.18% | 1412 | 1400 | Child |
| 10 | 99.00% | 948 | 938 | Child |
| 11 | 98.70% | 583 | 575 | Child |
| 12 | 98.18% | 626 | 615 | Adult |
| 13 | 97.95% | 760 | 776 | Child |
| 14 | 97.90% | 1541 | 1508 | Child |
| 15 | 97.68% | 778 | 797 | Adult |
| 16 | 97.56% | 1371 | 1405 | Adult |
| 17 | 97.54% | 733 | 751 | Adult |
| 18 | 97.42% | 698 | 717 | Child |
| 19 | 97.37% | 719 | 738 | Child |
| 20 | 96.97% | 1028 | 1060 | Adult |
