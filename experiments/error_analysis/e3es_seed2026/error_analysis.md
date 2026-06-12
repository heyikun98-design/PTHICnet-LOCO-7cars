# Error Analysis Report

Checkpoint: `experiments/pt_hicnet_es_seed2026_film-global/checkpoints/best_acc_model.pth` (epoch 19)
Samples: 172

## Overall
- Mean Accuracy: 83.13%
- Mean MSE: 93280
- Neg Rate: 0.0000

## By HIC Bucket
| Bucket | Count | Mean Acc | Mean MSE | Mean HIC |
|---|---:|---:|---:|---:|
| <500 | 0 | — | — | — |
| 500-1k | 110 | 82.26% | 64950 | 764 |
| 1k-1.5k | 40 | 85.19% | 99500 | 1215 |
| 1.5k-2k | 18 | 86.23% | 74637 | 1670 |
| >2k | 4 | 72.60% | 894057 | 2531 |

## By Age Group
| Age | Count | Mean Acc | Mean MSE |
|---|---:|---:|---:|
| Child | 104 | 85.12% | 46364 |
| Adult | 68 | 80.08% | 165034 |

## By HIC × Age
| Bucket | Age | Count | Mean Acc | Mean MSE |
|---|---:|---:|---:|
| 500-1k | Child | 65 | 84.35% | 33903 |
| 500-1k | Adult | 45 | 79.24% | 109797 |
| 1k-1.5k | Child | 23 | 88.83% | 30156 |
| 1k-1.5k | Adult | 17 | 80.27% | 193317 |
| 1.5k-2k | Child | 14 | 84.59% | 89860 |
| 1.5k-2k | Adult | 4 | 91.97% | 21357 |
| >2k | Child | 2 | 71.56% | 333291 |
| >2k | Adult | 2 | 73.64% | 1454823 |

## Worst 20 Samples (lowest accuracy_ratio)
| Rank | Accuracy | HIC True | HIC Pred | Age |
|---|---:|---:|---:|---:|
| 1 | 41.12% | 674 | 1640 | Adult |
| 2 | 43.10% | 516 | 1197 | Adult |
| 3 | 43.81% | 1067 | 2435 | Adult |
| 4 | 43.99% | 545 | 1238 | Adult |
| 5 | 48.88% | 692 | 1416 | Adult |
| 6 | 51.05% | 733 | 1435 | Adult |
| 7 | 52.37% | 778 | 1486 | Adult |
| 8 | 53.75% | 762 | 1417 | Adult |
| 9 | 55.17% | 958 | 1737 | Child |
| 10 | 59.12% | 819 | 484 | Child |
| 11 | 59.68% | 961 | 574 | Child |
| 12 | 61.29% | 802 | 491 | Child |
| 13 | 62.40% | 802 | 500 | Child |
| 14 | 63.34% | 1085 | 1713 | Adult |
| 15 | 63.80% | 957 | 611 | Child |
| 16 | 63.92% | 934 | 1461 | Adult |
| 17 | 65.49% | 3062 | 4675 | Adult |
| 18 | 66.00% | 529 | 801 | Adult |
| 19 | 67.58% | 2011 | 1359 | Child |
| 20 | 68.31% | 883 | 1293 | Adult |

## Best 20 Samples (highest accuracy_ratio)
| Rank | Accuracy | HIC True | HIC Pred | Age |
|---|---:|---:|---:|---:|
| 1 | 99.89% | 903 | 902 | Adult |
| 2 | 99.83% | 848 | 846 | Child |
| 3 | 99.52% | 745 | 742 | Adult |
| 4 | 99.48% | 574 | 571 | Adult |
| 5 | 99.27% | 583 | 587 | Child |
| 6 | 99.18% | 576 | 581 | Child |
| 7 | 99.17% | 1742 | 1757 | Child |
| 8 | 98.56% | 577 | 569 | Child |
| 9 | 98.51% | 792 | 804 | Adult |
| 10 | 98.43% | 953 | 938 | Child |
| 11 | 98.26% | 576 | 566 | Child |
| 12 | 98.24% | 1050 | 1032 | Child |
| 13 | 97.56% | 611 | 596 | Adult |
| 14 | 97.45% | 611 | 627 | Child |
| 15 | 96.93% | 1330 | 1372 | Child |
| 16 | 96.05% | 1209 | 1259 | Child |
| 17 | 95.94% | 752 | 784 | Child |
| 18 | 95.89% | 692 | 663 | Adult |
| 19 | 95.60% | 956 | 914 | Adult |
| 20 | 95.59% | 1556 | 1488 | Child |
