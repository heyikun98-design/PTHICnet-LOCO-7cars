# Error Analysis Report

Checkpoint: `experiments/pt_hicnet_ahw_seed3407_film-global/checkpoints/best_acc_model.pth` (epoch 40)
Samples: 172

## Overall
- Mean Accuracy: 72.50%
- Mean MSE: 307804
- Neg Rate: 0.0000

## By HIC Bucket
| Bucket | Count | Mean Acc | Mean MSE | Mean HIC |
|---|---:|---:|---:|---:|
| <500 | 0 | — | — | — |
| 500-1k | 110 | 74.22% | 113625 | 764 |
| 1k-1.5k | 40 | 73.99% | 174874 | 1215 |
| 1.5k-2k | 18 | 62.75% | 1001427 | 1670 |
| >2k | 4 | 54.30% | 3855705 | 2531 |

## By Age Group
| Age | Count | Mean Acc | Mean MSE |
|---|---:|---:|---:|
| Child | 104 | 76.73% | 135042 |
| Adult | 68 | 66.03% | 572027 |

## By HIC × Age
| Bucket | Age | Count | Mean Acc | Mean MSE |
|---|---:|---:|---:|
| 500-1k | Child | 65 | 82.67% | 30590 |
| 500-1k | Adult | 45 | 62.01% | 233565 |
| 1k-1.5k | Child | 23 | 70.68% | 184717 |
| 1k-1.5k | Adult | 17 | 78.47% | 161558 |
| 1.5k-2k | Child | 14 | 62.29% | 436408 |
| 1.5k-2k | Adult | 4 | 64.37% | 2978994 |
| >2k | Child | 2 | 54.43% | 848918 |
| >2k | Adult | 2 | 54.18% | 6862492 |

## Worst 20 Samples (lowest accuracy_ratio)
| Rank | Accuracy | HIC True | HIC Pred | Age |
|---|---:|---:|---:|---:|
| 1 | 36.43% | 1613 | 4428 | Adult |
| 2 | 39.87% | 516 | 1294 | Adult |
| 3 | 42.05% | 545 | 1295 | Adult |
| 4 | 43.99% | 1543 | 3507 | Adult |
| 5 | 45.52% | 529 | 1162 | Adult |
| 6 | 46.12% | 540 | 1172 | Adult |
| 7 | 46.89% | 545 | 1161 | Adult |
| 8 | 46.96% | 554 | 1179 | Adult |
| 9 | 48.33% | 623 | 1288 | Adult |
| 10 | 49.19% | 1884 | 927 | Child |
| 11 | 49.60% | 2011 | 997 | Child |
| 12 | 50.27% | 1712 | 861 | Child |
| 13 | 50.44% | 733 | 1453 | Adult |
| 14 | 50.97% | 3041 | 5966 | Adult |
| 15 | 51.19% | 1451 | 743 | Child |
| 16 | 51.26% | 778 | 1518 | Adult |
| 17 | 51.85% | 674 | 1301 | Adult |
| 18 | 52.46% | 626 | 1193 | Adult |
| 19 | 52.65% | 692 | 1314 | Adult |
| 20 | 53.53% | 565 | 1056 | Adult |

## Best 20 Samples (highest accuracy_ratio)
| Rank | Accuracy | HIC True | HIC Pred | Age |
|---|---:|---:|---:|---:|
| 1 | 99.86% | 948 | 946 | Child |
| 2 | 98.89% | 811 | 820 | Child |
| 3 | 98.57% | 760 | 771 | Child |
| 4 | 98.33% | 611 | 601 | Child |
| 5 | 97.89% | 1718 | 1755 | Adult |
| 6 | 97.86% | 731 | 747 | Child |
| 7 | 97.02% | 752 | 776 | Child |
| 8 | 96.13% | 748 | 719 | Child |
| 9 | 96.12% | 719 | 691 | Child |
| 10 | 95.89% | 1193 | 1245 | Adult |
| 11 | 95.71% | 862 | 901 | Child |
| 12 | 95.41% | 762 | 799 | Child |
| 13 | 95.27% | 1050 | 1000 | Child |
| 14 | 95.04% | 583 | 554 | Child |
| 15 | 94.90% | 763 | 724 | Child |
| 16 | 94.20% | 848 | 799 | Child |
| 17 | 93.95% | 742 | 790 | Child |
| 18 | 93.53% | 1209 | 1131 | Child |
| 19 | 93.31% | 819 | 764 | Child |
| 20 | 93.29% | 1371 | 1469 | Adult |
