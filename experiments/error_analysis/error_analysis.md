# Error Analysis Report

Checkpoint: `experiments/pt_hicnet_es_seed3407_film-global/checkpoints/best_acc_model.pth` (epoch 78)
Samples: 172

## Overall
- Mean Accuracy: 84.44%
- Mean MSE: 234814
- Neg Rate: 0.0000

## By HIC Bucket
| Bucket | Count | Mean Acc | Mean MSE | Mean HIC |
|---|---:|---:|---:|---:|
| <500 | 0 | — | — | — |
| 500-1k | 110 | 83.83% | 73476 | 764 |
| 1k-1.5k | 40 | 86.43% | 76982 | 1215 |
| 1.5k-2k | 18 | 89.20% | 50164 | 1670 |
| >2k | 4 | 60.14% | 7080882 | 2531 |

## By Age Group
| Age | Count | Mean Acc | Mean MSE |
|---|---:|---:|---:|
| Child | 104 | 86.46% | 40391 |
| Adult | 68 | 81.36% | 532169 |

## By HIC × Age
| Bucket | Age | Count | Mean Acc | Mean MSE |
|---|---:|---:|---:|
| 500-1k | Child | 65 | 85.88% | 31091 |
| 500-1k | Adult | 45 | 80.86% | 134698 |
| 1k-1.5k | Child | 23 | 87.71% | 36758 |
| 1k-1.5k | Adult | 17 | 84.68% | 131403 |
| 1.5k-2k | Child | 14 | 88.84% | 54568 |
| 1.5k-2k | Adult | 4 | 90.48% | 34753 |
| >2k | Child | 2 | 74.39% | 285161 |
| >2k | Adult | 2 | 45.90% | 13876603 |

## Worst 20 Samples (lowest accuracy_ratio)
| Rank | Accuracy | HIC True | HIC Pred | Age |
|---|---:|---:|---:|---:|
| 1 | 34.29% | 516 | 1505 | Adult |
| 2 | 35.08% | 545 | 1552 | Adult |
| 3 | 41.38% | 3062 | 7399 | Adult |
| 4 | 46.72% | 762 | 1631 | Adult |
| 5 | 48.00% | 733 | 1526 | Adult |
| 6 | 49.81% | 674 | 1354 | Adult |
| 7 | 50.41% | 3041 | 6031 | Adult |
| 8 | 51.54% | 778 | 1510 | Adult |
| 9 | 53.79% | 958 | 1782 | Child |
| 10 | 54.03% | 1067 | 1975 | Adult |
| 11 | 56.68% | 883 | 1558 | Adult |
| 12 | 56.73% | 1085 | 1912 | Adult |
| 13 | 57.27% | 934 | 1630 | Adult |
| 14 | 60.41% | 692 | 1146 | Adult |
| 15 | 65.84% | 1123 | 1705 | Adult |
| 16 | 67.35% | 2011 | 1354 | Child |
| 17 | 68.28% | 961 | 656 | Child |
| 18 | 68.64% | 623 | 907 | Adult |
| 19 | 69.68% | 811 | 1164 | Child |
| 20 | 70.05% | 529 | 755 | Adult |

## Best 20 Samples (highest accuracy_ratio)
| Rank | Accuracy | HIC True | HIC Pred | Age |
|---|---:|---:|---:|---:|
| 1 | 99.77% | 927 | 925 | Adult |
| 2 | 99.59% | 698 | 701 | Child |
| 3 | 99.54% | 737 | 741 | Adult |
| 4 | 99.41% | 739 | 735 | Child |
| 5 | 99.41% | 731 | 735 | Child |
| 6 | 99.31% | 1019 | 1012 | Adult |
| 7 | 99.28% | 1028 | 1021 | Adult |
| 8 | 99.23% | 609 | 604 | Adult |
| 9 | 99.14% | 848 | 840 | Child |
| 10 | 99.14% | 698 | 704 | Child |
| 11 | 98.87% | 792 | 801 | Adult |
| 12 | 98.81% | 618 | 626 | Adult |
| 13 | 98.78% | 626 | 634 | Adult |
| 14 | 98.70% | 1596 | 1575 | Child |
| 15 | 98.70% | 1541 | 1561 | Child |
| 16 | 98.65% | 763 | 753 | Child |
| 17 | 98.57% | 611 | 602 | Adult |
| 18 | 98.56% | 561 | 569 | Adult |
| 19 | 98.44% | 731 | 742 | Child |
| 20 | 97.84% | 1712 | 1675 | Child |
