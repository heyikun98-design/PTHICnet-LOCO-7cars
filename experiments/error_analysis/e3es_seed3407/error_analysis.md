# Error Analysis Report

Checkpoint: `experiments/pt_hicnet_es_seed3407_film-global/checkpoints/best_acc_model.pth` (epoch 78)
Samples: 172

## Overall
- Mean Accuracy: 84.02%
- Mean MSE: 322441
- Neg Rate: 0.0000

## By HIC Bucket
| Bucket | Count | Mean Acc | Mean MSE | Mean HIC |
|---|---:|---:|---:|---:|
| <500 | 0 | — | — | — |
| 500-1k | 110 | 84.36% | 60707 | 764 |
| 1k-1.5k | 40 | 86.20% | 83642 | 1215 |
| 1.5k-2k | 18 | 82.91% | 123463 | 1670 |
| >2k | 4 | 57.82% | 10803519 | 2531 |

## By Age Group
| Age | Count | Mean Acc | Mean MSE |
|---|---:|---:|---:|
| Child | 104 | 86.68% | 40271 |
| Adult | 68 | 79.95% | 753995 |

## By HIC × Age
| Bucket | Age | Count | Mean Acc | Mean MSE |
|---|---:|---:|---:|
| 500-1k | Child | 65 | 86.87% | 22643 |
| 500-1k | Adult | 45 | 80.74% | 115688 |
| 1k-1.5k | Child | 23 | 88.04% | 39204 |
| 1k-1.5k | Adult | 17 | 83.71% | 143763 |
| 1.5k-2k | Child | 14 | 85.27% | 77911 |
| 1.5k-2k | Adult | 4 | 74.64% | 282898 |
| >2k | Child | 2 | 74.75% | 361952 |
| >2k | Adult | 2 | 40.89% | 21245084 |

## Worst 20 Samples (lowest accuracy_ratio)
| Rank | Accuracy | HIC True | HIC Pred | Age |
|---|---:|---:|---:|---:|
| 1 | 35.92% | 3041 | 8466 | Adult |
| 2 | 39.76% | 545 | 1370 | Adult |
| 3 | 43.90% | 516 | 1176 | Adult |
| 4 | 45.75% | 733 | 1602 | Adult |
| 5 | 45.86% | 3062 | 6675 | Adult |
| 6 | 47.15% | 762 | 1616 | Adult |
| 7 | 47.32% | 1085 | 2292 | Adult |
| 8 | 50.78% | 883 | 1739 | Adult |
| 9 | 53.13% | 674 | 1269 | Adult |
| 10 | 54.63% | 778 | 1425 | Adult |
| 11 | 56.53% | 692 | 1224 | Adult |
| 12 | 58.70% | 2011 | 1180 | Child |
| 13 | 61.49% | 958 | 1558 | Child |
| 14 | 62.33% | 623 | 999 | Adult |
| 15 | 62.45% | 1451 | 906 | Child |
| 16 | 62.62% | 1067 | 1704 | Adult |
| 17 | 65.09% | 1543 | 2370 | Adult |
| 18 | 67.22% | 529 | 787 | Adult |
| 19 | 68.72% | 1028 | 1495 | Adult |
| 20 | 68.80% | 961 | 661 | Child |

## Best 20 Samples (highest accuracy_ratio)
| Rank | Accuracy | HIC True | HIC Pred | Age |
|---|---:|---:|---:|---:|
| 1 | 99.55% | 953 | 948 | Child |
| 2 | 99.31% | 698 | 703 | Child |
| 3 | 99.21% | 1541 | 1529 | Child |
| 4 | 99.17% | 731 | 737 | Child |
| 5 | 99.11% | 698 | 692 | Child |
| 6 | 99.06% | 1412 | 1398 | Child |
| 7 | 98.91% | 731 | 739 | Child |
| 8 | 98.72% | 618 | 611 | Adult |
| 9 | 98.55% | 977 | 962 | Adult |
| 10 | 98.28% | 609 | 619 | Adult |
| 11 | 98.26% | 1188 | 1209 | Adult |
| 12 | 98.14% | 1014 | 1034 | Child |
| 13 | 98.12% | 561 | 572 | Adult |
| 14 | 98.07% | 760 | 746 | Child |
| 15 | 97.33% | 979 | 953 | Adult |
| 16 | 97.31% | 903 | 879 | Adult |
| 17 | 97.29% | 985 | 1013 | Child |
| 18 | 97.24% | 1274 | 1310 | Adult |
| 19 | 97.05% | 611 | 593 | Adult |
| 20 | 97.00% | 1596 | 1548 | Child |
