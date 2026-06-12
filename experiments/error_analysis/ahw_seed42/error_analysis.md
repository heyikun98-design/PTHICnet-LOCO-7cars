# Error Analysis Report

Checkpoint: `experiments/pt_hicnet_ahw_seed42_film-global/checkpoints/best_acc_model.pth` (epoch 13)
Samples: 172

## Overall
- Mean Accuracy: 73.34%
- Mean MSE: 330404
- Neg Rate: 0.0000

## By HIC Bucket
| Bucket | Count | Mean Acc | Mean MSE | Mean HIC |
|---|---:|---:|---:|---:|
| <500 | 0 | — | — | — |
| 500-1k | 110 | 74.78% | 238440 | 764 |
| 1k-1.5k | 40 | 71.51% | 492750 | 1215 |
| 1.5k-2k | 18 | 68.07% | 516200 | 1670 |
| >2k | 4 | 75.90% | 399889 | 2531 |

## By Age Group
| Age | Count | Mean Acc | Mean MSE |
|---|---:|---:|---:|
| Child | 104 | 82.29% | 91873 |
| Adult | 68 | 59.66% | 695217 |

## By HIC × Age
| Bucket | Age | Count | Mean Acc | Mean MSE |
|---|---:|---:|---:|
| 500-1k | Child | 65 | 86.87% | 27875 |
| 500-1k | Adult | 45 | 57.31% | 542590 |
| 1k-1.5k | Child | 23 | 79.16% | 107649 |
| 1k-1.5k | Adult | 17 | 61.17% | 1013769 |
| 1.5k-2k | Child | 14 | 69.34% | 284595 |
| 1.5k-2k | Adult | 4 | 63.62% | 1326815 |
| >2k | Child | 2 | 60.17% | 641320 |
| >2k | Adult | 2 | 91.63% | 158457 |

## Worst 20 Samples (lowest accuracy_ratio)
| Rank | Accuracy | HIC True | HIC Pred | Age |
|---|---:|---:|---:|---:|
| 1 | 27.15% | 516 | 1900 | Adult |
| 2 | 30.30% | 692 | 2284 | Adult |
| 3 | 30.80% | 545 | 1768 | Adult |
| 4 | 31.30% | 529 | 1690 | Adult |
| 5 | 32.33% | 674 | 2086 | Adult |
| 6 | 34.73% | 1067 | 3071 | Adult |
| 7 | 35.19% | 545 | 1548 | Adult |
| 8 | 37.58% | 540 | 1438 | Adult |
| 9 | 38.98% | 565 | 1451 | Adult |
| 10 | 43.10% | 792 | 1838 | Adult |
| 11 | 43.63% | 1051 | 2409 | Adult |
| 12 | 45.70% | 1085 | 2374 | Adult |
| 13 | 45.86% | 762 | 1661 | Adult |
| 14 | 45.95% | 626 | 1362 | Adult |
| 15 | 46.66% | 1543 | 3307 | Adult |
| 16 | 46.68% | 612 | 1312 | Adult |
| 17 | 46.69% | 983 | 2105 | Adult |
| 18 | 46.94% | 1188 | 2531 | Adult |
| 19 | 47.80% | 956 | 2001 | Adult |
| 20 | 48.06% | 1408 | 2928 | Adult |

## Best 20 Samples (highest accuracy_ratio)
| Rank | Accuracy | HIC True | HIC Pred | Age |
|---|---:|---:|---:|---:|
| 1 | 99.44% | 719 | 715 | Child |
| 2 | 99.25% | 576 | 572 | Child |
| 3 | 99.20% | 628 | 633 | Child |
| 4 | 99.14% | 698 | 704 | Child |
| 5 | 99.08% | 1050 | 1060 | Child |
| 6 | 98.90% | 719 | 727 | Child |
| 7 | 98.77% | 3041 | 3003 | Adult |
| 8 | 98.72% | 583 | 575 | Child |
| 9 | 98.61% | 819 | 830 | Child |
| 10 | 98.34% | 1050 | 1068 | Child |
| 11 | 98.05% | 979 | 999 | Adult |
| 12 | 97.42% | 763 | 783 | Child |
| 13 | 97.40% | 862 | 885 | Child |
| 14 | 97.20% | 628 | 646 | Child |
| 15 | 97.01% | 731 | 753 | Child |
| 16 | 96.88% | 611 | 630 | Child |
| 17 | 96.87% | 698 | 676 | Child |
| 18 | 96.50% | 748 | 775 | Child |
| 19 | 96.44% | 748 | 776 | Child |
| 20 | 96.21% | 819 | 851 | Child |
