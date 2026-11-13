### Accuracy + beyond-accuracy (test, top-10 shown)

| model | recall@10 | ndcg@10 | recall@100 | cold_recall@100 | coverage@10 | gini@10 | novelty | new_item_share@10 | distinct_publishers@10 | top_publisher_share@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| recent_popularity | 0.017 | 0.0102 | 0.0916 | 0.0 | 0.0007 | 0.9996 | 10.1701 | 0.0 | 6.9105 | 0.4089 |
| two_tower | 0.022 | 0.0146 | 0.0975 | 0.0378 | 0.2021 | 0.975 | 11.8843 | 0.0506 | 5.5926 | 0.4218 |
| ranker | 0.0297 | 0.0184 | 0.1348 | 0.0282 | 0.1105 | 0.9936 | 11.9346 | 0.0242 | 6.121 | 0.3581 |

### Segments by history length

| history_len | users | share_of_users | recent_popularity/recall@100 | recent_popularity/ndcg@10 | two_tower/recall@100 | two_tower/ndcg@10 | ranker/recall@100 | ranker/ndcg@10 |
|---|---|---|---|---|---|---|---|---|
| 1-2 | 2968 | 0.1371 | 0.0827 | 0.0146 | 0.0835 | 0.0165 | 0.1106 | 0.0195 |
| 3-5 | 9860 | 0.4554 | 0.0953 | 0.0097 | 0.0989 | 0.0146 | 0.1362 | 0.018 |
| 6-10 | 5663 | 0.2615 | 0.0942 | 0.0096 | 0.1009 | 0.0132 | 0.1449 | 0.0181 |
| 11-20 | 2231 | 0.103 | 0.0838 | 0.0089 | 0.1026 | 0.0154 | 0.1353 | 0.0199 |
| 21+ | 931 | 0.043 | 0.0824 | 0.0088 | 0.0952 | 0.0136 | 0.1342 | 0.0171 |

### Cold-user fallback (Recall@100) — chosen on val: last_30_days

| index | val (5,234 users) | test (3,381 users) |
|---|---|---|
| all_time_popularity | 0.0492 | 0.0308 |
| recent_popularity_10pct | 0.1064 | 0.0689 |
| last_30_days | 0.1141 | 0.0652 |
| last_90_days | 0.1117 | 0.0703 |

### Re-rank rules on ranker top-100

| rule | recall@10 | ndcg@10 | cold_recall@10 | coverage@10 | gini@10 | novelty | new_item_share@10 | distinct_publishers@10 | top_publisher_share@10 |
|---|---|---|---|---|---|---|---|---|---|
| ranker (no re-rank) | 0.0297 | 0.0184 | 0.0055 | 0.1105 | 0.9936 | 11.9346 | 0.0242 | 6.121 | 0.3581 |
| publisher cap 3 | 0.0283 | 0.0177 | 0.0043 | 0.1031 | 0.9943 | 11.8617 | 0.0198 | 6.8081 | 0.2583 |
| new-item slots 2 | 0.0304 | 0.0186 | 0.0141 | 0.1254 | 0.9911 | 12.8135 | 0.1421 | 5.9677 | 0.3755 |
| cap 3 + new slots 2 | 0.0297 | 0.0182 | 0.0134 | 0.119 | 0.9918 | 12.7692 | 0.1402 | 6.4456 | 0.3061 |

