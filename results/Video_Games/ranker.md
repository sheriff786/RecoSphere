Candidates: two-tower top-200 + recent-pop top-100. Candidate recall (test): two_tower=0.1291, recent_pop=0.0810, union=0.1664

| model | all/recall@50 | all/recall@100 | all/ndcg@10 | warm/recall@100 | cold/recall@100 |
|---|---|---|---|---|---|
| two_tower order (sanity) | 0.0647 | 0.0975 | 0.0146 | 0.1327 | 0.0378 |
| recent_pop order (sanity) | 0.0537 | 0.0916 | 0.0102 | 0.1441 | 0.0000 |
| rrf (two_tower + pop) | 0.0717 | 0.1063 | 0.0168 | 0.1556 | 0.0213 |
| lightgbm ranker | 0.0940 | 0.1348 | 0.0184 | 0.1951 | 0.0282 |
