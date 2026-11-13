# Phase 7a–7c: Training pipeline, MLflow, eval gate

Notebooks 03–06 se jo kaam haath se hota tha, ab woh **ek command** se hota hai:

```
prepare → train_retrieval → build_index → train_ranker → evaluate → gate → promote
```

Har run apne folder me jata hai, aur API hamesha **production.json** waale run ko serve karti hai.

## Setup

```powershell
python -m pip install mlflow pyyaml
```

Pehle notebooks 02 aur 03 (Tasks 1–2) chal chuke hone chahiye, kyunki `data/gold/<cat>/` aur `item_text_emb_minilm.npy` wahi banate hain.

## Commands

```powershell
# 1. Notebook waale models ko pehla "production" run bana do (ek baar)
python -m pipeline.run --register-existing

# 2. Full run (train + gate + promote)
python -m pipeline.run

# 3. Koi bhi config value override karo
python -m pipeline.run --set two_tower.epochs=6 --set ranker.num_leaves=31

# 4. Ek run ke kuch steps dobara chalao (run ka apna config.yaml use hota hai)
python -m pipeline.run --run-id 20261111-120000 --steps train_ranker,evaluate,gate,promote

# 5. Gate fail hua, phir bhi promote karna hai (sirf soch samajh ke)
python -m pipeline.run --run-id <id> --steps promote --force

# 6. Production ko pichhle model pe wapas le jao
python -m pipeline.run --rollback

# 7. MLflow UI (repo root se)
mlflow ui --backend-store-uri sqlite:///mlflow.db      # -> http://127.0.0.1:5000
```

**Exit code:** `0` matlab ok hai, `2` matlab gate fail hua aur model promote nahi hua. Scheduler (cron/Airflow, 7f) isi exit code se react karega.

## Folder layout

```
models/Video_Games/
  production.json            <- API yahi padhti hai  {"run_id": "...", "path": "runs/<id>", ...}
  production_history.jsonl   <- har promotion/rollback ki ek line (audit trail)
  runs/<run_id>/
    config.yaml              <- is run ka exact config (reproducible)
    pipeline.log
    data/{train,val,test}.parquet, manifest.json   <- is run ka data + data_hash
    two_tower_trainonly.pt   <- val candidates ke liye (ranker training, leakage-free)
    two_tower.pt, item_emb.npy, faiss.index, faiss_meta.json
    ranker.txt, ranker_meta.json, serving_policy.json
    metrics.json, gate_report.json, production_eval_<prod_id>.json
```

`production.json` me path **relative** hota hai (`runs/<id>`). Isliye wahi pointer Windows, WSL aur Docker (jaha `models/` kahin aur mount hota hai) teeno me kaam karta hai. Model files kabhi delete ya copy nahi hoti, isliye rollback turant ho jata hai.

## Gate: model production me kab jata hai (Phase 7d version)

Saare checks **candidate ke test window** par hote hain:

| # | Check | Kyun |
|---|---|---|
| 1 | Metrics finite, candidate recall > 0 | Toota hua training (NaN ya empty) pakadna |
| 2 | Ranker recent-popularity se behtar (NDCG@10, Recall@100) | Popularity list se kharab model ka koi fayda nahi |
| 3 | Candidate ka data production se purana nahi | Leakage guard (neeche) |
| 4 | Regression test (paired bootstrap): ranker NDCG@10, ranker Recall@100, **candidate recall** | Naya model purane se kharab nahi hona chahiye |

**Check 4 kaise kaam karta hai:**
1. Production model ko candidate ke test window par **dobara evaluate** karte hain, taaki dono **same users** par measure hon. `per_user_eval.npz` me har user ka score save hota hai.
2. Users ko resample karte hain (2000 baar). Har baar *dono* models ke liye same resample use hota hai, isliye ise "paired" kehte hain. Isse milta hai: relative change + 95% CI.
3. **Block** tab hota hai jab:
   - **pakka kharab ho:** CI ka upper bound < −1% (`min_effect`). Matlab hum confident hain ki model 1% se zyada kharab hai; **ya**
   - **bada drop ho:** point change < −5% (`max_point_drop`). Ye safety net hai, jab CI itna chauda ho ki confident hona mushkil ho.

**Ye rule kyun chuna** (pehle do rule try kiye, dono fail hue):
- *Fixed 1% tolerance*: ye sirf point value dekhta hai, aur test set ka noise ignore karta hai.
- *Non-inferiority* ("saboot do ki 2% se zyada kharab nahi"): synthetic data par seed 1 ka change sirf −0.2% tha, lekin CI ±2% tha, isliye FAIL ho gaya. NDCG@10 bahut sparse hai (mean ~0.02), to asli data par CI aur chauda hoga. Is rule se **identical model bhi pass nahi hota**, aur har retrain block ho jaata.

**Bootstrap kya measure karta hai aur kya nahi:** ye test-set ka noise measure karta hai ("doosre users par bhi ye model aisa hi dikhega?"). Training randomness (seed variance) ye measure nahi karta. Same config ke do seeds asal me do alag models hote hain, aur gate *model* ko judge karta hai. Asli data par 3 seeds: NDCG@10 0.0202 ± 0.0008 (~4%). Ranker ka best iteration 18 se 81 ke beech badalta raha. Agar zyada retrains block hon, to fix ranker training ko stable banana hai, gate ko dheela karna nahi.

**Leakage guard (check 3):** ek time-travel run (cutoff 2021-06) chalaya, jab production `good1` tha. `good1` Nov 2021 tak ke data par train hua tha, yaani usne candidate ka test window **pehle hi dekh liya tha**. Uska score inflated tha, aur gate ne purane data wale model ko promote kar diya. Ab gate ye case block karta hai.

**Gate policy hamesha aaj ke config se aati hai.** `--run-id X --steps gate` chalao to `gate` aur `mlflow` sections current `config.yaml` se aate hain, run ke saved config se nahi. Training config run ka hi rehta hai.

**Stale gate report:** agar `--steps promote` akela chalao aur gate report kisi aur production ke against bani thi (beech me rollback ya promotion hua tha), to model promote nahi hoga.

**Rollback** history me current se pichhle model par le jaata hai. Do baar rollback karoge to wapas usi model par aa jaoge (undo/redo jaisa).

## MLflow me kya jata hai

| Type | Kya |
|---|---|
| params | poora config (flattened, ~36 values) |
| metrics | training time, ranker/two-tower/recent-pop ke all/warm/cold recall@k aur ndcg@k, candidate recall, served metrics (`@` ko `_at_` likha jata hai) |
| tags | `run_id`, `data_hash`, `split_mode`, `emb_version`, `gate_passed`, `compared_to`, `promoted` (kabhi promote hua ya nahi; **abhi** production me kaun hai, uska sach `production.json` hai) |
| artifacts | metrics.json, serving_policy.json, gate_report.json |

Backend ek local SQLite file hai (`mlflow.db`), server ki zaroorat nahi. Agar MLflow install nahi hai ya `mlflow.enabled=false` hai, to pipeline phir bhi chalti hai, bas tracking skip hoti hai. Agar MLflow ka koi call fail ho, to sirf warning aati hai aur training nahi rukti.

## Serving

`serve/config.py` ab `models/<cat>/production.json` follow karta hai.

- Pointer nahi hai → purana flat folder `models/<cat>/` use hota hai (Phase 6 behaviour).
- `MODEL_DIR` env var set hai → woh sab kuch override kar deta hai.

`/health` me `model_run`, `model_dir`, `reloads` aur `last_reload_error` dikhte hain. **Restart ki zaroorat nahi** (Phase 7d). Har worker `MODEL_WATCH_SECONDS` (default 10) me pointer check karta hai aur naya model khud load kar leta hai. Details `serve/README.md` me hain.

## Synthetic data par test kya hua

Sirf pipeline sahi chalti hai ya nahi, ye dekhne ke liye. Ye numbers asli Video_Games ke **nahi** hain.

| Run | Kya kiya | NDCG@10 | Gate | Result |
|---|---|---|---|---|
| `nb-…` | `--register-existing` | 0.2884 | (pehla run) | production |
| `good1` | full run, same seed | 0.2884 | PASS | promoted |
| `bad1` | epochs=1, n_estimators=2 | 0.1400 | FAIL (regression) | blocked, exit 2 |
| `tt-2021-06` | cutoff mode, purana data | 0.3878 | FAIL (older data) | blocked, exit 2 |

Aur ye bhi check hua:
- Pipeline evaluation = notebook 05 ke numbers, exact match (0.2884 / 0.9095). Same seed se run bit-for-bit reproducible hai.
- `good1` par serving parity: Test A PASS, Test B 300/300.
- `/health` ne `model_run: good1` dikhaya.
- `tests/test_mlops.py`: 7 tests. Poora suite: 21 passed.

## Asli data par expected time

Retrieval do baar train hota hai: train-only model + train+val model. Notebook 03 me ek model ko CPU par ~1 ghanta laga tha, isliye poora run CPU par ~2+ ghante lega. GPU ho to `train_retrieval` bahut tez hoga. Pipeline ka device auto-detect hai (`cuda` mil gaya to wahi use hota hai).

## Phase 7e: Time-travel replay

"Agar ye pipeline pichhle 2 saal se har 90 din chal rahi hoti, to kya hota?"

```powershell
python -m pipeline.replay --name q90 --start 2022-01-01 --end 2023-04-01 --every-days 90
```

**Har cutoff `c` par** poori pipeline cutoff mode me chalti hai: `train < c−365d <= val < c <= test < c+90d`. Uske baad gate chalta hai, aur pass ho to promote. Ye sab `models/<cat>/replay/<name>/` me hota hai, jiska apna `production.json` hai. **Asli production (jo API serve karti hai) ko replay kabhi nahi chhoota.**

**Decay matrix:** har model ko har baad ke window par evaluate kiya jaata hai. Model ke weights purane rehte hain, lekin user history taaza hoti hai, jaisa serving me hota hai jab retrain na karo.

| Output | Kya |
|---|---|
| `timeline.md` | har cutoff: data size, metrics, gate decision |
| `decay.md`, `decay.png` | model (row) × window (column), aur "stale vs fresh model" by age |
| `policies.md` | never / every k periods / gated / always retrain: inme se har policy se production kitna achha rehta |

Same command dobara chalao to jo runs ho chuke hain woh skip ho jaate hain (resume).

Synthetic data par (6 cutoffs, 120 din ke gap, ~7 min): 360 din purana model fresh model se NDCG@10 par −3.6% tha, aur "never retrain" policy −3.6%. Lekin har window me sirf ~880 test users the, isliye policies ke beech ke chhote farak noise hain.

Asli data par ise GPU ke saath chalao. Ek cutoff ~5 min ka hai (training + eval), aur decay ka har pair ~40 s.

**Known limit:** catalog (item features, text) 2023 ka snapshot hai. Isliye kisi purane cutoff par bhi woh items recommend ho sakte hain jo us waqt release hi nahi hue the. Retrieval me release date ka filter abhi nahi hai.
