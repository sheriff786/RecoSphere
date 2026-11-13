# Phase 6: Serving (FastAPI + Redis + Docker)

> Baad me jo add hua: hot-swap (7d, neeche), A/B experiments + exposure logs (Phase 8: `docs/AB_TESTING.md`),
> `/metrics` + Grafana (7g: `monitoring/README.md`).

Notebooks 03–06 ne models banaye. Ye folder unhe ek **live API** banata hai jo har request par top-10 deti hai.

## Request ke andar kya hota hai

```
GET /recommend/{user_id}
  1. Redis        user ki history, seen items, stats            (feature_store.py)
  2. history nahi? -> fallback popularity list                 (serving_policy.json, notebook 06)
  3. user tower   64-d vector                                   (retrieval/two_tower.py)
  4. FAISS        two-tower top-200 + recent-pop top-100        (models/faiss.index)
  5. features     ranking/pipeline.assemble()  <- notebook wala SAME code
  6. LightGBM     scores                                        (models/ranker.txt)
  7. re-rank      2 slots naye games ke liye                    (notebook 06 decision)
```

`?debug=true` lagao to har step ka time (ms) bhi aata hai.

## Files

| File | Kaam |
|---|---|
| `serve/app.py` | FastAPI: `/health`, `/recommend/{user_id}`, `/event` |
| `serve/recommender.py` | Startup par models + index load, phir upar wale 7 steps |
| `serve/feature_store.py` | User state: `InMemoryStore` (dev) aur `RedisStore` (production), same interface |
| `serve/config.py` | Saari settings environment variables se |
| `serve/parity_check.py` | **Training-serving skew test**: notebook path vs API path, same top-10 aana chahiye |
| `serve/make_loadtest_users.py` | Locust ke liye user/item lists |
| `loadtest/locustfile.py` | Load test |
| `Dockerfile`, `docker-compose.yml`, `requirements-serve.txt` | Container setup |
| `tests/test_serving.py` | Chhote unit tests (models ki zaroorat nahi) |

## Step 0: Pehle notebooks 05 aur 06 dobara chalao

Phase 6 mein do chhote fixes hue jo `ranking/pipeline.py` aur notebook 06 ko badalte hain:
1. **Tie-break:** barabar score wale items ab hamesha item id se order hote hain. FAISS aur numpy ties alag order mein todte the, jisse API aur notebook ke results alag aa rahe the.
2. **Re-rank order:** naye games insert hone ke baad final list ranker ke order mein sort hoti hai.

Isliye notebook **05 phir 06** ek baar dobara chalao, taaki saved ranker aur results naye code se bane hon. Train-only model cache hai, to notebook 05 jaldi chalega. Numbers mein bahut chhota farak aana chahiye, naye numbers REPORT mein update kar dena.

## Step 1: Bina Docker ke chalao (Windows par bhi)

Repo root se, venv active karke:

```powershell
python -m pip install fastapi "uvicorn[standard]" redis locust
python -m pytest tests/test_serving.py -q
python -m serve.parity_check --n 300          # "PARITY OK" aana chahiye (neeche dekho)
uvicorn serve.app:app --port 8000             # in-memory store
```

**Parity check ke do tests:**
- **Test A (asli skew test):** notebook wale candidates hi API ke feature code mein daale jaate hain. Saare 22 features same aane chahiye. Fail = API features alag tareeke se bana rahi hai, ye asli bug hai.
- **Test B (end-to-end):** poora API path vs notebook path. Kuch users ka top-10 alag ho sakta hai, kyunki do items ke two-tower score ~0.0000001 se alag hon to unka rank swap ho sakta hai, aur LightGBM (tree model) us chhote farak par bhi jump kar sakta hai. Har mismatch check hota hai ki wo sirf aise near-tie se aaya hai ya nahi. Koi bhi unexplained mismatch = FAIL.

Browser mein kholo:
- http://localhost:8000/health
- http://localhost:8000/docs (FastAPI ka interactive page, yahin se endpoints try kar sakte ho)
- http://localhost:8000/recommend/SOME_USER_ID?debug=true (koi bhi `user_id`, `data/gold/Video_Games/user_map.parquet` se)

**Naye user ka flow test karo** (`/docs` page se, ya PowerShell mein):
```powershell
curl http://localhost:8000/recommend/naya_user        # source: fallback
Invoke-RestMethod -Method Post -Uri http://localhost:8000/event -ContentType "application/json" -Body '{"user_id":"naya_user","item_id":"KOI_ITEM_ID"}'
curl http://localhost:8000/recommend/naya_user        # source: personalized
```

## Step 2: Docker + Redis ke saath

Docker Desktop (WSL2 backend) chahiye.

```powershell
docker compose up --build
```

- Pehli build mein PyTorch (CPU) download hoga, isliye kuch minute lagenge.
- Pehli startup par ~95k users ka data Redis mein load hota hai. Uske baad restart turant hota hai (data version match ho to dobara load nahi hota).
- `models/` aur `data/gold/` volumes ki tarah mount hote hain. Retrain ke baad sirf files badlo aur container restart karo, image dobara build nahi karni.

Redis ke saath parity check (Docker ke bahar se, jab compose chal raha ho):
```powershell
$env:REDIS_URL="redis://localhost:6379/0"; python -m serve.parity_check --n 300
```

## Step 3: Load test

```powershell
python -m serve.make_loadtest_users --n 5000
locust -f loadtest/locustfile.py --host http://127.0.0.1:8000 --headless -u 20 -r 5 -t 2m --processes 4 --csv results/loadtest
```

- **`127.0.0.1` likho, `localhost` nahi.** Windows par `localhost` pehle IPv6 try karta hai, aur har naye connection par ~2 second ka delay aata hai.
- **`--processes 4`:** Locust bhi ek Python process hai. Bina iske, zyada users par Locust khud bottleneck ban sakta hai.
- **Personalized aur fallback alag rows mein dikhte hain.** Report personalized wali row se karo.

### Ek worker vs bahut workers (GIL)

Ek Python process lagbhag **ek core** jitna kaam kar paata hai (GIL). Isliye ek worker par users badhane se throughput ek limit par ruk jaata hai, aur sirf latency badhti hai, chahe CPU ~20% hi dikhe. Zyada cores use karne ke liye zyada worker processes chalao. Workers ko user state **Redis** se share karni padti hai:

```powershell
# Redis chahiye (Docker: docker run -p 6379:6379 redis:7-alpine)
$env:REDIS_URL="redis://localhost:6379/0"
uvicorn serve.app:app --port 8000 --workers 4
```

Pehli startup par sirf ek worker Redis mein data load karta hai (load lock), baaki intezaar karte hain. Docker compose mein `API_WORKERS` se workers set karo.

**Experiment jo REPORT mein jaayega:** same `-u` (jaise 50) par workers 1 → 2 → 4 karke RPS aur p95 naapo, aur Task Manager mein CPU % bhi dekho. Agar workers ke saath RPS badhta hai, to bottleneck GIL tha.

`-u` (ek saath kitne users) ko 5, 20, 50 karke chalao, aur dekho ki p95 kahan bigadne lagta hai.

**REPORT mein kya likhna hai:** har `-u` ke liye RPS, p50, p95, failures, aur **hardware** (CPU model, cores, RAM, Docker ya bina Docker). `?debug=true` se dekho ki kaunsa step sabse slow hai.

⚠️ Resume par "served X users" kabhi mat likhna, asli traffic nahi hai. Sahi line: *"load-tested with Locust at R RPS with L ms p95 on [hardware]"*.

## Settings (environment variables)

| Variable | Default | Kaam |
|---|---|---|
| `REDIS_URL` | (khaali) | Khaali = in-memory store. Set = Redis |
| `RANKFORGE_CATEGORY` | `Video_Games` | |
| `RANKFORGE_DATA` / `RANKFORGE_REPO` | `data` / `.` | Paths (repo root se chalao) |
| `NEW_ITEM_SLOTS` | `2` | Re-rank rule. `0` = re-rank band |
| `SERVE_THREADS` | `1` | Har request ke compute threads (LightGBM, torch, BLAS). 1 se p95 stable rehta hai |
| `API_WORKERS` | `1` | Sirf Docker: uvicorn worker processes. 1 se zyada ho to `REDIS_URL` chahiye |
| `FORCE_RELOAD` | `0` | `1` = Redis data zabardasti dobara load |
| `MODEL_WATCH_SECONDS` | `10` | Itne seconds me `production.json` check hota hai, aur pointer badla ho to naya model hot-swap hota hai. `0` = watch band (sirf `/admin/reload` se reload) |
| `ADMIN_TOKEN` | (khaali) | Set ho to `POST /admin/reload` ko header `X-Admin-Token` chahiye |
| `MODEL_DIR` | (khaali) | Kisi bhi model folder ko pin kar do (pointer ignore hoga) |
| `EXPERIMENT_CONFIG` | `experiments/active.yaml` | A/B config. File nahi hai = koi experiment nahi (Phase 8) |
| `EXPOSURE_DIR` | (khaali) | Set ho to `exposures-<pid>.jsonl` aur `clicks-<pid>.jsonl` yahan likhe jaate hain |
| `PROMETHEUS_MULTIPROC_DIR` | (khaali) | Multi-worker `/metrics` ke liye khaali folder (Docker me set hai) |

## Phase 7d: bina restart model badalna

```
promote / rollback  ->  production.json badla
   -> har worker (<= MODEL_WATCH_SECONDS): naya Recommender BACKGROUND thread me load
      (purana model tab tak traffic serve karta rehta hai)
   -> smoke test: 20 asli users + 1 unknown user ke liye recommend(), output check
   -> ek line ka swap: self.current = new   (jo requests chal rahi hain, woh purane object par khatam hoti hain)
   -> fail? purana model chalta rehta hai, /health me last_reload_error, aur us model ka retry nahi hota
      (jab tak pointer badle nahi ya /admin/reload?force=true na chalao)
```

- **Feature store reload nahi hota.** User records data par depend karte hain, model par nahi (`data_version` = rows + last ts + hist_len). Isliye swap ke dauraan aaye live events bache rehte hain.
- **Multi-worker:** har worker khud pointer dekhta hai, isliye Redis pub/sub ki zaroorat nahi. 2 workers ka test: dono ne ~2 s ke andar swap kiya.
- **Response me `model_run`** aata hai. Isse har request ka model pata chalta hai, aur Phase 8 (A/B) me yahi exposure logging ka base banega.

Synthetic data par tested (4–6 threads lagatar requests bhej rahe the, aur beech me promote ya rollback hua):

| Scenario | Requests | Errors | Max latency |
|---|---|---|---|
| In-memory, 2 swaps (good1 → seed2 → good1) | 8,689 | 0 | 206 ms |
| Toota model (`ranker.txt` corrupt) | 4,041 | 0 | 62 ms (purana model hi chalta raha) |
| Redis, 2 workers, 1 swap | 7,095 | 0 | 94 ms. Swap ke baad event yaad raha, parity 200/200 |

Load ke dauraan us worker ka CPU busy hota hai, isliye kuch requests slow ho sakti hain. Asli data par load time zyada hoga, kyunki usme parquet, torch aur FAISS index load hote hain.

## Design decisions (interview ke liye)

| Decision | Kyun |
|---|---|
| Feature code `ranking/pipeline.py` mein, notebook aur API dono wahi use karte hain | Training-serving skew se bachne ke liye. `parity_check` ise prove karta hai |
| Ties ka fixed rule (score, phir item id) | Parity check ne pakda ki FAISS aur numpy ties alag todte hain, jisse ranker ke scores badal jaate the |
| Embedding version check (startup par) | Index ek model se bana ho aur user tower doosre se, to results chupchaap kachra ho jaate hain. Mismatch par API start hi nahi hoti |
| `/event` sirf user state update karta hai | Item popularity batch job (Phase 7) se refresh hoti hai. Isse online aur offline features consistent rehte hain |
| Error par crash nahi, popular list | Recommendation fail ho jaaye, phir bhi page khaali nahi dikhna chahiye |
| Models image mein nahi, volumes mein | Retrain = files badlo, image rebuild nahi |

## Known limits

- Item popularity model load ke waqt fix hoti hai (`t_ref` = test window ki shuruaat). Events usse update nahi karte. Hot-swap par bhi gold data wahi rehta hai. Naya data snapshot aane par hi popularity badlegi (7e).
- In-memory store har worker ka alag hota hai. Ek se zyada workers ho to hamesha Redis use karo.
- Torch ka pura model sirf user tower ke liye load hota hai. Ise ONNX mein export karke image chhoti ho sakti hai, ye baad ki optimization hai.
