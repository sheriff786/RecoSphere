# Phase 7g: Monitoring + CI

## Monitoring

```powershell
docker compose up --build
```

| Kya | Kahan |
|---|---|
| API metrics (raw) | http://localhost:8000/metrics |
| Prometheus (queries, alerts) | http://localhost:9090 , http://localhost:9090/alerts |
| Grafana dashboard "RecoSphere serving" | http://localhost:3000 (anonymous view; admin login: admin / admin) |

**Dashboard me 12 panels hain:**
- model served, RPS, p95, error share
- RPS by arm/source
- latency p50/p95/p99 by arm
- p95 per stage
- fallback share
- median #1 score by model (drift)
- new-item share
- A/B clicks per request by arm
- model reloads

**Alerts** (`alerts.yml`):

| Alert | Kab | Kyun |
|---|---|---|
| HighLatencyP95 | p95 > 50 ms, 5 min | Phase 6 me load ke under p95 ~30 ms tha, to 50 ms ka matlab asli problem |
| RecommendErrors | 0.1% se zyada requests error ki wajah se popular list par gayi | User ko list dikhti rehti hai, lekin personalization chupchaap band ho jaata hai |
| FallbackShareJump | bina history wale users ka share kal se double | Redis flush ho gaya ya data_version galat hai: sab users "naye" lagne lagte hain |
| ModelReloadFailed | hot-swap fail hua | Purana model chal raha hai, lekin promote ka matlab kuch nahi raha |
| WorkersOnDifferentModels | 10 min tak alag workers alag models serve kar rahe hain | Kuch workers par swap atka hua hai |
| TopScoreDrift | median #1 score kal se > 1 badla | Model ya feature drift. Promotion ke turant baad ye expected hai |

**Multi-worker metrics:** 4 uvicorn workers yaani 4 processes, aur har process ke apne counters hote hain. `PROMETHEUS_MULTIPROC_DIR` (Dockerfile me set hai, aur har start par khaali kiya jaata hai) se `/metrics` saare workers ka total dikhata hai. Bina iske, har scrape alag worker ke numbers dikhayega.

**Yahan kya test hua:**
- `promtool check rules/config` pass hua.
- API ko 2 workers + Redis + multiproc ke saath chalaya. Asli Prometheus 2.53 ne use scrape kiya, simulated traffic chala, aur ek promote + rollback hua. Dashboard ki saari 14 queries ne data diya (error/fallback panels "0" dikhate hain, khaali nahi), aur 6 alert rules `health=ok` ke saath load hue.
- Grafana UI aur Docker compose stack yahan nahi chal saka, kyunki is environment me Docker Hub blocked hai. Tumhare laptop par `docker compose up` pehla asli test hoga.

## CI (GitHub Actions: `.github/workflows/ci.yml`)

Har push/PR par ye chalta hai:
1. `pytest` (tests/, ~40 tests)
2. `scripts/ci_smoke.sh`, ek end-to-end test on synthetic data:
   - `make_synthetic_gold.py` se synthetic data
   - pipeline (train, gate, promote)
   - jaan-boojh kar ek bura model, jise gate ko block karna hi hai (exit 2)
   - API + policy A/B experiment (new-item slots 2 vs 0)
   - parity check
   - simulated traffic + `ab.analyze` (SRM, orphan clicks, users-in-2-arms checks)
   - `/metrics`
3. promtool se alert rules check

Locally bhi yahi chalta hai: `bash scripts/ci_smoke.sh` (WSL / Git Bash). Is environment me ye **~45 s me pass** hua. GitHub par CI abhi ek baar bhi nahi chala, kyunki yahan se CPU torch ka index (download.pytorch.org) blocked tha. Pehla push hi asli test hoga.
