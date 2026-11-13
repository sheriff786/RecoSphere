# Phase 7f: Orchestration (kaun chalayega retrain, aur kab)

Pipeline ek command me poori chalti hai (`python -m pipeline.run`), har step resume ho sakta hai, aur exit code batata hai ki kya hua (0 = promoted, 2 = gate ne mana kiya). Orchestrator ka kaam sirf itna hai: **schedule karna, fail hone par retry karna, history rakhna, aur alert karna.** Is project ke liye do option hain:

| | Windows Task Scheduler (`scripts/retrain.ps1`) | Airflow DAG (`orchestration/airflow/`) |
|---|---|---|
| Setup | 1 command | WSL2 ya Docker, Airflow install, scheduler + webserver |
| Retry | poora run dobara | sirf fail hua step (pipeline steps resume hote hain) |
| Gate ka faisla | exit code 2 | promote task **skipped** dikhta hai, DAG green rehta hai |
| Promote ke baad check | nahi | `verify_rollout`: API naya model serve kar rahi hai? Nahi to **auto rollback** |
| Backfill / time travel | `python -m pipeline.replay` | `airflow dags backfill` + `time_travel=true` = 7e replay |
| History / logs | `logs/retrain-*.log` | UI: har run, har task, duration, logs |
| Kab chuno | ek machine, ek model, hafte me ek baar | multiple pipelines ya dependencies (Spark ingestion → train), team, alerting |

**Meri recommendation:** abhi `retrain.ps1` kaafi hai, kyunki ek laptop aur ek weekly job hai. Airflow DAG isliye rakha hai ki dikhe ki pipeline orchestrator-ready hai (steps idempotent hain, exit codes ka matlab hai, sab kuch config se chalta hai), aur jab Spark ingestion (notebooks 01–02) script ban jaayega, to woh isi DAG ka pehla task banega.

## Option A: Task Scheduler

```powershell
schtasks /Create /TN "RecoSphere retrain" /SC WEEKLY /D MON /ST 03:00 `
  /TR "powershell -NoProfile -ExecutionPolicy Bypass -File `"$PWD\scripts\retrain.ps1`""
schtasks /Run /TN "RecoSphere retrain"        # abhi ek baar chala ke dekho
type logs\retrain-*.log
```

- Lock file hone se ek waqt me do runs nahi chal sakte.
- "Last Run Result" me: `0x0` = promoted, `0x2` = gate fail (normal hai), aur kuch bhi = error.
- Laptop us waqt so raha ho to task nahi chalega. Task ki settings me "Wake the computer to run this task" ya "Run task as soon as possible after a scheduled start is missed" on kar do.

## Option B: Airflow (WSL2 me)

Airflow Windows par native nahi chalta. WSL2 (Ubuntu) me:

```bash
python3.11 -m venv ~/af && source ~/af/bin/activate
pip install "apache-airflow==2.10.5" --constraint \
  "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.11.txt"
export AIRFLOW_HOME=~/airflow
export RECOSPHERE_REPO=/mnt/e/Sheriff_faang/full_end_to_end_project_implementation/RecoSphere
export RECOSPHERE_PYTHON=$HOME/reco-venv/bin/python     # project ka Linux venv (WSL me alag banana padega)
export RECOSPHERE_API=http://127.0.0.1:8000
mkdir -p $AIRFLOW_HOME/dags && cp $RECOSPHERE_REPO/orchestration/airflow/dags/recosphere_retrain.py $AIRFLOW_HOME/dags/
airflow standalone                                        # UI: http://localhost:8080 (password terminal me dikhega)
```

DAG ka flow:
```
check_data -> prepare -> train_retrieval -> build_index -> train_ranker -> evaluate -> gate -> promote -> verify_rollout
                                                                                                     \-> rollback (sirf verify fail hone par)
```

```bash
airflow dags test recosphere_retrain 2026-11-16                         # ek run, scheduler ke bina
airflow dags backfill recosphere_retrain -s 2022-01-03 -e 2022-12-26 \
  --conf '{"time_travel": true}'                                        # = time travel replay (alag namespace)
```

**Dhyan rakhna:**
- WSL ke andar GPU (CUDA) ke liye NVIDIA ka WSL driver chahiye. Windows wala driver aam taur par ye support karta hai, `nvidia-smi` WSL me chala ke check karo.
- `/mnt/e/...` se padhna Linux disk se dheema hota hai. Agar ye dikkat bane, to repo ko WSL ke home folder me clone karo.

## Kya test hua (synthetic / is environment me)

- `scripts/retrain.ps1`: Windows nahi hai, isliye **yahan nahi chala**. Pehli baar tum `schtasks /Run` se khud chala ke dekhna.
- Airflow DAG: Airflow 2.10.5 (Python 3.11) me synthetic data par `airflow dags test` se do baar chalaya:

| Run | Gate | promote | verify_rollout | API ne kya serve kiya | DAG |
|---|---|---|---|---|---|
| 2026-11-16 (normal config) | PASS | success | success ("API serves af-20261116T000000") | naya model, hot-swap ke through | success |
| 2026-11-23 (epochs=1, 2 trees) | FAIL (regression) | **SKIPPED** (exit 2) | skipped | purana model hi chalta raha | success |

- Extra pipeline args (jaise synthetic par `--set two_tower.text_encoder=tfidf`) `RECOSPHERE_PIPELINE_ARGS` env se dete hain.
- Airflow 3 ke import paths ke liye DAG me fallback hai, lekin test sirf 2.10.5 par hua hai.
