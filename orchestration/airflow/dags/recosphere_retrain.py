"""
RecoSphere weekly retrain DAG (Phase 7f).

    prepare -> train_retrieval -> build_index -> train_ranker -> evaluate -> gate -> promote -> verify_rollout
                                                                                  +-> rollback (only if verify fails)

Why Airflow on top of `python -m pipeline.run` (which can already run everything in one go):
  - per-step retries: a crash in train_ranker retries train_ranker only (steps resume from the run folder)
  - the gate is a real branch: exit code 2 -> task SKIPPED -> promote/verify do not run, the DAG stays green
  - verify_rollout: after promotion, wait until the API actually serves the new model; if it never does
    (bad files, hot-swap failed) -> automatic rollback
  - backfill = time travel (Phase 7e): with params.time_travel=true each run uses split.mode=cutoff and
    cutoff = the run's logical date, so `airflow dags backfill -s 2022-01-01 -e 2023-01-01` replays history
  - one place to see history, logs, durations, failures

Every task is a shell command, so the DAG has no Python dependency on the project: it runs where the repo
and its venv are (RECOSPHERE_REPO / RECOSPHERE_PYTHON). Data ingestion (notebooks 01-02, Spark) is NOT in
this DAG yet: it starts from the existing data/gold snapshot (check_data fails if it is missing).
"""
from datetime import datetime, timedelta
import os

from airflow import DAG

try:                                   # Airflow 3
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:                    # Airflow 2.x
    from airflow.operators.bash import BashOperator

REPO = os.environ.get("RECOSPHERE_REPO", os.path.expanduser("~/RecoSphere"))
PY = os.environ.get("RECOSPHERE_PYTHON", "python")
API = os.environ.get("RECOSPHERE_API", "http://127.0.0.1:8000")
EXTRA = os.environ.get("RECOSPHERE_PIPELINE_ARGS", "")    # e.g. "--set two_tower.epochs=4" (applied to every step)
RUN_ID = "af-{{ ts_nodash }}"
TT = ("{% if params.time_travel %} --set split.mode=cutoff --set split.cutoff={{ ds }}"
      " --set namespace=replay/airflow --set mlflow.experiment=recosphere-replay-airflow{% endif %}")
STEPS = ["prepare", "train_retrieval", "build_index", "train_ranker", "evaluate"]


def step(name, extra=""):
    return f"cd '{REPO}' && {PY} -m pipeline.run --run-id {RUN_ID} --steps {name}{TT} {EXTRA}{extra}"


with DAG(
    dag_id="recosphere_retrain",
    description="weekly retrain -> eval gate -> promote -> verify (RecoSphere Phase 7f)",
    start_date=datetime(2026, 11, 1),
    schedule="0 3 * * 1",                       # Mondays 03:00
    catchup=False,
    max_active_runs=1,                          # never two trainings at once (GPU, production pointer)
    params={"time_travel": False},
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5), "execution_timeout": timedelta(hours=3)},
    tags=["recosphere"],
) as dag:
    check_data = BashOperator(
        task_id="check_data",
        bash_command=f"cd '{REPO}' && {PY} -c \"from pipeline.config import load_config, Paths; "
                     "import sys; g=Paths(load_config()).gold; "
                     "sys.exit(0 if (g / 'train.parquet').exists() else 1)\"",
    )
    tasks = [check_data] + [BashOperator(task_id=s, bash_command=step(s)) for s in STEPS]
    gate = BashOperator(task_id="gate", bash_command=step("gate"), retries=0)
    # gate step itself always exits 0; the DECISION is applied by the promote step (exit 2 = not promoted)
    promote = BashOperator(task_id="promote", bash_command=step("promote"), skip_on_exit_code=2, retries=0)
    verify = BashOperator(
        task_id="verify_rollout",
        retries=0,
        bash_command=(
            "{% if params.time_travel %}echo 'time travel: nothing is served'{% else %}"
            f"for i in $(seq 60); do "
            f"  run=$(curl -sf {API}/health | {PY} -c \"import sys,json; print(json.load(sys.stdin)['model_run'])\"); "
            f"  [ \"$run\" = \"{RUN_ID}\" ] && echo \"API serves $run\" && exit 0; sleep 10; "
            f"done; echo 'API never switched to {RUN_ID}'; exit 1"
            "{% endif %}"),
    )
    rollback = BashOperator(
        task_id="rollback",
        trigger_rule="one_failed",               # only when verify_rollout failed
        retries=0,
        bash_command=f"cd '{REPO}' && {PY} -m pipeline.run --rollback",
    )
    for a, b in zip(tasks, tasks[1:] + [gate]):
        a >> b
    gate >> promote >> verify >> rollback
