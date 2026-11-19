"""评测子系统：指标（metrics）与 CLI（run_eval）。"""

from eval.metrics import (  # noqa: F401
    aggregate_cohort,
    aggregate_retrieval,
    confusion_matrix,
    first_hit_rank,
    hit_at_k,
    reciprocal_rank,
    recall_at_k,
)
