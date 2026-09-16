"""Executable reproductions of known issues in the original scripts.

These functions do not patch ``model_full_SST.py``. They show why a printed
test score from that file is not a full-set metric.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchOverwriteDemo:
    n_batches: int
    batch_size: int
    n_items: int
    extend_preds: list[int]
    overwrite_preds: list[int]
    extend_accuracy: float
    overwrite_accuracy: float
    overwrite_covers: int
    note: str


def simulate_test_loop(n_items: int = 1186, batch_size: int = 256) -> BatchOverwriteDemo:
    """Reproduce the extend-vs-assign bug in ``model_full_SST.py``.

    The validation loop does ``all_preds.extend(...)``.
    The test loop does ``all_preds = preds`` and keeps only the last batch.
    """
    labels = [i % 3 for i in range(n_items)]
    # A dummy model that is correct on every item except the last batch,
    # where it is wrong on all but the first example. That makes the two
    # aggregation rules disagree sharply.
    batches = []
    start = 0
    while start < n_items:
        end = min(start + batch_size, n_items)
        batches.append(list(range(start, end)))
        start = end

    extend_preds: list[int] = []
    overwrite_preds: list[int] = []
    for batch_i, idxs in enumerate(batches):
        last = batch_i == len(batches) - 1
        preds = []
        for offset, idx in enumerate(idxs):
            if last and offset > 0:
                preds.append((labels[idx] + 1) % 3)
            else:
                preds.append(labels[idx])
        extend_preds.extend(preds)
        overwrite_preds = preds  # the bug

    def acc(preds: list[int], gold: list[int]) -> float:
        return sum(int(p == y) for p, y in zip(preds, gold)) / len(gold)

    last_gold = [labels[i] for i in batches[-1]]
    return BatchOverwriteDemo(
        n_batches=len(batches),
        batch_size=batch_size,
        n_items=n_items,
        extend_preds=extend_preds,
        overwrite_preds=overwrite_preds,
        extend_accuracy=acc(extend_preds, labels),
        overwrite_accuracy=acc(overwrite_preds, last_gold),
        overwrite_covers=len(overwrite_preds),
        note=(
            "overwrite_accuracy is computed on the last batch only, which is what "
            "the current test loop prints. extend_accuracy is the full-set number."
        ),
    )


KNOWN_BUGS = (
    {
        "id": "full-sst-test-overwrite",
        "file": "model_full_SST.py",
        "severity": "high",
        "summary": "Test loop assigns all_preds = batch instead of extend.",
        "effect": "Printed test metrics cover only the last batch (~158 of 1186 rows at batch_size=256).",
    },
    {
        "id": "checkpoint-comment-f1",
        "file": "model_full_SST.py",
        "severity": "low",
        "summary": "Comment and save log say F1; the predicate uses validation accuracy.",
        "effect": "Easy to misquote which metric selected the checkpoint.",
    },
    {
        "id": "hardcoded-three-classes",
        "file": "model_ZuCo_SST.py / model_full_SST.py",
        "severity": "low",
        "summary": "Fusion loss uses logits.view(-1, 3) even though num_labels exists.",
        "effect": "Changing num_labels is not enough to reconfigure the fusion head.",
    },
    {
        "id": "subject3-reindex",
        "file": "utils_ZuCo.py + get_average_sentence_level.py",
        "severity": "high",
        "summary": "Subject 3 drops sentences 150–249 and 399, then reindexes 0..298.",
        "effect": "Index-wise averages mix the wrong reader-3 sentence into ids 150–298.",
    },
    {
        "id": "path-leftovers",
        "file": "read_ZuCo_mat.py, convert_full_SST.py, utils_ZuCo.py",
        "severity": "medium",
        "summary": "Producers point at folders that are not in this clone.",
        "effect": "Re-running producers raises FileNotFoundError; consumer CSVs are already present.",
    },
    {
        "id": "spilt-typo",
        "file": "ZuCo_SST_data/spilt.py, SST_data/spilt.py",
        "severity": "low",
        "summary": "Filename is spilt, not split.",
        "effect": "Cosmetic, but the ZuCo 40-row valid split is also unstratified.",
    },
)
