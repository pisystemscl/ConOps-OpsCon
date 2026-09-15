from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Callable
from uuid import uuid4

import pandas as pd

from src.pipeline.experiment_runner import ExperimentRunner


@dataclass
class BatchRunResult:
    run_id: str
    by_document: pd.DataFrame
    by_configuration: pd.DataFrame
    best_configuration: str


class BatchDocumentRunner:
    def __init__(
        self,
        runner_factory: Callable[[], ExperimentRunner],
    ):
        self.runner_factory = runner_factory

    def run(
        self,
        documents: list[tuple[str, str]],
        *,
        fixed_model: str,
        n_repeats: int = 1,
    ) -> BatchRunResult:
        if not documents:
            raise ValueError("Au moins un document est requis.")
        batch_run_id = uuid4().hex
        rows = []
        for document_name, document_text in documents:
            runner = self.runner_factory()
            results, _, _ = runner.run_strategy_comparison(
                document_text,
                fixed_model=fixed_model,
                document_name=document_name,
                n_repeats=n_repeats,
            )
            for result in results:
                row = result.model_dump()
                row["batch_run_id"] = batch_run_id
                rows.append(row)
        by_document = pd.DataFrame(rows)
        successful = by_document[by_document["status"] == "success"]
        aggregates = []
        for configuration, group in successful.groupby("configuration"):
            scores = [float(value) for value in group["final_score"]]
            score_std = pstdev(scores) if len(scores) > 1 else 0.0
            aggregates.append(
                {
                    "configuration": configuration,
                    "documents_count": len(group),
                    "mean_score": round(mean(scores), 3),
                    "std_score": round(score_std, 3),
                    "stability_score": round(max(0.0, 100 - score_std), 3),
                    "mean_grounding_score": round(
                        float(group["grounding_score"].mean()),
                        3,
                    ),
                    "mean_human_review_rate": round(
                        float(group["human_review_rate"].mean()),
                        3,
                    ),
                }
            )
        by_configuration = pd.DataFrame(aggregates)
        best = (
            str(
                by_configuration.sort_values(
                    "mean_score",
                    ascending=False,
                ).iloc[0]["configuration"]
            )
            if not by_configuration.empty
            else ""
        )
        return BatchRunResult(
            run_id=batch_run_id,
            by_document=by_document,
            by_configuration=by_configuration,
            best_configuration=best,
        )
