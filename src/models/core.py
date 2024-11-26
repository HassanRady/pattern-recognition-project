from pathlib import Path
from typing import Any, Tuple, Callable, Optional

import optuna

from src.logger import (
    get_console_logger,
)

LOGGER = get_console_logger(logger_name=__name__)

def run_hpo(
    n_trials: int,
    hpo_path: Path,
    study_name: str,
    objective: Callable[
        ...,
        float,
    ],
    n_jobs: Optional[int] = -1,
) -> Tuple[int, dict[str, Any], float, dict[str, Any]]:
    hpo_path.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=f"sqlite:///{hpo_path}/{study_name}.sqlite3",
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
    )
    return (
        study.best_trial.number,
        study.best_trial.params,
        study.best_trial.value,
        study.best_trial.user_attrs,
    )