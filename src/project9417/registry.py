from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TaskType = Literal["regression", "classification"]
SourceType = Literal["kaggle", "uci"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    task_type: TaskType
    source_type: SourceType
    source_id: str
    target_column: str
    target_aliases: tuple[str, ...]
    primary_metric: str
    group_column: str | None = None
    group_aliases: tuple[str, ...] = ()
    drop_columns: tuple[str, ...] = ()
    drop_aliases: tuple[str, ...] = ()
    notes: str = ""
    interpretability_candidate: bool = False
    scaling_candidate: bool = False
    preferred_id_columns: tuple[str, ...] = field(default_factory=tuple)


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "job_salary": DatasetSpec(
        name="job_salary",
        display_name="Job Salary Prediction",
        task_type="regression",
        source_type="kaggle",
        source_id="nalisha/job-salary-prediction-dataset",
        target_column="salary",
        target_aliases=("salary", "Salary", "target", "Salary($)", "estimated_salary"),
        primary_metric="rmse",
        notes="Large mixed-type dataset used for the scaling analysis.",
        scaling_candidate=True,
        preferred_id_columns=("job_id", "id"),
    ),
    "student_exam": DatasetSpec(
        name="student_exam",
        display_name="Student Exam Performance",
        task_type="regression",
        source_type="kaggle",
        source_id="grandmaster07/student-exam-performance-dataset-analysis",
        target_column="Exam_Score",
        target_aliases=("Exam_Score", "exam_score", "overall_score", "Overall_Score"),
        primary_metric="rmse",
        preferred_id_columns=("Student_ID", "student_id", "id"),
    ),
    "appendicitis": DatasetSpec(
        name="appendicitis",
        display_name="Regensburg Pediatric Appendicitis",
        task_type="classification",
        source_type="uci",
        source_id="938",
        target_column="Diagnosis",
        target_aliases=("Diagnosis", "diagnosis", "appendicitis", "Appendicitis"),
        primary_metric="accuracy",
        interpretability_candidate=True,
        preferred_id_columns=("Unnamed: 0", "patient_id", "id"),
        notes="Uses the tabular UCI dataset only; no image data is consumed.",
    ),
    "parkinsons": DatasetSpec(
        name="parkinsons",
        display_name="Parkinsons Telemonitoring",
        task_type="regression",
        source_type="uci",
        source_id="189",
        target_column="total_UPDRS",
        target_aliases=("total_UPDRS", "Total_UPDRS"),
        primary_metric="rmse",
        group_column="subject#",
        group_aliases=("subject#", "subject", "Subject"),
        drop_columns=("motor_UPDRS",),
    ),
    "iris": DatasetSpec(
        name="iris",
        display_name="Iris",
        task_type="classification",
        source_type="uci",
        source_id="53",
        target_column="class",
        target_aliases=("class", "Class", "species", "Species"),
        primary_metric="accuracy",
    ),
}


ALL_DATASETS = tuple(DATASET_REGISTRY.keys())
DEFAULT_MODELS = ("xrfm", "xgboost", "random_forest")
