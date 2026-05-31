"""Resume support for benchmark episode JSONL files."""

import json
import os
from typing import Any, Dict, Optional, Set, Tuple

from core.metrics import EpisodeMetrics


class ResumeManager:
    """Load completed episode records for resume/skip behavior."""

    def __init__(self, resume_jsonl: Optional[str], resume_skip: str = "all"):
        self.resume_jsonl = resume_jsonl
        self.resume_skip = resume_skip
        self.episode_records: Dict[Tuple[int, int, int], EpisodeMetrics] = {}
        self.episode_keys: Set[Tuple[int, int, int]] = set()

    @staticmethod
    def episode_key(level: int, point_id: int, episode_id: int) -> Tuple[int, int, int]:
        return int(level), int(point_id), int(episode_id)

    @staticmethod
    def record_to_episode_metrics(record: Dict[str, Any]) -> EpisodeMetrics:
        kwargs = {}
        for field_name in EpisodeMetrics.__dataclass_fields__:
            if field_name in ("trajectory", "drone_trajectory"):
                continue
            if field_name in record:
                kwargs[field_name] = record[field_name]
        return EpisodeMetrics(**kwargs)

    def load(self) -> Tuple[Dict[Tuple[int, int, int], EpisodeMetrics], Set[Tuple[int, int, int]]]:
        self.episode_records = {}
        self.episode_keys = set()
        if not self.resume_jsonl:
            return self.episode_records, self.episode_keys
        if not os.path.exists(self.resume_jsonl):
            raise FileNotFoundError(f"Resume JSONL not found: {self.resume_jsonl}")

        with open(self.resume_jsonl, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON in resume file {self.resume_jsonl}:{line_no}: {e}"
                    ) from e
                metrics = self.record_to_episode_metrics(record)
                keep_record = self.resume_skip == "all" or metrics.success
                if not keep_record:
                    continue
                key = self.episode_key(metrics.level, metrics.point_id, metrics.episode_id)
                self.episode_records[key] = metrics

        self.episode_keys = set(self.episode_records.keys())
        return self.episode_records, self.episode_keys

    def get(self, level: int, point_id: int, episode_id: int) -> Optional[EpisodeMetrics]:
        return self.episode_records.get(self.episode_key(level, point_id, episode_id))
