"""banking77 as a verifiers taskset, one row per task, for distillation.

    # hosted SFT: the teacher answers through this taskset, its reasoning
    # dropped before recording; the student learns the answer only
    prime train configs/distill-banking77.toml

Rewards: `agrees` (the reply's intent equals the dataset's label). With a
teacher, it measures the teacher; prime-rl can reject rollouts by it.
"""

from collections.abc import Iterator

import verifiers.v1 as vf
from datasets import load_dataset

from lmfn_verifiers import AnswerOnlyTask, AnswerOnlyTaskConfig, OneTurnEnv, RowTaskData, turns

from .program import DATASET, classify


class Banking77TaskConfig(AnswerOnlyTaskConfig):
    pass


class Banking77Config(vf.TasksetConfig):
    split: str = "train"
    limit: int | None = None
    shuffle_seed: int | None = 0
    """The splits are sorted by label; shuffled (fixed seed) so any first N
    rows are a fair sample. None keeps the file order."""
    task: Banking77TaskConfig = Banking77TaskConfig()


class Banking77Task(AnswerOnlyTask[Banking77TaskConfig]):
    @vf.reward(weight=1.0)
    async def agrees(self, trace: vf.Trace) -> float:
        recorded = turns(trace)
        if not recorded or "refusal" in recorded[0]:
            return 0.0
        return float(recorded[0]["outputs"].get("answer") == self.data.info["label"])


class Banking77Env(OneTurnEnv):
    pass


class Banking77Taskset(vf.Taskset[Banking77Task, Banking77Config]):
    def load(self) -> Iterator[Banking77Task]:
        rows = load_dataset(DATASET, split=self.config.split)
        if self.config.shuffle_seed is not None:
            rows = rows.shuffle(seed=self.config.shuffle_seed)
        for i, row in enumerate(rows):
            if self.config.limit is not None and i >= self.config.limit:
                return
            yield Banking77Task(RowTaskData(idx=i, prompt=None,
                                            info={"inputs": {"text": row["text"]},
                                                  "label": row["label_text"]}),
                                self.config.task)
