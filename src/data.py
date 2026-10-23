"""Memory-mapped packed-token dataset.

Training examples are random contiguous windows from one long uint16 array.
Both models use the same window sampler and the same seed/step, so they see
exactly the same tokens when trained with the same batch configuration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class PackedTokens:
    def __init__(self, path: str | Path, block_size: int) -> None:
        self.path = Path(path)
        self.block_size = int(block_size)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Run `python -m src.prepare_data` first."
            )
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self.data) <= self.block_size + 1:
            raise ValueError(
                f"{self.path} has {len(self.data)} tokens, too few for block_size={self.block_size}"
            )

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(
        self,
        batch_size: int,
        step: int,
        device: torch.device | str,
        seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic-by-(seed, step) random windows.

        Using a step-derived RNG rather than global RNG state makes runs
        reproducible across resumes and makes dense/fast comparison exact.
        """
        rng = np.random.default_rng((int(seed) * 1_000_003 + int(step)) & 0xFFFFFFFF)
        high = len(self.data) - self.block_size - 1
        offsets = rng.integers(0, high, size=batch_size, dtype=np.int64)

        x = np.empty((batch_size, self.block_size), dtype=np.int64)
        y = np.empty_like(x)
        for row, offset in enumerate(offsets):
            x[row] = self.data[offset : offset + self.block_size]
            y[row] = self.data[offset + 1 : offset + 1 + self.block_size]

        return (
            torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(y).to(device, non_blocking=True),
        )
