"""
Integration test for the Disributed_frame.communication logic (gather + scatter).
"""
import os
import sys
import copy

import torch
import torch.nn as nn

# Make project root importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.Buliding_Units.Model_frames import Disributed_frame


class TinyBlock(nn.Module):
    """Single-weight linear block for simple distributed syncing."""
    def __init__(self, weight: float):
        super().__init__()
        self.lin = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.lin.weight.fill_(weight)

    def forward(self, x):
        return self.lin(x)


class TinyModel(nn.Module):
    """Model composed of a sequence of TinyBlock blocks."""
    def __init__(self, weights):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(w) for w in weights])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


def test_distributed_frame_communicate_with_delay():
    # Base weights for two-block model
    base_weights = [1.0, 2.0]
    model = TinyModel(base_weights)

    # Hyperparameters: CPU mode, no delay, one block-step
    H = {
        "step_size": 0.1,
        "batch_size": 1,
        "cuda_core": -1,
        "K": 1,
        "communication_delay": 0,
    }
    # Initialize distributed frame
    frame = Disributed_frame(model, H)
    frame.init_distributed_models()

    # All workers initially match central model (but separate storage)
    for name, (worker_model, _) in frame.distributed_models.items():
        for p_c, p_w in zip(frame.center_model.parameters(), worker_model.parameters()):
            # Ensure no shared storage
            assert p_c.data_ptr() != p_w.data_ptr(), f"Shared storage in {name}"
            # Ensure equal values
            assert torch.allclose(p_c, p_w), f"Values differ at init for {name}"

    # Mutate only block_0 of worker_0
    w0_model, _ = frame.distributed_models.get("block_0")
    with torch.no_grad():
        for p in w0_model.blocks[0].parameters():
            p.add_(0.5)

    # Before communication, central model should be unchanged
    assert torch.allclose(frame.center_model.blocks[0].lin.weight, torch.tensor([[1.0]]))
    assert torch.allclose(frame.center_model.blocks[1].lin.weight, torch.tensor([[2.0]]))

    # Perform gather + scatter
    frame.communicate_withDelay()

    # After gather: central block_0 updated, block_1 unchanged
    assert torch.allclose(frame.center_model.blocks[0].lin.weight, torch.tensor([[1.5]])), \
        "Central block_0 did not receive update"
    assert torch.allclose(frame.center_model.blocks[1].lin.weight, torch.tensor([[2.0]])), \
        "Central block_1 was incorrectly modified"

    # After scatter: all workers match the central model for both blocks
    for name, (worker_model, _) in frame.distributed_models.items():
        assert torch.allclose(worker_model.blocks[0].lin.weight,
                              frame.center_model.blocks[0].lin.weight), \
            f"Worker {name} block_0 did not sync back"
        assert torch.allclose(worker_model.blocks[1].lin.weight,
                              frame.center_model.blocks[1].lin.weight), \
            f"Worker {name} block_1 did not sync back"