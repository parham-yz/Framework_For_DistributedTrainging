"""Test isolation of distributed models and correctness of block communication."""

import copy
import os
import sys

import torch

# ------------------------------------------------------------------
# Make project package importable when tests are executed from repo root
# ------------------------------------------------------------------

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src import utils  # noqa: E402


# ------------------------------------------------------------------
# Tiny model with explicit block structure
# ------------------------------------------------------------------


class TinyBlock(torch.nn.Module):
    """Single‑weight linear block for easy inspection."""

    def __init__(self, weight: float):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.lin.weight.fill_(weight)

    def forward(self, x):  # noqa: D401
        return self.lin(x)


class TinyModel(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.blocks = torch.nn.ModuleList([TinyBlock(w) for w in weights])

    def forward(self, x):  # noqa: D401
        for blk in self.blocks:
            x = blk(x)
        return x


# ------------------------------------------------------------------
# Test case
# ------------------------------------------------------------------


def test_isolation_and_block_communication():
    torch.manual_seed(0)

    # Create central model with two distinct blocks
    centre = TinyModel([1.0, 2.0])

    # Build distributed copies (one per block)
    distributed_models = {}
    for i in range(len(centre.blocks)):
        copy_model = copy.deepcopy(centre)
        opt = torch.optim.SGD(copy_model.blocks[i].parameters(), lr=1.0)
        distributed_models[f"block_{i}"] = (copy_model, opt)

    # ------------------------------------------------------------------
    # 1. Isolation check – parameter storage must differ
    # ------------------------------------------------------------------
    ptrs_centre = [p.data_ptr() for p in centre.parameters()]

    for name_i, (model_i, _) in distributed_models.items():
        for p_c, p_i in zip(centre.parameters(), model_i.parameters()):
            assert p_c.data_ptr() != p_i.data_ptr(), (
                f"Parameter storage shared between centre and {name_i}"
            )

    # Ensure distributed models are mutually isolated as well
    names = list(distributed_models.keys())
    for idx_a in range(len(names)):
        for idx_b in range(idx_a + 1, len(names)):
            model_a = distributed_models[names[idx_a]][0]
            model_b = distributed_models[names[idx_b]][0]
            for pa, pb in zip(model_a.parameters(), model_b.parameters()):
                assert pa.data_ptr() != pb.data_ptr(), "Distributed models share storage"

    # ------------------------------------------------------------------
    # 2. Simulate local update on block_0 only
    # ------------------------------------------------------------------
    block0_model, _ = distributed_models["block_0"]
    with torch.no_grad():
        # Add 0.5 to *all* params of block 0 in that distributed model
        for p in block0_model.blocks[0].parameters():
            p.add_(0.5)

    # Centre model must be unchanged at this point
    w_c_before = centre.blocks[0].lin.weight.clone()
    assert torch.allclose(w_c_before, torch.tensor([[1.0]])), "Centre changed before communication"

    # Other distributed model (block_1) should remain unchanged
    block1_model, _ = distributed_models["block_1"]
    w1_before = block1_model.blocks[0].lin.weight.clone()
    assert torch.allclose(w1_before, torch.tensor([[1.0]])), "Unrelated distributed model mutated"

    # ------------------------------------------------------------------
    # 3. Communication: copy updated block back to centre
    # ------------------------------------------------------------------
    utils.copy_block(block0_model, centre, 0)

    # Centre must now match updated value 1.5
    assert torch.allclose(centre.blocks[0].lin.weight, torch.tensor([[1.5]])), (
        "Communication failed to update central block"
    )

    # Block 1 in centre must remain unchanged (value 2.0)
    assert torch.allclose(centre.blocks[1].lin.weight, torch.tensor([[2.0]])), (
        "Communication incorrectly modified untouched block"
    )

    # ------------------------------------------------------------------
    # 4. Broadcast back to distributed models (simulate second phase)
    # ------------------------------------------------------------------
    for name, (model, _) in distributed_models.items():
        utils.copy_model(centre, model, device=torch.device("cpu"))

    for name, (model, _) in distributed_models.items():
        assert torch.allclose(model.blocks[0].lin.weight, centre.blocks[0].lin.weight)
        assert torch.allclose(model.blocks[1].lin.weight, centre.blocks[1].lin.weight)
