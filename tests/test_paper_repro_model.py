import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))
from paper_repro_model import SyncNet, multitask_loss


def test_cnn_and_cnn_lstm_emit_three_finite_targets():
    # Catches a broken 4x96x2 feature interface or dropped output head.
    x = torch.randn(5, 4, 96, 2)
    for temporal in (False, True):
        net = SyncNet(temporal=temporal)
        out = net(x)
        assert out["offset"].shape == (5,)
        assert out["cfo_sc"].shape == (5,)
        assert out["sector_logits"].shape == (5, 3)
        assert all(torch.isfinite(v).all() for v in out.values())


def test_multitask_loss_backpropagates_on_local_gpu():
    # Catches CPU-only execution or a detached loss path.
    assert torch.cuda.is_available()
    net = SyncNet(temporal=True).to("cuda:0")
    x = torch.randn(4, 4, 96, 2, device="cuda:0")
    out = net(x)
    loss = multitask_loss(out, {
        "offset": torch.tensor([20., 30., 40., 50.], device="cuda:0"),
        "cfo_sc": torch.tensor([.8, 2., 3.5, .8], device="cuda:0"),
        "nid2": torch.tensor([0, 1, 2, 0], device="cuda:0"),
    })
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in net.parameters())


def test_cfo_output_bound_can_cover_jitter_without_changing_default():
    custom = SyncNet(temporal=True, max_cfo_sc=3.8)
    default = SyncNet(temporal=True)
    for net in (custom, default):
        for parameter in net.parameters():
            parameter.data.zero_()
    x = torch.zeros(2, 4, 96, 2)
    assert torch.allclose(custom(x)["cfo_sc"], torch.full((2,), 1.9))
    assert torch.allclose(default(x)["cfo_sc"], torch.full((2,), 1.75))
