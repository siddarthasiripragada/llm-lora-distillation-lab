import pytest

torch = pytest.importorskip("torch")

from sidsearch_lora_lab.training.lora import assert_only_lora_trainable, parameter_report


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.lora_A = torch.nn.Linear(2, 1, bias=False)
        self.lora_B = torch.nn.Linear(1, 2, bias=False)
        for parameter in self.base.parameters():
            parameter.requires_grad = False


def test_parameter_report_counts_lora_trainable():
    model = TinyModel()
    report = parameter_report(model)
    assert report["trainable_parameters"] == 4
    assert report["unexpected_trainable_parameters"] == []


def test_assert_only_lora_trainable_passes_for_lora_only():
    assert assert_only_lora_trainable(TinyModel())["trainable_parameters"] > 0


def test_assert_only_lora_trainable_rejects_base_trainable():
    model = TinyModel()
    model.base.weight.requires_grad = True
    try:
        assert_only_lora_trainable(model)
    except RuntimeError as exc:
        assert "Unexpected non-LoRA" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
