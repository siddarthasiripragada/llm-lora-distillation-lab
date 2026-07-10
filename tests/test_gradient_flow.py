import pytest

torch = pytest.importorskip("torch")

from sidsearch_lora_lab.training.lora import is_lora_parameter


class TinyTrainable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(3, 3)
        self.lora_A = torch.nn.Linear(3, 2, bias=False)
        self.lora_B = torch.nn.Linear(2, 1, bias=False)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        return self.lora_B(self.lora_A(self.base(x))).sum()


def test_lora_gradient_nonzero_and_base_absent():
    model = TinyTrainable()
    loss = model(torch.ones(1, 3))
    loss.backward()
    assert any(parameter.grad is not None and torch.any(parameter.grad != 0) for name, parameter in model.named_parameters() if is_lora_parameter(name))
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if name.startswith("base."))
