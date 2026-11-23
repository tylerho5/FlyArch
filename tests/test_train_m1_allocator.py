import pytest

pytest.importorskip("torch")


@pytest.mark.parametrize("device, index", [("cuda", 0), ("cuda:1", 1)])
def test_cuda_allocator_uses_current_free_memory_minus_margin(monkeypatch, device, index):
    import torch
    from scripts import train_m1

    calls = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (3 * 2**30, 8 * 2**30))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    # torch refuses a bare "cuda" here; the cap must name a device index.
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, device: calls.append((fraction, device)))

    train_m1._cap_cuda_allocator(device)

    assert calls == [((3 * 2**30 - 512 * 2**20) / (8 * 2**30), index)]


def test_non_cuda_device_does_not_touch_cuda_allocator(monkeypatch):
    import torch
    from scripts import train_m1

    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda *_: pytest.fail("CPU/MPS must not inspect CUDA memory"))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda *_: pytest.fail("CPU/MPS must not change the CUDA allocator"))

    train_m1._cap_cuda_allocator("cpu")
    train_m1._cap_cuda_allocator("mps")


def test_cuda_allocator_refuses_memory_below_margin(monkeypatch):
    import torch
    from scripts import train_m1

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1, 8 * 2**30))
    with pytest.raises(ValueError, match="insufficient free memory"):
        train_m1._cap_cuda_allocator("cuda")
