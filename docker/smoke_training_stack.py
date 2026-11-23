"""Exercise PyG sampling and the TensorBoard CLI during image build."""

from importlib.metadata import version
import subprocess
import sys

import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader


data = Data(
    x=torch.arange(8, dtype=torch.float32).reshape(4, 2),
    edge_index=torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]],
        dtype=torch.long,
    ),
)
loader = NeighborLoader(
    data,
    input_nodes=torch.tensor([0, 1], dtype=torch.long),
    num_neighbors=[2],
    batch_size=2,
    shuffle=False,
)
batch = next(iter(loader))

assert batch.batch_size == 2
assert batch.num_nodes >= batch.batch_size
assert batch.edge_index.numel() > 0
print(
    "PyG sampling smoke test passed: "
    f"torch={torch.__version__}, "
    f"torch-geometric={version('torch-geometric')}, "
    f"pyg-lib={version('pyg-lib')}"
)

subprocess.run([sys.executable, "-m", "tensorboard.main", "--version"], check=True)
print(f"TensorBoard CLI smoke test passed: setuptools={version('setuptools')}")
