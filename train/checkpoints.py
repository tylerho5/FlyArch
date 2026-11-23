"""Load a frozen expert using its manifest and the current feature schema."""

from __future__ import annotations

from pathlib import Path

import torch

from src.features import FeatureSchema

from .milestones import _read_manifest
from .model import M1, MODEL_KNOBS

__all__ = ["load_expert"]


def load_expert(run_dir: str | Path, schema: FeatureSchema) -> M1:
    run_dir = Path(run_dir)
    manifest = _read_manifest(run_dir)
    expected = manifest["schema_digest"]
    if expected == "none":
        raise ValueError(f"{run_dir}: manifest has no feature schema digest")
    if expected != schema.digest():
        raise ValueError(
            f"{run_dir}: schema digest {schema.digest()} does not match manifest {expected}"
        )
    config = manifest["config"]
    model = M1(
        schema,
        hidden=int(config.get("hidden", 128)),
        layers=int(config.get("layers", 2)),
        encoder_dropout=float(config.get("encoder_dropout", 0.0)),
        decoder_dropout=float(config.get("decoder_dropout", 0.0)),
        sibling_features=bool(config.get("sibling_features", False)),
        layer_norm=bool(config.get("layer_norm", False)),
        residual=bool(config.get("residual", False)),
        branch_norm=bool(config.get("branch_norm", False)),
        decoder_product_rank=int(config.get("decoder_product_rank", 0)),
        **{name: type(off)(config.get(name, off)) for name, off in MODEL_KNOBS.items()},
    )
    try:
        checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
            raise ValueError("checkpoint has no model state dictionary")
        state = checkpoint["model"]
        model.load_state_dict(state)
    except Exception as exc:
        raise ValueError(f"{run_dir}: best.pt is unreadable or incompatible: {exc}") from exc
    model.eval()
    return model
