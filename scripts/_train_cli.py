"""Shared M1 CLI declarations, without training imports or lock acquisition."""

import argparse
from pathlib import Path


DEVICES = ("cpu", "cuda", "mps")
LR_SCHEDULES = ("constant", "cosine")
MASK_REGIMES = ("R", "T", "S")
MASKED_INNERVATION = ("uniform", "real")


#: TrainConfig fields a flag can override, with their flag and type.
_TRAIN_FLAGS = (
    ("--epochs", "epochs", int),
    ("--hidden", "hidden", int),
    ("--layers", "layers", int),
    ("--lr", "lr", float),
    ("--encoder-dropout", "encoder_dropout", float),
    ("--decoder-dropout", "decoder_dropout", float),
    ("--steps-per-epoch", "steps_per_epoch", int),
    ("--patience", "patience", int),
    ("--gap-fraction", "gap_fraction", float),
    ("--aux-weight", "aux_weight", float),
    ("--eval-every", "eval_every", int),
    ("--warmup-epochs", "warmup_epochs", int),
    ("--pair-batch", "pair_batch", int),
    ("--decoder-product-rank", "decoder_product_rank", int),
    ("--feature-noise", "feature_noise", float),
    ("--type-negatives", "type_negatives", int),
    ("--train-negatives", "train_negatives", int),
    ("--ema-decay", "ema_decay", float),
    ("--type-profile-rank", "type_profile_rank", int),
)

#: Boolean TrainConfig fields, each a --name/--no-name flag pair.
_BOOLEAN_FLAGS = (
    "sibling_features", "type_pair_features", "pair_geometry", "sibling_pooling",
    "nt_edges", "holdout_probe", "type_pair_shares", "annotation_groups", "nt_pair_features",
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one M1 model from processed graph, feature, and F1-val artifacts.",
        epilog="Each training value resolves as the flag if given, else the "
        "--config file's [train] value, else the TrainConfig default.",
    )
    parser.add_argument(
        "--config", type=Path,
        help="a train/configs/<code>.toml; makes --epochs, --processed-dir, "
        "--suite and --suites-dir optional",
    )
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--suites-dir", type=Path)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True, choices=DEVICES)
    parser.add_argument(
        "--test-run", action="store_true",
        help="suffix the run ID with -test so it never shares a run directory "
        "with a real run",
    )
    for flag, name, value_type in _TRAIN_FLAGS:
        parser.add_argument(flag, dest=name, type=value_type)
    parser.add_argument("--lr-schedule", choices=LR_SCHEDULES)
    for name in _BOOLEAN_FLAGS:
        parser.add_argument(
            "--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
        )
    parser.add_argument("--aggregation", choices=("sum", "degree_norm", "attention"))
    parser.add_argument("--edge-weight", choices=("fraction", "count"))
    parser.add_argument("--gap-regimes", nargs="+", choices=MASK_REGIMES)
    parser.add_argument("--masked-innervation", choices=MASKED_INNERVATION)
    parser.add_argument(
        "--title", help="what this run tests; recorded in the run manifest",
    )
    return parser


def _seed(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not a non-negative integer seed")
    return value


class _DistinctSeeds(argparse.Action):
    """Distinct seeds, so a sweep never runs one twice."""

    def __call__(self, parser, namespace, values, option_string=None):
        if len(set(values)) != len(values):
            parser.error("--seeds requires distinct integer seeds")
        setattr(namespace, self.dest, values)


class _RefusedSingleSeed(argparse.Action):
    """A seed is the sweep's to pass to each run, so ``--seed`` is refused here."""

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error("--seed is managed by the sweep; use --seeds instead")


def build_sweep_argparser() -> argparse.ArgumentParser:
    """The training parser with a list of seeds in place of one."""
    parser = build_argparser()
    parser.description = "Run one M1 training configuration at several seeds."
    # --seed's action is replaced below so the pair can be mutually exclusive.
    single = next(action for action in parser._actions if action.dest == "seed")
    parser._optionals._remove_action(single)
    del parser._option_string_actions["--seed"]
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, action=_RefusedSingleSeed)
    seeds.add_argument(
        "--seeds", nargs="+", type=_seed, metavar="SEED", action=_DistinctSeeds,
        help="distinct seeds, one run each; defaults to the --config file's calibration_seeds",
    )
    return parser
