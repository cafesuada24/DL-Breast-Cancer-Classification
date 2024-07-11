from typing import Union, Optional
import sys

import argparse
from pathlib import Path

MODEL_FILENAME = r"model.keras"
STD_SCALER_FILENAME = r"standard_scaler.pkl"


def check_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.train and args.model:
        parser.error("train mode cannot be combined with --model flag")


def validate_path(
    path: Union[str, Path],
    err_msg: Optional[str] = None,
    *,
    file_type: Optional[str] = None,
    dir: bool = False
) -> bool:
    if isinstance(path, str):
        path = Path(path)
    if (not dir and (not path.is_file() or path.suffix != file_type)) or (
        dir and not path.is_dir()
    ):
        print(err_msg, file=sys.stderr)
        return False
    return True
