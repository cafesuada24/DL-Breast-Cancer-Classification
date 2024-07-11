import os
import uuid
from pathlib import Path
import argparse

from lib import check_args, validate_path

current_dir = os.getcwd()
DEFAULT_MODEL_DIRECTORY = Path(current_dir + r"\model")
DEFAULT_PREDICTED_OUPUT_DIRECTORY = Path(current_dir + r"\predicted")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="BreastCancerClassification.py",
        description="The model with ability to predict the Breast Cancer Type",
    )
    parser.add_argument(
        "-t", "--train", action="store_true", help="switch to train mode"
    )
    parser.add_argument(
        "-f",
        "--file",
        metavar="CSV_FILE",
        type=str,
        nargs="?",
        help="predict the Breast Cancer type from dataset",
        required=True,
    )
    parser.add_argument(
        "-o", "--output", metavar="OUTPUT", type=str, nargs="?", help="provide output"
    )
    parser.add_argument(
        "-m",
        "--model",
        metavar="DIRECTORY",
        type=Path,
        nargs="?",
        help="the model directory",
        const=DEFAULT_MODEL_DIRECTORY,
    )

    return parser


def main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()

    check_args(args, parser)
    if args.train:
        dataset = Path(args.file)
        output = Path(args.output) if args.output else DEFAULT_MODEL_DIRECTORY
        if not output.exists():
            os.mkdir(output)
        if not validate_path(
            dataset,
            "Dataset file does not exists or is not a CSV file",
            file_type=".csv",
        ) and validate_path(output, "The ouput is not a directory", dir=True):
            return

        from core.train import train

        train(dataset, output)
    else:

        input = Path(args.file)
        model = Path(args.model) if args.model else DEFAULT_MODEL_DIRECTORY
        output = (
            Path(args.output)
            if args.output
            else DEFAULT_PREDICTED_OUPUT_DIRECTORY.joinpath(str(uuid.uuid4()) + ".csv")
        )
        if not (
            validate_path(
                input,
                "Data to predict does not exists or is not a CSV file",
                file_type=".csv",
            )
            and validate_path(
                model,
                "The directory containing model does not exists or is not a directory",
                dir=True,
            )
        ):
            return

        from core.predict import predict

        predict(model, input, output)
    print("DONE")


if __name__ == "__main__":
    parser = get_parser()
    main(parser)
