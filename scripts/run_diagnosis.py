import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fhad_daic_au_cleaner.config import DATA_ROOT, OUTPUT_ROOT
from fhad_daic_au_cleaner.diagnose import run_diagnosis
from fhad_daic_au_cleaner.utils import get_logger

logger = get_logger("run_diagnosis")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose why cleaned session CSVs are empty or missing."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT,
                        help="Root directory containing raw session folders.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT,
                        help="Root directory containing cleaned output CSVs.")
    parser.add_argument("--save-report", type=Path, default=None,
                        help="Optional path to save the diagnosis report as CSV.")
    args = parser.parse_args()

    df = run_diagnosis(args.data_root, args.output_root)

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save_report, index=False)
        logger.info("Diagnosis report saved to %s", args.save_report)


if __name__ == "__main__":
    main()
