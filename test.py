import argparse
import logging

from parselabs.dataset import build_integrity_report, print_integrity_report


def parse_args():
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Validate parselabs configuration and extracted outputs")
    parser.add_argument("--profile", "-p", help="Profile name to validate (omit to validate all configured profiles)")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args()
    profile_names = [args.profile] if args.profile else None
    report = build_integrity_report(profile_names=profile_names)
    print_integrity_report(report)


if __name__ == "__main__":
    main()
