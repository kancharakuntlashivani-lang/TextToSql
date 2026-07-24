from __future__ import annotations
import argparse
from src.datasets_manager import load_all
from src.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the two-dataset Text-to-SQL experiment.')
    parser.add_argument('--limit', type=int, default=10, help='Questions per dataset.')
    parser.add_argument('--dataset', choices=['Both datasets', 'BIRD Mini-Dev', 'Spider'], default='Both datasets')
    args = parser.parse_args()

    frame = load_all(auto_download=True)
    if args.dataset == 'Both datasets':
        frame = frame.groupby('dataset', group_keys=False).head(args.limit)
    else:
        frame = frame[frame['dataset'] == args.dataset].head(args.limit)
    output = run_experiment(frame, len(frame), provider='OpenAI')
    print(output['summary'].to_string(index=False))


if __name__ == '__main__':
    main()
