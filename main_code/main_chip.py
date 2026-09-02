import sys
import importlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

SUBCOMMANDS = {
    "preprocess": ("chip.preprocessing", "Curve preprocessing + kinetic features + optional LSTM-AE outlier detection"),
    "cross-dataset": ("chip.cross_dataset_training", "Cross-dataset leave-one-folder-out training"),
    "ablation6": ("chip.ablation6", "Per-fold outlier-filter x model ablation"),
    "saliency": ("chip.saliency", "Saliency + latent-feature-mapping plots"),
    "embedding-analysis": ("chip.embedding_analysis", "Per-model t-SNE embedding grid"),
    "confidence-shift": ("chip.confidence_shift", "In-distribution vs LOCO softmax confidence"),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python main_chip.py <subcommand> [args...]\n")
        print("Subcommands:")
        for name, (_, desc) in SUBCOMMANDS.items():
            print(f"  {name:<14} {desc}")
        sys.exit(0)

    subcommand = sys.argv[1]
    if subcommand not in SUBCOMMANDS:
        print(f"Unknown subcommand '{subcommand}'. Choices: {', '.join(SUBCOMMANDS)}")
        sys.exit(1)

    module_name, _ = SUBCOMMANDS[subcommand]
    module = importlib.import_module(module_name)
    module.main(sys.argv[2:])


if __name__ == "__main__":
    main()
