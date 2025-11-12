\
import argparse
import os

# A simple runner that proxies to train or eval scripts for convenience.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    if parsed.mode == "train":
        os.system("python -m src.scripts.train_attention_model " + " ".join(parsed.args or []))
    else:
        os.system("python -m src.scripts.evaluate_attention_model " + " ".join(parsed.args or []))

if __name__ == "__main__":
    main()
