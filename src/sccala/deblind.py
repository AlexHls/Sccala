import argparse

from itsdangerous import URLSafeSerializer
import pandas as pd
import numpy as np


def main(args):

    file = pd.read_csv(args.file)

    s = URLSafeSerializer(args.key)

    if args.level == "hard":
        norm = np.genfromtxt(args.file.replace(".csv", ".key"), dtype=str)
        norm = s.loads(str(norm))
    else:
        norm = 1.0

    h0 = []
    for i in range(len(file["H0"])):
        h0.append(s.loads(file["H0"][i]))
    h0 = np.array(h0)

    if args.preserve_file:
        return h0 * norm
    else:
        file["H0"] = h0 * norm
        file.to_csv(args.file)

    return file


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("file", help="File to be decrypted.")
    parser.add_argument("key", help="Decryption key.")
    parser.add_argument(
        "-l",
        "--level",
        choices=["soft", "hard"],
        default="soft",
        help="Level of decryption. Soft only decrypts to float, hard also renormalizes values. Default: soft",
    )
    parser.add_argument(
        "--preserve_file",
        action="store_true",
        help="If flag is given, decripted values will be returned, but not written to the given file.",
    )

    args = parser.parse_args()

    print(main(args))

    return


if __name__ == "__main__":
    cli()
