"""Train an A-Z HOG/SVM classifier from the EMNIST Letters CSV split.

Download the EMNIST Letters CSV training file and pass its path here. The CSV
format is: label, then 784 grayscale pixel values. EMNIST labels are 1..26.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="EMNIST Letters training CSV")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("letter_svm.yml"))
    parser.add_argument("--limit", type=int, default=0, help="optional maximum rows (0 means all)")
    args = parser.parse_args()

    if not args.csv_path.is_file():
        parser.error(f"CSV file not found: {args.csv_path}")

    hog = cv2.HOGDescriptor((28, 28), (14, 14), (7, 7), (7, 7), 9)
    features: list[np.ndarray] = []
    labels: list[int] = []
    with args.csv_path.open("r", newline="") as source:
        reader = csv.reader(source)
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue
            try:
                label = int(row[0])
                pixels = np.asarray(row[1:], dtype=np.uint8)
            except ValueError:
                # Be tolerant of a header row.
                if row_number == 1:
                    continue
                raise
            if pixels.size != 784 or not 1 <= label <= 26:
                raise ValueError(f"Invalid EMNIST row {row_number}: expected label 1..26 and 784 pixels")
            image = pixels.reshape(28, 28)
            # EMNIST is stored rotated and mirrored relative to normal reading.
            image = cv2.flip(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 1)
            features.append(hog.compute(image).reshape(-1))
            labels.append(label)
            if args.limit and len(labels) >= args.limit:
                break
            if len(labels) % 10000 == 0:
                print(f"Loaded {len(labels)} examples...")

    if not labels:
        parser.error("No training examples found in CSV")

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(2.0)
    svm.train(x, cv2.ml.ROW_SAMPLE, y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    svm.save(str(args.output))
    print(f"Trained on {len(labels)} examples; saved model to {args.output}")
    print("For an accuracy estimate, use the separate EMNIST test split or validate on your own handwriting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
