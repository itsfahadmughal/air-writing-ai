# Air Writing AI

Write in the air using a webcam. **Air Writing AI** tracks your hand, draws when you pinch your thumb and index finger, and can attempt to recognize separated block letters and words.

Built with Python, OpenCV, MediaPipe, and NumPy.

## Features

- Real-time hand tracking with MediaPipe's 21 hand landmarks
- Index fingertip cursor with pinch-to-write and release-to-lift controls
- Smoothed strokes and separate pen-up movements
- Live hand skeleton with highlighted thumb and index fingertips
- Blurred camera-preview background to make the hand easier to see
- Optional A–Z recognition with an OpenCV HOG + SVM classifier
- Experimental recognition of separated block words using stroke grouping

## Demo

Add a screenshot or a link to your project video here when you publish it.

## Requirements

- Python 3.10 or newer
- A webcam
- macOS, Windows, or Linux with a graphical desktop for OpenCV preview windows

## Setup

Clone the repository and enter the project folder:

```bash
git clone https://github.com/<your-username>/air-writing-ai.git
cd air-writing-ai
```

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell
python -m pip install -r requirements.txt
```

Start the app:

```bash
python main.py
```

If your default webcam is not device 0, pass its index:

```bash
python main.py --camera 1
```

The first launch with MediaPipe Tasks downloads the hand landmark model (`hand_landmarker.task`) next to `main.py` if it is not present.

## How to use

1. Move your hand so the green cursor reaches the desired position on the board.
2. Pinch your thumb and index finger to start drawing. The cursor turns red.
3. Release the pinch to lift the pen and move to another stroke or letter.
4. Press `R` to recognize your writing when the optional letter model is installed.

| Key | Action |
| --- | --- |
| `C` | Clear the board |
| `R` | Recognize the writing |
| `Q` or `Esc` | Quit |

## Optional letter and word recognition

Drawing works without a recognition model. To enable recognition, obtain the **EMNIST Letters training CSV** (26 labels, numbered 1–26) and train the included HOG + linear SVM classifier:

```bash
python train_classifier.py data/emnist-letters/emnist-letters-train.csv
```

The trainer writes `letter_svm.yml` in the project folder. Start or restart the app and press `R` after writing. A quick training smoke run can use `--limit 1000`, but that small sample is not suitable for useful recognition.

The current recognizer groups nearby pen strokes into character shapes and uses larger horizontal gaps as spaces. Write clear, separated **block letters**, leaving small gaps between letters and larger gaps between words. This is an experimental baseline trained on isolated EMNIST letters; air-written characters may be misclassified. It does not reliably recognize connected cursive writing. Better word recognition needs word-level air-writing examples and a sequence OCR model.

## How it works

1. OpenCV reads and mirrors webcam frames.
2. Frames are converted from BGR to RGB for MediaPipe hand tracking.
3. The index fingertip controls the cursor; thumb-to-index distance relative to palm width detects a pinch.
4. Pinched movement is smoothed and saved as a stroke. Releasing the pinch starts a new stroke on the next pinch.
5. The optional recognizer groups strokes and classifies letter images with HOG features and an SVM.

## Project files

```text
main.py              Webcam app and recognition
train_classifier.py  Train the optional A–Z classifier
requirements.txt     Python dependencies
```

## License

Add a license before redistributing this project. If you add a pretrained model or dataset, check and follow its license and attribution requirements.
