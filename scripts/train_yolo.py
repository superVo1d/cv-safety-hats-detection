import torch
from pathlib import Path
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR = BASE_DIR / "datasets" / "hardhat"


device = 0 if torch.cuda.is_available() else "cpu"

model = YOLO("yolo26n.pt")

results = model.train(
    data=str(OUTPUT_DIR / "data.yaml"),
    epochs=100,
    imgsz=512,
    device=device,
    project=str(ARTIFACTS_DIR / "yolo"),
    name="train",
    exist_ok=True,
    plots=True,
)
