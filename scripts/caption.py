import argparse
import asyncio
from io import BytesIO
import json
from pathlib import Path
import os
import re
import base64

from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from tqdm import tqdm


VLLM_BASE_URL = "http://127.0.0.1:1234/v1"
VLLM_MODEL_NAME = "qwen/qwen3-vl-30b"

EXTRACTION_PROMPT = """
Extract all people with and without safety helmet on with its bounding boxes on related image
Bounding box coordinates are in 0-1000 relative to image width and height.
x1,y1 is top-left, x2,y2 is bottom-right.
List each person once. At most 40 people. Output one JSON object and stop.

Output format:
{
    persons: [
        {
            "has_helmet": bool,
            "bbox": {
                "x1": float,
                "y1": float,
                "x2": float,
                "y2": float,
            }
        }
    ]
}
"""

VQA_MAX_SIZE = (512, 512)
BBOX_SCALE = 1000.0
TEMPERATURE = 0.0
MAX_TOKENS = 2048
MAX_PERSONS = 40
MAX_ATTEMPTS = 3

CLASSES_COLORS = {
    '0': (1.0, 0.0, 0.0), # нет шлема
    '1': (0.0, 0.0, 1.0) # есть шлем
}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "images"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


client = AsyncOpenAI(
    api_key="-",
    base_url=VLLM_BASE_URL
)

class BoundingBoxXYWH(BaseModel):
    """Bounding box in [x_min, y_min, width, height] format."""
    x1: float = Field(..., description="Top-left x, 0-1000")
    y1: float = Field(..., description="Top-left y, 0-1000")
    x2: float = Field(..., description="Bottom-right x, 0-1000")
    y2: float = Field(..., description="Bottom-right y, 0-1000")


class Person(BaseModel):
    has_helmet: bool = Field(..., description="Indicates if the person is wearing a safety helmet")
    bbox: BoundingBoxXYWH = Field(..., description="Bounding box")


class PersonsResponse(BaseModel):
    persons: list[Person] = Field(..., max_length=MAX_PERSONS)


async def inference_vqa(prompt: str, img_data_url: str) -> PersonsResponse | None:
    for _ in range(MAX_ATTEMPTS):
        try:
            completion = await client.chat.completions.create(
                model=VLLM_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": img_data_url
                                }
                            }
                        ]
                    }
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "hats-detection",
                        "schema": PersonsResponse.model_json_schema()
                    },
                },
            )
            content = completion.choices[0].message.content
            return PersonsResponse.model_validate_json(content)
        except Exception as e:
            print(f"Error: {e}")
            continue
    print(f"Error: failed to load response after {MAX_ATTEMPTS} attempts")
    return None


def _bbox_to_norm(x1, y1, x2, y2):
    x1 = max(0, min(1, x1 / BBOX_SCALE))
    y1 = max(0, min(1, y1 / BBOX_SCALE))
    x2 = max(0, min(1, x2 / BBOX_SCALE))
    y2 = max(0, min(1, y2 / BBOX_SCALE))

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    width = x2 - x1
    height = y2 - y1

    return center_x, center_y, width, height


def draw_masks(img_path: str, img_stem: str, persons: list[Person]):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = ARTIFACTS_DIR / f"{img_stem}_mask.png"

    img = mpimg.imread(img_path)
    img_h, img_w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(img_w / 100, img_h / 100), dpi=100)
    ax.imshow(img)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for person in persons:
        bbox = person.bbox
        helmet_color = CLASSES_COLORS['1' if person.has_helmet else '0']
        helmet_caption = 'Helmet' if person.has_helmet else 'No helmet'
        x1, y1 = bbox.x1 / BBOX_SCALE * img_w, bbox.y1 / BBOX_SCALE * img_h
        x2, y2 = bbox.x2 / BBOX_SCALE * img_w, bbox.y2 / BBOX_SCALE * img_h
        w, h = x2 - x1, y2 - y1
        ax.add_patch(Rectangle(
            (x1, y1), w, h,
            fill=True,
            facecolor=(*helmet_color, 0.2),
            edgecolor=(*helmet_color, 1.0),
            linewidth=2,
        ))
        ax.text(
            x1,
            y1,
            helmet_caption,
            color="white",
            fontsize=8,
            va="bottom",
            bbox={"facecolor": helmet_color, "edgecolor": "none", "alpha": 0.8, "pad": 1},
        )

    fig.savefig(mask_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

async def process_image(img_path: str, test: bool = False) -> None:
    if not os.path.exists(img_path):
        raise FileNotFoundError
    img_filename = os.path.basename(img_path)
    img_stem, _ = os.path.splitext(img_filename)

    annotation_path = img_path.with_name(f"{img_stem}.txt")
    raw_response_path = img_path.with_name(f"{img_stem}.json")
    if os.path.exists(annotation_path):
        return

    img = Image.open(img_path)
    img.thumbnail(VQA_MAX_SIZE, Image.Resampling.LANCZOS)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    b64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
    img_data_url = f"data:image/png;base64,{b64_img}"

    response = await inference_vqa(EXTRACTION_PROMPT, img_data_url)
    if response is None:
        return

    if test:
        draw_masks(img_path, img_stem, response.persons)
        return

    with open(raw_response_path, 'w', encoding='utf-8') as f:
        json.dump(response.model_dump(), f, indent=4)

    with open(annotation_path, 'w', encoding='utf-8') as f:
        for person in response.persons:
            class_id = '1' if person.has_helmet else '0'
            bbox = person.bbox
            f.write(f'{class_id} {" ".join(str(x) for x in _bbox_to_norm(bbox.x1, bbox.y1, bbox.x2, bbox.y2))}\n')


async def main():
    parser = argparse.ArgumentParser(description="Описание работы скрипта.")

    parser.add_argument("--test", action="store_true", help="Тестовый прогон с масками")
    args = parser.parse_args()

    # сортировка по номеру в названии
    img_paths = sorted(
        DATA_DIR.glob("*.png"),
        key=lambda p: int(re.search(r"\d+", p.stem).group()) if re.search(r"\d+", p.stem) else p.stem, 
    )

    if args.test:
        img_paths = img_paths[:15]

    for p in tqdm(img_paths, desc="Processing images"):
        await process_image(p, test=args.test)


if __name__ == "__main__":
    asyncio.run(main())
