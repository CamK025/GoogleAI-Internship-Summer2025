from typing import List, Dict
from PIL import Image, ImageDraw


# ---------------- Normalized trace → pixel points ----------------
def traces_to_bbox(traces: List[Dict], img_w: int, img_h: int) -> List[tuple]:
    """
    Convert normalized traces to a list of pixel point coordinates.
    """
    if not traces:
        return []
    points = []
    for pt in traces:
        x = int(pt["x"] * img_w)
        y = int(pt["y"] * img_h)
        points.append((x, y))
    return points


# ---------------- IoU calculation ----------------
def bbox_iou(boxA: List[int], boxB: List[int]) -> float:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    """
    boxA = list(boxA)
    boxB = list(boxB)
    if isinstance(boxA[0], (tuple, list)):
        boxA = [x for pt in boxA for x in pt]
    if isinstance(boxB[0], (tuple, list)):
        boxB = [x for pt in boxB for x in pt]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union = areaA + areaB - inter
    return inter / union if union else 0.0


# ---------------- ROI cropping ----------------
def crop_bbox(img: Image.Image, bbox: List[int]) -> Image.Image:
    """
    Return the cropped PIL image region.
    """
    return img.crop(tuple(bbox))


# ---------------- Draw trace points for visualization ----------------
def draw_traces(img: Image.Image, traces: List[Dict], color="green", radius=2):
    """
    Draw all trace points on the original image.
    """
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size
    for pt in traces:
        x = int(pt["x"] * img_w)
        y = int(pt["y"] * img_h)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return img

def draw_bbox(img: Image.Image, bbox: List[int], color="red", width=3):
    """
    Draw a bounding box on the original image (modifies the image in place).
    """
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline=color, width=width)
    return img
