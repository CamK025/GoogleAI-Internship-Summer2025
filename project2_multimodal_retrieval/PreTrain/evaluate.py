from utils_bbox import bbox_iou
from PIL import Image
import math


def compute_point_recall(pred_bbox, gt_points, img_w, img_h):
    count_inside = 0
    for pt in gt_points:
        x = int(pt["x"] * img_w)
        y = int(pt["y"] * img_h)
        if (pred_bbox[0] <= x <= pred_bbox[2]) and (pred_bbox[1] <= y <= pred_bbox[3]):
            count_inside += 1
    total_points = len(gt_points)
    if total_points == 0:
        return 0.0
    return count_inside / total_points

def evaluate_metrics(
    retriever, 
    samples, 
    k_list=(1,5), 
    iou_thr=0.5
):
    agg = {f"P@{k}":0 for k in k_list}
    agg |= {f"R@{k}":0 for k in k_list}
    agg.update({"MAP":0, "NDCG":0, "PointCoverage@1":0, "PointCoverage@5":0})

    for item in samples:
        query_caption = item["caption"]
        if len(query_caption) > 77:
            query_caption = query_caption[:77]
        results = retriever.query(query_caption, k=max(k_list))
        
        # IoU 
        hit_pos = None
        hits = []
        for r, rec in enumerate(results, 1):
            rec_bbox = rec["bbox"]
            if isinstance(rec_bbox, tuple):
                rec_bbox = list(rec_bbox)
            elif isinstance(rec_bbox, list) and isinstance(rec_bbox[0], (tuple, list)):
                rec_bbox = [x for pt in rec_bbox for x in pt]
            iou = bbox_iou(rec["bbox"], _get_gt_bbox(item))
            is_hit = iou >= iou_thr
            if is_hit and hit_pos is None:
                hit_pos = r
            hits.append(is_hit)
        
        for k in k_list:
            topk_hits = hits[:k]
            agg[f"P@{k}"] += sum(topk_hits) / k
            agg[f"R@{k}"] += 1.0 if any(topk_hits) else 0.0
        
        # MAP
        if hit_pos is not None:
            agg["MAP"] += 1.0 / hit_pos
        
        # NDCG
        if hit_pos is not None:
            agg["NDCG"] += 1.0 / (math.log2(hit_pos + 1))
        
        # Point Coverage
        img = Image.open(item["image_path"])
        img_w, img_h = img.size
        for k in [1,5]:
            coverages = [
                compute_point_recall(results[r]["bbox"], item["traces"], img_w, img_h)
                for r in range(min(k, len(results)))
            ]
            agg[f"PointCoverage@{k}"] += max(coverages) if coverages else 0.0

    N = len(samples)
    for k in k_list:
        agg[f"P@{k}"] /= N
        agg[f"R@{k}"] /= N
    agg["MAP"] /= N
    agg["NDCG"] /= N
    agg["PointCoverage@1"] /= N
    agg["PointCoverage@5"] /= N
    return agg

def _get_gt_bbox(item):
    from utils_bbox import traces_to_bbox
    traces = item.get("traces", [])
    img = Image.open(item["image_path"])
    img_w, img_h = img.size
    bbox = traces_to_bbox(traces, img_w, img_h)
    
    # flatten
    if isinstance(bbox, tuple):
        bbox = list(bbox)
    elif isinstance(bbox, list) and isinstance(bbox[0], (tuple, list)):
        bbox = [x for pt in bbox for x in pt]
    
    return bbox