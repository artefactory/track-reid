POSSIBLE_CLASSES = [0.0, 1.0]
MAPPING_CLASSES = {0.0: "shop_item", 1.0: "personal_item"}

INPUT_POSITIONS = {
    "object_id": 4,
    "category": 5,
    "bbox": [0, 1, 2, 3],
    "confidence": 6,
}

OUTPUT_POSITIONS = {
    "frame_id": 0,
    "object_id": 1,
    "category": 2,
    "bbox": [3, 4, 5, 6],
    "confidence": 7,
    "mean_confidence": 8,
    "tracker_id": 9,
}
