import json
import copy


categories_21_openworld = [
    "person", "car", "trashbin", "road",
    "sidewalk", "crosswalk", "building", "house", "fence", "tree",
    "grass", "bench", "sign", "traffic_light", "swimming_pool"
]

object_captions_openworld = ". ".join(categories_21_openworld) + "."

projection_openworld = {i: cat for i, cat in enumerate(categories_21_openworld)}
projection_reverse_openworld = {cat: idx for idx, cat in projection_openworld.items()}