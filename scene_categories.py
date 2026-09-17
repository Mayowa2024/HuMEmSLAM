"""Road-localisation groupings for selected Places365 output classes."""

from typing import Dict, Mapping





PLACES365_GROUP_BY_ID = {
    4: "urban_road",
    79: "urban_road",
    112: "urban_road",
    125: "urban_road",
    270: "urban_road",
    273: "urban_road",
    301: "urban_road",
    307: "urban_road",
    308: "urban_road",
    319: "urban_road",

    8: "residential",
    49: "residential",
    74: "residential",
    87: "residential",
    107: "residential",
    127: "residential",
    183: "residential",
    184: "residential",
    193: "residential",
    220: "residential",
    221: "residential",
    231: "residential",
    283: "residential",
    296: "residential",

    31: "commercial",
    47: "commercial",
    50: "commercial",
    72: "commercial",
    99: "commercial",
    114: "commercial",
    119: "commercial",
    128: "commercial",
    139: "commercial",
    147: "commercial",
    158: "commercial",
    161: "commercial",
    162: "commercial",
    172: "commercial",
    181: "commercial",
    198: "commercial",
    223: "commercial",
    261: "commercial",
    262: "commercial",
    267: "commercial",
    284: "commercial",
    286: "commercial",
    300: "commercial",
    302: "commercial",
    321: "commercial",
    335: "commercial",

    156: "parking",
    157: "parking",
    255: "parking",
    256: "parking",
    257: "parking",

    0: "major_transport",
    2: "major_transport",
    66: "major_transport",
    67: "urban_road",
    71: "major_transport",
    129: "major_transport",
    171: "major_transport",
    174: "major_transport",
    175: "major_transport",
    207: "major_transport",
    216: "major_transport",
    266: "major_transport",
    278: "major_transport",
    293: "major_transport",
    320: "major_transport",
    336: "major_transport",
    337: "major_transport",
    347: "major_transport",

    18: "industrial",
    23: "industrial",
    28: "industrial",
    103: "industrial",
    133: "industrial",
    136: "industrial",
    144: "industrial",
    169: "industrial",
    170: "industrial",
    192: "industrial",
    199: "industrial",
    206: "industrial",
    247: "industrial",
    282: "industrial",
    298: "industrial",

    118: "rural_road",
    138: "rural_road",
    142: "rural_road",
    152: "rural_road",
    173: "rural_road",
    249: "rural_road",
    258: "rural_road",
    287: "rural_road",
    338: "rural_road",
    348: "rural_road",
    349: "rural_road",
    359: "rural_road",
    360: "rural_road",

    30: "natural", 36: "natural", 48: "natural", 62: "natural",
    73: "natural", 76: "natural", 78: "natural", 81: "natural",
    94: "natural", 97: "natural", 104: "natural", 110: "natural",
    111: "natural", 113: "natural", 116: "natural", 117: "natural",
    140: "natural", 141: "natural", 145: "natural", 150: "natural",
    151: "natural", 163: "natural", 164: "natural", 167: "natural",
    180: "natural", 186: "natural", 187: "natural", 190: "natural",
    194: "natural", 204: "natural", 205: "natural", 209: "natural",
    224: "natural", 232: "natural", 233: "natural", 234: "natural",
    243: "natural", 254: "natural", 265: "natural", 271: "natural",
    279: "natural", 288: "natural", 289: "natural", 304: "natural",
    305: "natural", 306: "natural", 309: "natural", 323: "natural",
    324: "natural", 341: "natural", 342: "natural", 344: "natural",
    350: "natural", 355: "natural", 356: "natural", 357: "natural",

    5: "restricted_special", 15: "restricted_special",
    16: "restricted_special", 17: "restricted_special",
    24: "restricted_special", 42: "restricted_special",
    68: "restricted_special", 77: "restricted_special",
    86: "restricted_special", 108: "restricted_special",
    132: "restricted_special", 149: "restricted_special",
    168: "restricted_special", 178: "restricted_special",
    188: "restricted_special", 189: "restricted_special",
    225: "restricted_special", 230: "restricted_special",
    251: "restricted_special", 252: "restricted_special",
    275: "restricted_special", 276: "restricted_special",
    310: "restricted_special", 312: "restricted_special",
    313: "restricted_special", 314: "restricted_special",
    327: "restricted_special", 330: "restricted_special",
    351: "restricted_special", 354: "restricted_special",
}


COMPATIBLE_GROUPS = {
    "urban_road": {"commercial", "residential", "parking"},
    "residential": {"urban_road", "rural_road", "parking"},
    "commercial": {"urban_road", "parking"},
    "parking": {"commercial", "residential", "urban_road"},
    "major_transport": {"urban_road", "industrial"},
    "industrial": {"major_transport", "urban_road"},
    "rural_road": {"natural", "residential"},
    "natural": {"rural_road"},
    "restricted_special": set(),
    "other": set(),
}


def group_places365_probabilities(
    probabilities, top_k: int = 3
) -> Dict[str, float]:
    """Aggregate the top-k Places365 probabilities into broad scene groups."""
    import numpy as np

    values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if values.size == 0 or top_k <= 0:
        return {}
    count = min(int(top_k), values.size)
    indices = np.argpartition(values, -count)[-count:]
    grouped: Dict[str, float] = {}
    for class_id in indices:
        group = PLACES365_GROUP_BY_ID.get(int(class_id), "other")
        grouped[group] = grouped.get(group, 0.0) + max(0.0, float(values[class_id]))
    total = sum(grouped.values())
    if total <= 0.0:
        return {}
    return {name: value / total for name, value in grouped.items()}


def category_compatibility(
    query: Mapping[str, float], candidate: Mapping[str, float]
) -> float:
    """Expected compatibility between two grouped top-k distributions."""
    if not query or not candidate:
        return 0.0
    score = 0.0
    for q_group, q_probability in query.items():
        for c_group, c_probability in candidate.items():
            if q_group == c_group and q_group != "other":
                compatibility = 1.0
            elif c_group in COMPATIBLE_GROUPS.get(q_group, set()):
                compatibility = 0.6
            elif "other" in (q_group, c_group):
                compatibility = 0.3
            else:
                compatibility = 0.0
            score += float(q_probability) * float(c_probability) * compatibility
    return max(0.0, min(1.0, score))
