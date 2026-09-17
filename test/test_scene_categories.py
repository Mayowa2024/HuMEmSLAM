import numpy as np

from slam.cognitive_math_model import CognitiveMathModel
from slam.scene_categories import (
    category_compatibility,
    group_places365_probabilities,
)
from slam.types import SceneRecord


def test_places365_top_k_is_grouped_and_normalised():
    probabilities = np.zeros(365, dtype=np.float32)
    probabilities[319] = 0.5
    probabilities[283] = 0.3
    probabilities[67] = 0.2

    grouped = group_places365_probabilities(probabilities, top_k=3)

    assert np.isclose(sum(grouped.values()), 1.0)
    assert np.isclose(grouped["urban_road"], 0.7)
    assert np.isclose(grouped["residential"], 0.3)


def test_category_compatibility_orders_same_related_and_incompatible():
    same = category_compatibility({"urban_road": 1.0}, {"urban_road": 1.0})
    related = category_compatibility({"urban_road": 1.0}, {"residential": 1.0})
    incompatible = category_compatibility({"urban_road": 1.0}, {"natural": 1.0})
    assert same == 1.0
    assert related == 0.6
    assert incompatible == 0.0


def test_category_modulates_but_does_not_replace_embedding_similarity():
    model = CognitiveMathModel(
        use_object=False,
        use_text=False,
        scene_category_weight=0.1,
    )
    query = SceneRecord(
        embedding=np.array([1.0, 0.0]),
        category_distribution={"urban_road": 1.0},
    )
    same = SceneRecord(
        embedding=np.array([1.0, 0.0]),
        category_distribution={"urban_road": 1.0},
    )
    different = SceneRecord(
        embedding=np.array([1.0, 0.0]),
        category_distribution={"natural": 1.0},
    )

    assert np.isclose(model.scene_similarity([query], [same]), 1.0)
    assert 0.9 < model.scene_similarity([query], [different]) < 1.0


def test_scene_category_layer_can_be_ablated():
    model = CognitiveMathModel(
        use_object=False,
        use_text=False,
        use_scene_category=False,
    )
    query = SceneRecord(
        embedding=np.array([1.0, 0.0]),
        category_distribution={"urban_road": 1.0},
    )
    candidate = SceneRecord(
        embedding=np.array([1.0, 0.0]),
        category_distribution={"natural": 1.0},
    )
    assert np.isclose(model.scene_similarity([query], [candidate]), 1.0)
