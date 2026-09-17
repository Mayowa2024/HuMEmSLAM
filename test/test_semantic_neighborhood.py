from slam.cognitive_math_model import CognitiveMathModel
from slam.human_slam_node import HuMemSLAMNode
from slam.types import KeyframeRecord, SceneRecord, StaticObject, TextAnchor


def record(identifier, objects):
    return KeyframeRecord(
        keyframe_id=identifier,
        timestamp=float(identifier),
        scene=SceneRecord(embedding=None),
        static_objects=objects,
        source_frame_id=identifier,
    )


def obj(name, confidence=1.0, x=0.5, text=""):
    return StaticObject(
        class_name=name,
        seg_conf=confidence,
        x_centroid=x,
        y_centroid=0.5,
        area=0.1,
        texts=[TextAnchor(text, 1.0)] if text else [],
    )


def node_stub():
    node = HuMemSLAMNode.__new__(HuMemSLAMNode)
    node.semantic_neighborhood_enabled = True
    node.semantic_neighborhood_weights = [1.0, 0.6, 0.3]
    node.semantic_neighborhood_merge_distance = 0.15
    node.matcher = CognitiveMathModel()
    return node


def test_neighbour_supplies_missing_object_with_decayed_confidence():
    centre = record(3, [])
    previous = record(2, [obj("traffic_sign", confidence=0.9)])

    result = node_stub()._semantic_neighborhood(centre, [previous])

    assert len(result.static_objects) == 1
    assert result.static_objects[0].class_name == "traffic_sign"
    assert abs(result.static_objects[0].seg_conf - 0.9 * 0.6 * 0.6) < 1e-9
    assert result.keyframe_id == centre.keyframe_id


def test_repeated_neighbour_object_is_merged_and_text_is_retained():
    centre = record(3, [obj("building", x=0.5)])
    previous = record(2, [obj("building", x=0.51, text="MAYOWA CAFE")])

    result = node_stub()._semantic_neighborhood(centre, [previous])

    assert len(result.static_objects) == 1
    assert [text.text for text in result.static_objects[0].texts] == ["MAYOWA CAFE"]
    assert result.static_objects[0].observation_count == 2


def test_neighbourhood_is_non_recursive_and_does_not_mutate_centre():
    centre = record(3, [])
    previous = record(2, [obj("pole")])

    result = node_stub()._semantic_neighborhood(centre, [previous])

    assert centre.static_objects == []
    assert result is not centre
