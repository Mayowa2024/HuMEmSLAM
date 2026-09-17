import numpy as np

from slam.cognitive_math_model import CognitiveMathModel
from slam.types import KeyframeRecord, SceneRecord, StaticObject, TextAnchor


def scene(values, confidence=1.0):
    return SceneRecord(
        embedding=np.asarray(values, dtype=np.float32),
        confidence=confidence,
        label="test",
    )


def landmark(name="sign", x=0.5, text="B214"):
    return StaticObject(
        class_name=name,
        seg_conf=1.0,
        x_centroid=x,
        y_centroid=0.5,
        area=0.1,
        texts=[TextAnchor(text=text, conf=1.0)],
    )


def keyframe(identifier, scene_record, objects=None):
    return KeyframeRecord(
        keyframe_id=identifier,
        timestamp=float(identifier),
        scene=scene_record,
        static_objects=list(objects or []),
    )


def test_identical_semantics_score_one():
    model = CognitiveMathModel()
    query_scene = scene([1.0, 0.0, 0.0])
    candidate_scene = scene([1.0, 0.0, 0.0])
    query = keyframe(1, query_scene, [landmark()])
    candidate = keyframe(2, candidate_scene, [landmark()])

    score = model.unified_score(
        query, candidate, [query_scene], [candidate_scene]
    )

    assert np.isclose(score, 1.0)


def test_score_breakdown_exposes_each_semantic_layer():
    model = CognitiveMathModel()
    query_scene = scene([1.0, 0.0, 0.0])
    candidate_scene = scene([1.0, 0.0, 0.0])
    query = keyframe(1, query_scene, [landmark()])
    candidate = keyframe(2, candidate_scene, [landmark()])

    result = model.score_breakdown(
        query, candidate, [query_scene], [candidate_scene]
    )

    assert set(result) == {
        "scene_score", "object_score", "text_score", "text_evidence",
        "unified_score",
    }
    assert np.isclose(result["scene_score"], 1.0)
    assert np.isclose(result["object_score"], 1.0)
    assert np.isclose(result["text_score"], 1.0)
    assert np.isclose(result["unified_score"], 1.0)


def test_different_semantics_score_zero():
    model = CognitiveMathModel()
    query_scene = scene([1.0, 0.0, 0.0])
    candidate_scene = scene([0.0, 1.0, 0.0])
    query = keyframe(1, query_scene, [landmark("sign", text="B214")])
    candidate = keyframe(2, candidate_scene, [landmark("clock", text="EXIT")])

    score = model.unified_score(
        query, candidate, [query_scene], [candidate_scene]
    )

    assert 0.0 < score < 0.1


def test_classifier_label_confidence_does_not_suppress_embedding_score():
    model = CognitiveMathModel()
    query_scene = scene([1.0, 0.0], confidence=0.1)
    candidate_scene = scene([1.0, 0.0], confidence=0.1)
    query = keyframe(1, query_scene)
    candidate = keyframe(2, candidate_scene)

    score = model.unified_score(
        query, candidate, [query_scene], [candidate_scene]
    )

    assert np.isclose(score, 1.0)
    assert score >= model.semantic_threshold


def test_scene_only_match_is_renormalized_over_available_evidence():
    model = CognitiveMathModel()
    query_scene = scene([1.0, 0.0])
    candidate_scene = scene([1.0, 0.0])
    query = keyframe(1, query_scene)
    candidate = keyframe(2, candidate_scene)

    score = model.unified_score(
        query, candidate, [query_scene], [candidate_scene]
    )

    assert np.isclose(score, 1.0)
    assert score >= model.semantic_threshold


def test_every_valid_layer_ablation_combination():
    query_scene = scene([1.0, 0.0])
    candidate_scene = scene([1.0, 0.0])
    query = keyframe(1, query_scene, [landmark()])
    candidate = keyframe(2, candidate_scene, [landmark()])

    combinations = [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ]
    for use_scene, use_object, use_text in combinations:
        model = CognitiveMathModel(
            use_scene=use_scene,
            use_object=use_object,
            use_text=use_text,
        )
        score = model.unified_score(
            query, candidate, [query_scene], [candidate_scene]
        )
        assert np.isclose(score, 1.0)


def test_disabling_every_layer_is_rejected():
    try:
        CognitiveMathModel(
            use_scene=False,
            use_object=False,
            use_text=False,
        )
    except ValueError:
        return
    raise AssertionError("all-disabled configuration should fail")


def test_candidate_missing_query_supported_text_keeps_query_weight():
    model = CognitiveMathModel()
    query_scene = scene([1.0, 0.0])
    candidate_scene = scene([1.0, 0.0])
    query = keyframe(1, query_scene, [landmark(text="MAYOWA CAFE 214")])
    candidate = keyframe(
        2,
        candidate_scene,
        [StaticObject("sign", 1.0, 0.5, 0.5, 0.1, texts=[])],
    )

    score = model.unified_score(
        query, candidate, [query_scene], [candidate_scene]
    )



    assert np.isclose(score, 0.6)


def test_candidate_missing_query_supported_objects_keeps_query_weight():
    model = CognitiveMathModel(use_text=False)
    query_scene = scene([1.0, 0.0])
    candidate_scene = scene([1.0, 0.0])
    query = keyframe(1, query_scene, [landmark()])
    candidate = keyframe(2, candidate_scene, [])

    result = model.score_breakdown(
        query, candidate, [query_scene], [candidate_scene]
    )

    assert result["object_score"] == 0.0
    assert np.isclose(result["unified_score"], 0.5)


def test_query_without_objects_disables_object_layer_for_all_candidates():
    model = CognitiveMathModel(use_text=False)
    query_scene = scene([1.0, 0.0])
    candidate_scene = scene([1.0, 0.0])
    query = keyframe(1, query_scene, [])
    candidate = keyframe(2, candidate_scene, [landmark()])

    result = model.score_breakdown(
        query, candidate, [query_scene], [candidate_scene]
    )

    assert result["object_score"] is None
    assert np.isclose(result["unified_score"], 1.0)


def test_object_appearance_embedding_separates_same_geometry_objects():
    model = CognitiveMathModel(
        use_scene=False, use_text=False, object_appearance_weight=0.5
    )
    query_object = landmark()
    matching_object = landmark()
    different_object = landmark()
    query_object.appearance_embedding = np.asarray([1.0, 0.0])
    matching_object.appearance_embedding = np.asarray([1.0, 0.0])
    different_object.appearance_embedding = np.asarray([-1.0, 0.0])
    query = keyframe(1, scene([1.0]), [query_object])

    matching_score = model.object_similarity(
        query, keyframe(2, scene([1.0]), [matching_object])
    )
    different_score = model.object_similarity(
        query, keyframe(3, scene([1.0]), [different_object])
    )

    assert np.isclose(matching_score, 1.0)
    assert np.isclose(different_score, 0.5)


def test_garbage_query_ocr_does_not_activate_text_denominator():
    model = CognitiveMathModel()
    query_scene = scene([1.0, 0.0])
    candidate_scene = scene([1.0, 0.0])
    query_object = landmark(text="]")
    candidate_object = landmark(text="MAYOWA CAFE 214")
    query = keyframe(1, query_scene, [query_object])
    candidate = keyframe(2, candidate_scene, [candidate_object])

    result = model.score_breakdown(
        query, candidate, [query_scene], [candidate_scene]
    )

    assert result["text_evidence"] == 0.0
    assert np.isclose(result["unified_score"], 1.0)


def test_disabled_layers_do_not_enter_query_denominator():
    model = CognitiveMathModel(use_scene=True, use_object=False, use_text=False)
    query_scene = scene([1.0, 0.0])
    candidate_scene = scene([1.0, 0.0])
    query = keyframe(1, query_scene, [landmark()])
    candidate = keyframe(2, candidate_scene, [])

    assert np.isclose(model.unified_score(
        query, candidate, [query_scene], [candidate_scene]
    ), 1.0)


def test_changed_storefront_text_is_a_bounded_penalty():
    model = CognitiveMathModel(text_conflict_floor=0.25)
    query_scene = scene([1.0, 0.0])
    candidate_scene = scene([1.0, 0.0])
    query = keyframe(1, query_scene, [landmark(text="MAYOWA CAFE 214")])
    candidate = keyframe(2, candidate_scene, [landmark(text="NEW SHOP 800")])

    score = model.unified_score(
        query, candidate, [query_scene], [candidate_scene]
    )

    assert 0.6 < score < 1.0


def test_partial_ocr_match_and_distinctiveness():
    model = CognitiveMathModel()

    assert model.string_similarity(
        "MAYOWA CAFE", "MAYOWA CAFE LONDON"
    ) > 0.5
    assert (
        model.text_distinctiveness("MAYOWA CAFE 214")
        > model.text_distinctiveness("OPEN")
    )


def test_temporal_scene_sequence_uses_recency_weights():
    model = CognitiveMathModel()
    aligned = scene([1.0, 0.0])
    different = scene([0.0, 1.0])

    recent_match = model.scene_similarity(
        [aligned, different, different],
        [aligned, aligned, aligned],
    )
    old_match = model.scene_similarity(
        [different, different, aligned],
        [aligned, aligned, aligned],
    )

    assert np.isclose(recent_match, 0.5)
    assert np.isclose(old_match, 0.2)
    assert recent_match > old_match


def test_object_spatial_similarity_decays_with_displacement():
    model = CognitiveMathModel(sigma_mask=0.25)
    query = landmark(x=0.2)
    close = landmark(x=0.21)
    far = landmark(x=0.9)

    assert model.spatial_similarity(query, close) > 0.99
    assert model.spatial_similarity(query, far) < 0.05


def test_different_object_classes_are_gated_out():
    model = CognitiveMathModel()

    assert model.object_match_score(
        landmark(name="traffic sign"),
        landmark(name="building"),
    ) == 0.0


def test_object_confidence_uses_geometric_mean_not_product():
    model = CognitiveMathModel()
    query = landmark()
    candidate = landmark()
    query.seg_conf = 0.25
    candidate.seg_conf = 0.64

    assert np.isclose(model.object_match_score(query, candidate), 0.4)


def test_object_assignment_is_one_to_one_within_exact_class():
    model = CognitiveMathModel(sigma_mask=0.1)
    query = keyframe(1, scene([1.0]), [
        landmark(name="traffic_sign", x=0.2),
        landmark(name="traffic_sign", x=0.8),
    ])
    candidate = keyframe(2, scene([1.0]), [
        landmark(name="traffic_sign", x=0.2),
    ])



    assert np.isclose(model.object_similarity(query, candidate), 0.5)


def test_generic_object_classes_have_less_influence():
    model = CognitiveMathModel(object_class_weights={"pole": 0.25, "traffic_sign": 1.0})
    query = keyframe(1, scene([1.0]), [
        landmark(name="pole", x=0.2),
        landmark(name="traffic_sign", x=0.8),
    ])
    candidate = keyframe(2, scene([1.0]), [
        landmark(name="pole", x=0.2),
    ])

    assert np.isclose(model.object_similarity(query, candidate), 0.2)


def test_relative_layout_penalises_reversed_object_arrangement():
    model = CognitiveMathModel(
        relative_layout_weight=0.25, relative_layout_sigma=0.1
    )
    query = keyframe(1, scene([1.0]), [
        landmark(name="building", x=0.2),
        landmark(name="traffic_sign", x=0.8),
    ])
    aligned = keyframe(2, scene([1.0]), [
        landmark(name="building", x=0.2),
        landmark(name="traffic_sign", x=0.8),
    ])
    reversed_layout = keyframe(3, scene([1.0]), [
        landmark(name="building", x=0.8),
        landmark(name="traffic_sign", x=0.2),
    ])

    assert model.object_similarity(query, aligned) > model.object_similarity(
        query, reversed_layout
    )


def test_repeated_local_observation_increases_evidence_reliability():
    model = CognitiveMathModel(persistence_gain=0.25)
    query_object = landmark()
    candidate_object = landmark()
    query_object.seg_conf = 0.6
    candidate_object.seg_conf = 0.6
    repeated_query = landmark()
    repeated_candidate = landmark()
    repeated_query.seg_conf = 0.6
    repeated_candidate.seg_conf = 0.6
    repeated_query.observation_count = 3
    repeated_candidate.observation_count = 3

    assert model.object_match_score(
        repeated_query, repeated_candidate
    ) > model.object_match_score(query_object, candidate_object)


def test_text_requires_geometrically_consistent_supporting_object():
    model = CognitiveMathModel(text_geom_threshold=0.6)
    query = landmark(x=0.1, text="CAFE 214")
    far_candidate = landmark(x=0.9, text="CAFE 214")

    assert not model.text_gate(query, far_candidate)
    assert model.object_text_similarity(query, far_candidate) == 0.0


def test_string_similarity_normalises_case_and_punctuation():
    model = CognitiveMathModel()

    assert np.isclose(
        model.string_similarity("Mayowa-Cafe 214!", "mayowacafe 214"),
        1.0,
    )


def test_recovery_policy_requires_both_conditions_and_strict_threshold():
    model = CognitiveMathModel(semantic_threshold=0.76)

    assert model.should_trigger_semantic_recovery(10, 30, 0.90)
    assert not model.should_trigger_semantic_recovery(40, 30, 0.90)
    assert not model.should_trigger_semantic_recovery(10, 30, 0.70)
    assert not model.should_trigger_semantic_recovery(10, 30, 0.76)


def test_candidate_selection_returns_highest_scoring_record():
    model = CognitiveMathModel(use_object=False, use_text=False)
    query_scene = scene([1.0, 0.0])
    query = keyframe(1, query_scene)
    wrong = keyframe(2, scene([0.0, 1.0]))
    correct = keyframe(3, scene([1.0, 0.0]))

    selected, score_value = model.select_best_candidate(
        query,
        [query_scene],
        [wrong, correct],
        [[wrong.scene], [correct.scene]],
    )

    assert selected is correct
    assert np.isclose(score_value, 1.0)


def test_no_available_evidence_returns_zero():
    model = CognitiveMathModel()
    query = keyframe(1, scene_record=scene([1.0, 0.0]))
    candidate = keyframe(2, scene_record=scene([1.0, 0.0]))

    score_value = model.unified_score(query, candidate, [], [])

    assert score_value == 0.0


def test_scene_support_mode_does_not_penalise_missing_support():
    model = CognitiveMathModel(fusion_mode="scene_support")
    query_scene = scene([1.0, 0.0])
    candidate_scene = scene([0.8, 0.6])
    query = keyframe(1, query_scene)
    candidate = keyframe(2, candidate_scene)

    breakdown = model.score_breakdown(
        query, candidate, [query_scene], [candidate_scene]
    )

    assert np.isclose(breakdown["unified_score"], breakdown["scene_score"])
