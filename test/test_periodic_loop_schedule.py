from slam.human_slam_node import (
    HuMemSLAMNode,
    apply_weak_scene_consensus_policy,
    apply_scene_only_score_policy,
    candidates_above_threshold,
    preselect_for_expensive_rerank,
    should_run_candidate_retrieval,
)
from slam.types import KeyframeRecord, SceneRecord, StaticObject
import numpy as np


def test_only_candidates_strictly_above_threshold_are_submitted():
    ranked = [
        ("high", 0.90, {}),
        ("just_over", 0.760001, {}),
        ("boundary", 0.76, {}),
        ("low", 0.40, {}),
    ]

    selected = candidates_above_threshold(ranked, 0.76, 5)

    assert [item[0] for item in selected] == ["high", "just_over"]


def test_threshold_filter_respects_response_limit():
    ranked = [(str(index), 0.9 - index * 0.01, {}) for index in range(6)]

    assert len(candidates_above_threshold(ranked, 0.76, 3)) == 3


def test_expensive_rerank_uses_only_eigenplaces_prefix():
    candidates = list(range(25))
    sequences = [f"scene-{index}" for index in candidates]

    selected_candidates, selected_sequences = preselect_for_expensive_rerank(
        candidates, sequences, 10
    )

    assert selected_candidates == list(range(10))
    assert selected_sequences == [f"scene-{index}" for index in range(10)]


def test_object_only_ranking_is_not_scene_prefiltered():
    candidates = list(range(12))
    sequences = list(reversed(candidates))

    assert preselect_for_expensive_rerank(
        candidates, sequences, 5, enabled=False
    ) == (candidates, sequences)


def test_neighborhoods_are_built_only_after_scene_preselection():
    node = object.__new__(HuMemSLAMNode)
    node.map_memory = [
        KeyframeRecord(
            keyframe_id=index,
            timestamp=float(index),
            scene=SceneRecord(embedding=[float(index)], confidence=1.0),
        )
        for index in range(30)
    ]
    node.candidate_min_separation = 0
    node.candidate_top_k = 25
    node.candidate_rerank_top_k = 10
    node.use_scene = True

    class Matcher:
        @staticmethod
        def scene_similarity(_query, candidate):
            return float(candidate[0].embedding[0])

    node.matcher = Matcher()
    calls = []

    def neighborhood(centre, _prior):
        calls.append(centre.keyframe_id)
        return centre

    node._semantic_neighborhood = neighborhood
    candidates, _ = node.build_candidate_inputs([node.map_memory[-1].scene])

    assert len(candidates) == 10
    assert len(calls) == 10
    assert calls == list(range(29, 19, -1))


def test_selected_candidate_neighborhoods_are_cached_between_queries():
    node = object.__new__(HuMemSLAMNode)
    node.map_memory = [
        KeyframeRecord(
            keyframe_id=index,
            timestamp=float(index),
            scene=SceneRecord(embedding=[float(index)], confidence=1.0),
        )
        for index in range(12)
    ]
    node._semantic_neighborhood_cache = {}
    node.candidate_min_separation = 0
    node.candidate_top_k = 5
    node.candidate_rerank_top_k = 5
    node.use_scene = True

    class Matcher:
        @staticmethod
        def scene_similarity(_query, candidate):
            return float(candidate[0].embedding[0])

    node.matcher = Matcher()
    calls = []

    def neighborhood(centre, _prior):
        calls.append(centre.keyframe_id)
        return centre

    node._semantic_neighborhood = neighborhood
    query = [node.map_memory[-1].scene]
    node.build_candidate_inputs(query)
    node.build_candidate_inputs(query)

    assert len(calls) == 5
    assert len(node._semantic_neighborhood_cache) == 5


def test_object_embedding_retrieval_requires_same_class_and_prefers_match():
    node = object.__new__(HuMemSLAMNode)
    node.object_embedding_enabled = True
    node.object_class_weights = {"building": 0.7}
    query_object = StaticObject(
        "building", 1.0, 0.5, 0.5, 0.2,
        appearance_embedding=np.asarray([1.0, 0.0]),
    )
    match_object = StaticObject(
        "building", 1.0, 0.5, 0.5, 0.2,
        appearance_embedding=np.asarray([1.0, 0.0]),
    )
    wrong_class = StaticObject(
        "fence", 1.0, 0.5, 0.5, 0.2,
        appearance_embedding=np.asarray([1.0, 0.0]),
    )
    query = KeyframeRecord(1, 1.0, SceneRecord(None), [query_object])
    match = KeyframeRecord(2, 2.0, SceneRecord(None), [match_object])
    wrong = KeyframeRecord(3, 3.0, SceneRecord(None), [wrong_class])

    assert np.isclose(node._object_embedding_retrieval_score(query, match), 1.0)
    assert node._object_embedding_retrieval_score(query, wrong) == 0.0


def test_dino_retrieval_requires_two_query_tracks_in_same_history_cluster():
    node = object.__new__(HuMemSLAMNode)
    node.dino_object_enabled = True
    node.dino_candidate_classes = {"building", "traffic_sign"}
    node.dino_single_landmark_classes = {"traffic_sign"}
    node.dino_similarity_threshold = 0.8
    node.dino_similarity_margin = 0.05
    node.dino_search_neighbors_per_track = 5
    node.dino_min_consensus_tracks = 2
    node.dino_candidate_cluster_width = 30
    node._last_perception_timings = {}
    node.map_memory = [
        KeyframeRecord(i, float(i), SceneRecord(None), source_frame_id=i * 10)
        for i in range(20)
    ]
    node._landmark_index = {
        "building": [
            {"embedding": np.asarray([1.0, 0.0]), "memory_index": 5,
             "track_id": 100, "source_frame_id": 50},
            {"embedding": np.asarray([0.0, 1.0]), "memory_index": 15,
             "track_id": 101, "source_frame_id": 150},
        ],
        "traffic_sign": [
            {"embedding": np.asarray([0.0, 1.0]), "memory_index": 6,
             "track_id": 200, "source_frame_id": 60},
            {"embedding": np.asarray([1.0, 0.0]), "memory_index": 16,
             "track_id": 201, "source_frame_id": 160},
        ],
    }
    building = StaticObject(
        "building", 1.0, 0.5, 0.5, 0.2,
        appearance_embedding=np.asarray([1.0, 0.0]),
        appearance_model="dinov2_vits14_dense", landmark_track_id=1,
    )
    sign = StaticObject(
        "traffic_sign", 1.0, 0.5, 0.5, 0.05,
        appearance_embedding=np.asarray([0.0, 1.0]),
        appearance_model="dinov2_vits14_dense", landmark_track_id=2,
    )
    query = KeyframeRecord(99, 99.0, SceneRecord(None), [building, sign])

    assert node._dino_candidate_indices(query, set(range(20))) == [5]


def test_candidate_specific_calibrated_threshold_is_respected():
    ranked = [
        ("object_ablation", 0.63, {"submission_threshold": 0.62}),
        ("full_system", 0.69, {"submission_threshold": 0.70}),
    ]

    selected = candidates_above_threshold(ranked, 0.70, 5)

    assert [item[0] for item in selected] == ["object_ablation"]


def test_ok_tracking_runs_only_on_configured_keyframe_period():
    assert should_run_candidate_retrieval(True, 10, 2, 5)
    assert not should_run_candidate_retrieval(True, 11, 2, 5)
    assert not should_run_candidate_retrieval(False, 10, 2, 5)


def test_weak_tracking_always_runs_recovery_retrieval():
    assert should_run_candidate_retrieval(False, 11, 3, 5)
    assert should_run_candidate_retrieval(False, 12, 4, 5)


def test_period_is_clamped_to_one():
    assert should_run_candidate_retrieval(True, 7, 2, 0)


def test_qualifying_scene_only_score_is_demoted_to_0701():
    breakdown = {
        "scene_score": 0.95,
        "object_score": None,
        "text_score": None,
    }

    effective, evidence = apply_scene_only_score_policy(
        breakdown, 0.95, 0.70, 0.701
    )

    assert effective == 0.701
    assert evidence == "weak_scene_only"
    assert candidates_above_threshold([("candidate", effective, {})], 0.70, 5)


def test_scene_only_score_below_threshold_is_not_promoted():
    breakdown = {
        "scene_score": 0.68,
        "object_score": None,
        "text_score": None,
    }

    effective, evidence = apply_scene_only_score_policy(
        breakdown, 0.68, 0.70, 0.701
    )

    assert effective == 0.68
    assert evidence == "weak_scene_only"


def test_multi_layer_score_is_not_changed():
    breakdown = {
        "scene_score": 0.80,
        "object_score": 0.72,
        "text_score": None,
    }

    effective, evidence = apply_scene_only_score_policy(
        breakdown, 0.76, 0.70, 0.701
    )

    assert effective == 0.76
    assert evidence == "strong_multi_layer"


def _scene_only(frame, score):
    return (
        KeyframeRecord(
            keyframe_id=frame, timestamp=0.0, scene=None,
            source_frame_id=frame,
        ),
        score,
        {
            "scene_score": score,
            "object_score": None,
            "text_score": None,
            "raw_unified_score": score,
            "effective_score": score,
            "evidence_label": "weak_scene_only",
        },
    )


def test_weak_scene_cluster_promotes_only_top_candidate_to_cap():
    ranked = [_scene_only(frame, score) for frame, score in (
        (58, 0.597), (61, 0.554), (56, 0.544), (50, 0.513), (48, 0.486)
    )]
    selected = apply_weak_scene_consensus_policy(
        ranked, 0.70, 0.701, raw_threshold=0.55,
        top_k=5, minimum_support=3, cluster_width_frames=30,
    )
    assert selected[0][1] == 0.701
    assert selected[0][2]["evidence_label"] == "weak_scene_consensus"
    assert selected[0][2]["scene_consensus_support"] == 5
    assert len(candidates_above_threshold(selected, 0.70, 5)) == 1


def test_weak_scene_without_cluster_is_not_promoted():
    ranked = [_scene_only(frame, score) for frame, score in (
        (58, 0.60), (200, 0.59), (400, 0.58), (600, 0.57), (800, 0.56)
    )]
    selected = apply_weak_scene_consensus_policy(
        ranked, 0.70, 0.701, raw_threshold=0.55,
        top_k=5, minimum_support=3, cluster_width_frames=30,
    )
    assert selected[0][1] == 0.60
    assert not candidates_above_threshold(selected, 0.70, 5)


def test_normal_above_threshold_candidate_is_not_relabelled():
    ranked = [_scene_only(58, 0.701), _scene_only(61, 0.65), _scene_only(56, 0.64)]
    selected = apply_weak_scene_consensus_policy(ranked, 0.70, 0.701)
    assert selected == ranked
