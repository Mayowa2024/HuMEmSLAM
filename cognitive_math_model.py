import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from slam.types import KeyframeRecord, SceneRecord, StaticObject
from slam.scene_categories import category_compatibility


class CognitiveMathModel:
    """
    Hierarchical semantic-contextual matching model.

    Layers:
        1. Scene sequence similarity:
           compares recent scene records only.

        2. Static object layout similarity:
           compares segmented static objects in the current query keyframe
           against objects in a candidate keyframe.

        3. Object-grounded text similarity:
           compares OCR text only when object class matches and object
           spatial similarity is high.

    Output:
        Lambda(Q, C_i): semantic-contextual match score.
    """

    def __init__(
        self,
        w_scene: float = 0.3,
        w_object: float = 0.3,
        w_text: float = 0.4,
        sigma_mask: float = 0.25,
        lambda_area: float = 1.0,
        text_geom_threshold: float = 0.6,
        semantic_threshold: float = 0.70,
        use_scene: bool = True,
        use_object: bool = True,
        use_text: bool = True,
        text_conflict_floor: float = 0.25,
        use_scene_category: bool = True,
        scene_category_weight: float = 0.10,
        object_class_weights: Optional[dict] = None,
        relative_layout_weight: float = 0.15,
        relative_layout_sigma: float = 0.25,
        persistence_gain: float = 0.25,
        object_appearance_weight: float = 0.0,
        fusion_mode: str = "tri_layer",
        object_support_gain: float = 0.15,
        text_support_gain: float = 0.25,
    ) -> None:
        self.w_scene = float(w_scene)
        self.w_object = float(w_object)
        self.w_text = float(w_text)

        self.sigma_mask = float(sigma_mask)
        self.lambda_area = float(lambda_area)
        self.text_geom_threshold = float(text_geom_threshold)
        self.semantic_threshold = float(semantic_threshold)
        self.use_scene = bool(use_scene)
        self.use_object = bool(use_object)
        self.use_text = bool(use_text)
        self.text_conflict_floor = self._clamp(
            text_conflict_floor, 0.0, 1.0
        )
        self.use_scene_category = bool(use_scene_category)
        self.scene_category_weight = self._clamp(
            scene_category_weight, 0.0, 1.0
        )
        self.object_class_weights = {
            str(name): self._clamp(float(weight), 0.0, 1.0)
            for name, weight in (object_class_weights or {}).items()
        }
        self.relative_layout_weight = self._clamp(
            float(relative_layout_weight), 0.0, 1.0
        )
        self.relative_layout_sigma = max(1e-6, float(relative_layout_sigma))
        self.persistence_gain = max(0.0, float(persistence_gain))
        self.object_appearance_weight = self._clamp(
            float(object_appearance_weight), 0.0, 1.0
        )
        if fusion_mode not in {"tri_layer", "scene_support"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.object_support_gain = self._clamp(object_support_gain, 0.0, 1.0)
        self.text_support_gain = self._clamp(text_support_gain, 0.0, 1.0)

        if not any((self.use_scene, self.use_object, self.use_text)):
            raise ValueError("At least one cognitive layer must be enabled")


    def scene_similarity(
        self,
        query_scene_seq: List[SceneRecord],
        candidate_scene_seq: List[SceneRecord],
    ) -> float:
        """
        Computes S_scene between scene sequences.

        Expected order:
            [current_scene, previous_scene, second_previous_scene]
        """

        k = min(len(query_scene_seq), len(candidate_scene_seq))

        if k == 0:
            return 0.0

        base_weights = [0.5, 0.3, 0.2]
        weights = np.array(base_weights[:k], dtype=np.float32)
        weight_sum = float(np.sum(weights))

        if weight_sum <= 0.0:
            return 0.0

        weights = weights / weight_sum

        score = 0.0

        for idx in range(k):
            q_scene = query_scene_seq[idx]
            c_scene = candidate_scene_seq[idx]

            if q_scene is None or c_scene is None:
                continue

            if q_scene.embedding is None or c_scene.embedding is None:
                continue

            q_emb = self._normalize(q_scene.embedding)
            c_emb = self._normalize(c_scene.embedding)

            feature_sim = float(np.dot(q_emb, c_emb))
            feature_sim = self._clamp(feature_sim, 0.0, 1.0)

            if self.use_scene_category and self.scene_category_weight > 0.0:
                compatibility = category_compatibility(
                    q_scene.category_distribution,
                    c_scene.category_distribution,
                )
                if q_scene.category_distribution and c_scene.category_distribution:



                    feature_sim *= (
                        1.0 + self.scene_category_weight * compatibility
                    ) / (1.0 + self.scene_category_weight)



            score += float(weights[idx]) * feature_sim

        return self._clamp(float(score), 0.0, 1.0)


    def mask_distance(self, q_obj: StaticObject, c_obj: StaticObject) -> float:
        """
        Computes d_mask between two segmented objects.

        d_mask = sqrt(dx^2 + dy^2 + lambda_area * da^2)
        """

        dx = q_obj.x_centroid - c_obj.x_centroid
        dy = q_obj.y_centroid - c_obj.y_centroid
        da = q_obj.area - c_obj.area

        return math.sqrt(dx * dx + dy * dy + self.lambda_area * da * da)

    def spatial_similarity(self, q_obj: StaticObject, c_obj: StaticObject) -> float:
        """
        Converts d_mask into S_geom using a Gaussian kernel.
        """

        if self.sigma_mask <= 0.0:
            return 0.0

        d = self.mask_distance(q_obj, c_obj)
        score = math.exp(-((d * d) / (2.0 * self.sigma_mask * self.sigma_mask)))
        return self._clamp(score, 0.0, 1.0)

    def object_match_score(self, q_obj: StaticObject, c_obj: StaticObject) -> float:
        """
        Computes M_obj between two objects.

        M_obj = class_gate * sqrt(query_conf * candidate_conf) * S_geom

        Confidence expresses evidence reliability.  The geometric similarity
        must not square confidence merely because the same detector observed
        both frames.
        """

        if q_obj.class_name != c_obj.class_name:
            return 0.0

        def persistent_confidence(obj):
            confidence = self._clamp(obj.seg_conf, 0.0, 1.0)
            observations = max(1, int(getattr(obj, "observation_count", 1)))
            return 1.0 - (1.0 - confidence) / (
                1.0 + self.persistence_gain * (observations - 1)
            )

        q_conf = persistent_confidence(q_obj)
        c_conf = persistent_confidence(c_obj)
        s_geom = self.spatial_similarity(q_obj, c_obj)

        reliability = math.sqrt(q_conf * c_conf)
        similarity = s_geom
        q_embedding = getattr(q_obj, "appearance_embedding", None)
        c_embedding = getattr(c_obj, "appearance_embedding", None)
        if (
            self.object_appearance_weight > 0.0
            and q_embedding is not None and c_embedding is not None
        ):
            cosine = float(np.dot(
                np.asarray(q_embedding).reshape(-1),
                np.asarray(c_embedding).reshape(-1),
            ))
            appearance = self._clamp((cosine + 1.0) * 0.5, 0.0, 1.0)
            similarity = (
                (1.0 - self.object_appearance_weight) * s_geom
                + self.object_appearance_weight * appearance
            )
        return self._clamp(reliability * similarity, 0.0, 1.0)

    def object_similarity(
        self,
        query_keyframe: KeyframeRecord,
        candidate_keyframe: KeyframeRecord,
    ) -> float:
        """
        Computes S_obj between the current query keyframe and one candidate keyframe.

        Find an optimal one-to-one assignment within each exact object class.
        Unmatched query objects contribute zero; a candidate object cannot be
        reused to explain multiple query objects.
        """

        q_objects = query_keyframe.static_objects
        c_objects = candidate_keyframe.static_objects

        if len(q_objects) == 0 or len(c_objects) == 0:
            return 0.0

        matched_score = 0.0
        assignments = []
        query_weight = sum(
            self.object_class_weights.get(item.class_name, 1.0)
            for item in q_objects
        )
        if query_weight <= 0.0:
            return 0.0
        classes = {item.class_name for item in q_objects}
        for class_name in classes:
            query_class = [item for item in q_objects
                           if item.class_name == class_name]
            candidate_class = [item for item in c_objects
                               if item.class_name == class_name]
            if not candidate_class:
                continue
            scores = np.asarray([
                [self.object_match_score(query, candidate)
                 for candidate in candidate_class]
                for query in query_class
            ], dtype=np.float64)
            query_indices, candidate_indices = linear_sum_assignment(
                scores, maximize=True
            )
            class_weight = self.object_class_weights.get(class_name, 1.0)
            matched_score += class_weight * float(
                scores[query_indices, candidate_indices].sum()
            )
            assignments.extend(
                (query_class[q_index], candidate_class[c_index], class_weight)
                for q_index, c_index in zip(query_indices, candidate_indices)
            )

        assignment_score = self._clamp(
            matched_score / query_weight, 0.0, 1.0
        )
        if self.relative_layout_weight <= 0.0 or len(assignments) < 2:
            return assignment_score

        relation_total = 0.0
        relation_weight = 0.0
        for first in range(len(assignments)):
            q_first, c_first, first_weight = assignments[first]
            for second in range(first + 1, len(assignments)):
                q_second, c_second, second_weight = assignments[second]
                q_dx = q_second.x_centroid - q_first.x_centroid
                q_dy = q_second.y_centroid - q_first.y_centroid
                c_dx = c_second.x_centroid - c_first.x_centroid
                c_dy = c_second.y_centroid - c_first.y_centroid
                distance = math.hypot(q_dx - c_dx, q_dy - c_dy)
                similarity = math.exp(-(
                    distance * distance
                    / (2.0 * self.relative_layout_sigma ** 2)
                ))
                weight = math.sqrt(first_weight * second_weight)
                relation_total += weight * similarity
                relation_weight += weight
        if relation_weight <= 0.0:
            return assignment_score
        relation_score = relation_total / relation_weight
        return self._clamp(
            (1.0 - self.relative_layout_weight) * assignment_score
            + self.relative_layout_weight * relation_score,
            0.0, 1.0,
        )


    def text_gate(self, q_obj: StaticObject, c_obj: StaticObject) -> bool:
        """
        Allows text comparison only when:
            1. object classes match
            2. spatial similarity is high enough
        """

        if q_obj.class_name != c_obj.class_name:
            return False

        return self.spatial_similarity(q_obj, c_obj) >= self.text_geom_threshold

    def object_text_similarity(self, q_obj: StaticObject, c_obj: StaticObject) -> float:
        """
        Computes object-level OCR similarity between two text-bearing objects.
        """

        if not self.text_gate(q_obj, c_obj):
            return 0.0

        if len(q_obj.texts) == 0 or len(c_obj.texts) == 0:
            return 0.0

        weighted_scores = []

        for q_text in q_obj.texts:
            best = 0.0

            for c_text in c_obj.texts:
                s_str = self.string_similarity(q_text.text, c_text.text)
                q_conf = self._clamp(q_text.conf, 0.0, 1.0)
                c_conf = self._clamp(c_text.conf, 0.0, 1.0)

                best = max(best, s_str * q_conf * c_conf)

            weight = max(self.text_distinctiveness(q_text.text), 0.05)
            weighted_scores.append((weight, best))

        denominator = sum(weight for weight, _ in weighted_scores)
        if denominator <= 0.0:
            return 0.0
        return self._clamp(
            sum(weight * score for weight, score in weighted_scores)
            / denominator,
            0.0,
            1.0,
        )

    def text_similarity(
        self,
        query_keyframe: KeyframeRecord,
        candidate_keyframe: KeyframeRecord,
    ) -> Tuple[float, float]:
        """
        Computes S_text and gamma_T.

        gamma_T = 1 only when both keyframes contain text-bearing static objects.
        """

        q_text_objects = [
            obj for obj in query_keyframe.static_objects if len(obj.texts) > 0
        ]
        c_text_objects = [
            obj for obj in candidate_keyframe.static_objects if len(obj.texts) > 0
        ]

        if len(q_text_objects) == 0 or len(c_text_objects) == 0:
            return 0.0, 0

        best_scores = []

        for q_obj in q_text_objects:
            best = 0.0

            for c_obj in c_text_objects:
                best = max(best, self.object_text_similarity(q_obj, c_obj))

            best_scores.append(best)

        raw_score = self._clamp(float(np.mean(best_scores)), 0.0, 1.0)
        bounded_score = self.text_conflict_floor + (
            1.0 - self.text_conflict_floor
        ) * raw_score
        query_strength = float(np.mean([
            self.text_distinctiveness(text.text)
            for obj in q_text_objects
            for text in obj.texts
        ]))
        candidate_strength = float(np.mean([
            self.text_distinctiveness(text.text)
            for obj in c_text_objects
            for text in obj.texts
        ]))
        evidence_strength = math.sqrt(query_strength * candidate_strength)
        return (
            self._clamp(bounded_score, 0.0, 1.0),
            self._clamp(evidence_strength, 0.0, 1.0),
        )

    def text_evidence_strength(self, keyframe: KeyframeRecord) -> float:
        """Return query/candidate OCR evidence quality independent of its peer."""
        strengths = [
            self.text_distinctiveness(text.text)
            for obj in keyframe.static_objects
            for text in obj.texts
        ]
        if not strengths:
            return 0.0
        return self._clamp(float(np.mean(strengths)), 0.0, 1.0)

    def string_similarity(self, a: str, b: str) -> float:
        """
        Normalised Levenshtein similarity.

        Returns 1 for identical strings and approaches 0 for very different strings.
        """

        a = self._clean_text(a)
        b = self._clean_text(b)

        if len(a) == 0 or len(b) == 0:
            return 0.0

        dist = self._levenshtein_distance(a, b)
        denom = max(len(a), len(b), 1)
        edit_score = 1.0 - (dist / denom)

        tokens_a = set(a.split())
        tokens_b = set(b.split())
        token_union = tokens_a | tokens_b
        token_score = (
            len(tokens_a & tokens_b) / len(token_union)
            if token_union
            else 0.0
        )
        compact_a = a.replace(" ", "")
        compact_b = b.replace(" ", "")
        containment_score = 0.0
        if min(len(compact_a), len(compact_b)) >= 4 and (
            compact_a in compact_b or compact_b in compact_a
        ):
            containment_score = (
                min(len(compact_a), len(compact_b))
                / max(len(compact_a), len(compact_b))
            )

        return self._clamp(
            max(edit_score, token_score, containment_score), 0.0, 1.0
        )

    def text_distinctiveness(self, text: str) -> float:
        cleaned = self._clean_text(text)
        compact = cleaned.replace(" ", "")
        if not compact:
            return 0.0

        common = {
            "exit", "open", "closed", "stop", "road", "street",
            "shop", "store", "parking", "welcome",
        }
        length_score = min(1.0, len(compact) / 8.0)
        digit_bonus = (
            0.15 if any(character.isdigit() for character in compact) else 0.0
        )
        common_factor = 0.35 if cleaned in common else 1.0
        return self._clamp(
            (0.2 + 0.8 * length_score + digit_bonus) * common_factor,
            0.0,
            1.0,
        )


    def unified_score(
        self,
        query_keyframe: KeyframeRecord,
        candidate_keyframe: KeyframeRecord,
        query_scene_seq: List[SceneRecord],
        candidate_scene_seq: List[SceneRecord],
    ) -> float:
        """
        Computes Lambda(Q_t, C_i).

        Scene uses a three-scene sequence.
        Objects and text use the supplied semantic keyframe records. HuMemSLAM
        may enrich those records from a small local neighbourhood before this
        model is called; the centre keyframe identity and pose are preserved.
        """

        return self.score_breakdown(
            query_keyframe,
            candidate_keyframe,
            query_scene_seq,
            candidate_scene_seq,
        )["unified_score"]

    def score_breakdown(
        self,
        query_keyframe: KeyframeRecord,
        candidate_keyframe: KeyframeRecord,
        query_scene_seq: List[SceneRecord],
        candidate_scene_seq: List[SceneRecord],
    ) -> dict:
        """Return fused score using a denominator determined only by the query.

        A layer supported by the query keeps the same weight for every
        candidate. Missing candidate evidence therefore contributes zero
        instead of removing that layer from the denominator.
        """
        weighted_scores = []
        result = {
            "scene_score": None,
            "object_score": None,
            "text_score": None,
            "text_evidence": 0.0,
            "unified_score": 0.0,
        }

        query_scene_available = (
            bool(query_scene_seq)
            and any(item is not None and item.embedding is not None
                    for item in query_scene_seq)
        )
        candidate_scene_available = (
            bool(candidate_scene_seq)
            and any(item is not None and item.embedding is not None
                    for item in candidate_scene_seq)
        )
        if self.use_scene and self.w_scene > 0.0 and query_scene_available:
            result["scene_score"] = (
                self.scene_similarity(query_scene_seq, candidate_scene_seq)
                if candidate_scene_available else 0.0
            )
            weighted_scores.append((self.w_scene, result["scene_score"]))

        query_object_available = bool(query_keyframe.static_objects)
        candidate_object_available = bool(candidate_keyframe.static_objects)
        if self.use_object and self.w_object > 0.0 and query_object_available:
            result["object_score"] = (
                self.object_similarity(query_keyframe, candidate_keyframe)
                if candidate_object_available else 0.0
            )
            weighted_scores.append((self.w_object, result["object_score"]))

        if self.use_text and self.w_text > 0.0:
            query_text_strength = self.text_evidence_strength(query_keyframe)
            if query_text_strength > 0.0:
                candidate_text_strength = self.text_evidence_strength(
                    candidate_keyframe
                )
                s_text, _ = self.text_similarity(
                    query_keyframe, candidate_keyframe
                )


                shared_strength = min(
                    query_text_strength, candidate_text_strength
                )
                reliability_ratio = shared_strength / query_text_strength
                result["text_score"] = s_text
                result["text_evidence"] = shared_strength
                weighted_scores.append((
                    self.w_text * query_text_strength,
                    s_text * reliability_ratio,
                ))

        if self.fusion_mode == "scene_support" and result["scene_score"] is not None:
            scene_score = float(result["scene_score"])
            support = 0.0
            if result["object_score"] is not None:
                support += self.object_support_gain * float(result["object_score"])
            if result["text_score"] is not None:
                support += (
                    self.text_support_gain
                    * float(result["text_evidence"])
                    * float(result["text_score"])
                )
            result["support_score"] = self._clamp(support, 0.0, 1.0)
            result["unified_score"] = self._clamp(
                scene_score + (1.0 - scene_score) * result["support_score"],
                0.0,
                1.0,
            )
            return result

        denominator = sum(weight for weight, _ in weighted_scores)
        if denominator <= 0.0:
            return result

        score = sum(
            weight * layer_score for weight, layer_score in weighted_scores
        ) / denominator

        result["unified_score"] = self._clamp(float(score), 0.0, 1.0)
        return result

    def select_best_candidate(
        self,
        query_keyframe: KeyframeRecord,
        query_scene_seq: List[SceneRecord],
        candidate_keyframes: List[KeyframeRecord],
        candidate_scene_sequences: List[List[SceneRecord]],
    ) -> Tuple[Optional[KeyframeRecord], float]:
        """
        Selects C* = argmax Lambda(Q_t, C_i).
        """

        best_candidate = None
        best_score = -1.0

        for candidate, cand_scene_seq in zip(candidate_keyframes, candidate_scene_sequences):
            score = self.unified_score(
                query_keyframe=query_keyframe,
                candidate_keyframe=candidate,
                query_scene_seq=query_scene_seq,
                candidate_scene_seq=cand_scene_seq,
            )

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate, float(best_score)

    def should_trigger_semantic_recovery(
        self,
        n_inliers: int,
        geom_threshold: int,
        semantic_score: float,
    ) -> bool:
        """
        Recovery condition:
            ORB-SLAM3 inliers are low and semantic confidence is high.
        """

        return n_inliers < geom_threshold and semantic_score > self.semantic_threshold


    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(vec))

        if norm <= 1e-12:
            return vec

        return vec / norm

    def _clean_text(self, text: str) -> str:
        return "".join(ch for ch in text.strip().lower() if ch.isalnum() or ch.isspace())

    def _levenshtein_distance(self, a: str, b: str) -> int:
        if a == b:
            return 0

        if len(a) == 0:
            return len(b)

        if len(b) == 0:
            return len(a)

        prev = list(range(len(b) + 1))

        for i, ca in enumerate(a, start=1):
            curr = [i]

            for j, cb in enumerate(b, start=1):
                cost = 0 if ca == cb else 1
                curr.append(
                    min(
                        prev[j] + 1,
                        curr[j - 1] + 1,
                        prev[j - 1] + cost,
                    )
                )

            prev = curr

        return prev[-1]

    def _clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))
