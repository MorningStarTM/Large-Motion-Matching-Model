"""
KNN search over the Matching Database.

Two backends behind one interface: scipy's cKDTree, and brute force NumPy. The
brute force path is the honest reference -- it is what
`tests/test_knn_matches_reference.py` checks the tree against -- and for
locomotion sized data it is fast enough to use directly.

The search returns the top K, about 4, not only the best: the top K are the
model's retrieved context in plan section 4.1, while the teacher's own playback
uses K equal to 1.

Exclusions matter as much as the search itself. Phase 1 forbids retrieving from
the sequence's own clip (leave one clip out), and the controller forbids rows
adjacent to the frame already playing, so a search cannot trivially return
where the character already is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mmm.data.database import MotionDatabase

__all__ = ["SearchResult", "KnnIndex"]


@dataclass
class SearchResult:
    """What one query returned.

    Attributes:
        rows: `(k,)` row indices into the database, nearest first.
        distances: `(k,)` distances in the normalized, weighted feature space.
        clip_ids: `(k,)` which clip each row came from.
    """

    rows: np.ndarray
    distances: np.ndarray
    clip_ids: np.ndarray

    @property
    def best(self) -> int:
        return int(self.rows[0])

    def __len__(self) -> int:
        return len(self.rows)


class KnnIndex:
    """Nearest neighbour index over a database's normalized feature rows.

    Args:
        database: the database to index. Its normalization must be fitted.
        backend: "kdtree" (scipy cKDTree) or "brute" (NumPy).
    """

    def __init__(self, database: MotionDatabase, backend: str = "kdtree"):
        if backend not in ("kdtree", "brute"):
            raise ValueError(f"backend must be 'kdtree' or 'brute'; got {backend!r}")

        self.database = database
        self.backend = backend
        self._rows = database.matching.normalized()
        self._tree = None

        if backend == "kdtree":
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self._rows)

    @property
    def n_rows(self) -> int:
        return int(self._rows.shape[0])

    def query(
        self,
        query: np.ndarray,
        k: int = 4,
        exclude_clips=None,
        exclude_rows=None,
    ) -> SearchResult:
        """Find the nearest rows to one normalized query vector.

        Args:
            query: `(feature_dim,)`, already normalized and weighted -- pass a
                raw query through `FeatureNormalization.apply` first.
            k: how many to return.
            exclude_clips: clip ids that may not be returned. This is leave one
                clip out.
            exclude_rows: specific row indices that may not be returned, e.g.
                the window around the currently playing frame.

        Returns:
            A `SearchResult` with at most `k` entries, nearest first.
        """
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self._rows.shape[1]:
            raise ValueError(
                f"query has {query.shape[0]} dimensions, database rows have "
                f"{self._rows.shape[1]}. The query layout and the row layout "
                "must match exactly -- see mmm/data/features.py."
            )

        excluded = self._excluded_mask(exclude_clips, exclude_rows)

        # Over-fetch so that after dropping excluded rows there are still k
        # left. Excluding a whole clip can remove a large contiguous block, so
        # the margin scales with how much was excluded rather than being a
        # fixed constant.
        n_excluded = int(excluded.sum()) if excluded is not None else 0
        fetch = min(self.n_rows, k + n_excluded) if n_excluded else min(self.n_rows, k)

        if self.backend == "kdtree":
            distances, rows = self._tree.query(query, k=fetch)
            distances = np.atleast_1d(distances)
            rows = np.atleast_1d(rows)
            # cKDTree pads with inf / n_rows when fewer than k points exist.
            keep = rows < self.n_rows
            rows, distances = rows[keep], distances[keep]
        else:
            diff = self._rows - query
            all_distances = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            if fetch >= self.n_rows:
                rows = np.argsort(all_distances)
            else:
                rows = np.argpartition(all_distances, fetch - 1)[:fetch]
                rows = rows[np.argsort(all_distances[rows])]
            distances = all_distances[rows]

        if excluded is not None:
            keep = ~excluded[rows]
            rows, distances = rows[keep], distances[keep]

        rows = rows[:k]
        distances = distances[:k]

        if len(rows) == 0:
            raise RuntimeError(
                "search returned no candidates: every nearby row was excluded. "
                "Check the exclusion sets -- excluding the only clip in the "
                "database leaves nothing to retrieve."
            )

        return SearchResult(
            rows=rows.astype(np.int64),
            distances=distances.astype(np.float32),
            clip_ids=self.database.matching.clip_ids[rows],
        )

    def _excluded_mask(self, exclude_clips, exclude_rows):
        if not exclude_clips and exclude_rows is None:
            return None

        mask = np.zeros(self.n_rows, dtype=bool)
        if exclude_clips:
            clip_ids = np.asarray(list(exclude_clips), dtype=np.int32)
            mask |= np.isin(self.database.matching.clip_ids, clip_ids)
        if exclude_rows is not None:
            rows = np.asarray(exclude_rows, dtype=np.int64)
            rows = rows[(rows >= 0) & (rows < self.n_rows)]
            mask[rows] = True
        return mask

    def query_batch(self, queries: np.ndarray, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """Unexcluded search for many queries at once, used by training data
        generation where per query exclusions are not needed.

        Returns `(rows, distances)`, both `(n_queries, k)`.
        """
        queries = np.asarray(queries, dtype=np.float32)
        if self.backend == "kdtree":
            distances, rows = self._tree.query(queries, k=k)
        else:
            rows = np.empty((len(queries), k), dtype=np.int64)
            distances = np.empty((len(queries), k), dtype=np.float32)
            for i, query in enumerate(queries):
                result = self.query(query, k=k)
                rows[i] = result.rows
                distances[i] = result.distances
        return np.atleast_2d(rows).astype(np.int64), np.atleast_2d(distances).astype(np.float32)
