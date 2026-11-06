"""
An obstacle aware KNN controller.

One overridden method. That is the measure of whether the additive design
worked, so it is worth saying why the other five are not needed:

- `_search` and `_choose` both normalise a query through
  `database.matching.normalization.apply`. `AugmentedNormalization` accepts the
  32 wide vector, so both work as written.
- `_normalized_rows` calls `database.matching.normalized()`, which
  `AugmentedMatching` already returns 32 wide.
- `__init__` builds a `KnnIndex` over the database, which `AugmentedMatching`
  duck types.
- `step`, `_jump` and the inertialization never touch features at all.

So `mmm/teacher/controller.py` is untouched, and the only thing that actually
changes is the question being asked of the database: instead of "which frame
has feet and a future path like mine", it is "which frame has feet, a future
path, *and a surface ahead* like mine".

What this does not do
---------------------
There is no collision. The character's height comes from the retrieved pose's
hips height, which is absolute and therefore already includes the box the actor
was standing on -- step onto a 47 cm platform and the pose carries the
character up with it. But nothing *prevents* a foot passing through a cube when
the search returns a flat walking frame anyway. Reducing how often that happens
is exactly what the obstacle features are for; eliminating it needs a ground
constraint, which belongs with the regrounding work in `mmm/models` rather than
here.
"""

from __future__ import annotations

import numpy as np

from mmm.data import features as feat
from mmm.obstacles.database import AugmentedDatabase
from mmm.obstacles.features import build_obstacle_query
from mmm.obstacles.geometry import BoxField
from mmm.teacher.controller import ControllerState, KnnController

__all__ = ["ObstacleController"]


class ObstacleController(KnnController):
    """`KnnController` with the obstacle block in its query.

    Args:
        database: an `AugmentedDatabase`. A plain `MotionDatabase` is refused,
            because the failure would otherwise be a shape error thrown from
            inside the search, several frames later, with nothing pointing at
            the cause.
        field: the obstacles in the live scene. Mutable afterwards through
            `self.field`, which is what lets the viewer add a cube mid-run
            without rebuilding anything -- the database rows do not change when
            the scene does, only the query.
        **kwargs: passed through to `KnnController`.
    """

    def __init__(self, database, field: BoxField | None = None, **kwargs):
        if not isinstance(database, AugmentedDatabase):
            raise TypeError(
                "ObstacleController needs an AugmentedDatabase; got "
                f"{type(database).__name__}. Build one with "
                "mmm.obstacles.database.augment(database, motions, fields)."
            )
        super().__init__(database, **kwargs)
        self.field = field if field is not None else BoxField()

    def _current_query(self, state: ControllerState) -> np.ndarray:
        """The base 27 numbers, with the obstacle block appended.

        The future path is read back out of the base query rather than
        recomputed. `KnnController._current_query` advances the trajectory
        predictor, so asking it a second time in one frame would return a
        different path from the one the base half of this query describes --
        the two halves would then be talking about different futures, which is
        the subtle version of a layout mismatch and just as damaging.
        """
        base_query = super()._current_query(state)

        desired_positions = base_query[
            feat.feature_block_slice("future_positions")
        ].reshape(-1, 2)

        obstacle = build_obstacle_query(
            self.field,
            origin_xz=state.origin[[0, 2]],
            forward_xz=state.forward,
            desired_positions=desired_positions,
        )
        return np.concatenate([base_query, obstacle]).astype(np.float32)

    # -- diagnostics ------------------------------------------------------

    def support_under_character(self) -> float:
        """Height of the surface the character is standing over.

        For the HUD, and for a caller that wants to know whether the character
        is on a box without reaching into the geometry itself.
        """
        origin = self.state.origin
        return float(self.field.support_height(origin[0], origin[2]))

    def edge_ahead(self) -> tuple:
        """Distance to the next change in support, and what it changes to.

        Returned in world terms rather than relative, since a HUD wants to read
        "47 cm box in 150 cm" and not "+47 in 150".
        """
        origin = self.state.origin
        return self.field.edge_ahead(origin[[0, 2]], self.state.forward)
