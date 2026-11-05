"""
The classic motion matching controller: the 10 frame loop.

Every 10 frames, build a query from the character's current feet and hips plus
the desired path from the joystick, search, and if the best candidate is not
simply the continuation of the frame already playing, jump there and start an
inertialization transition. Every frame, advance one row of the Animation
Database and apply the decaying transition offset.

This class is used in four places, which is why it is a library object rather
than a demo script:

1. Phase 0, as the baseline controller driven by a gamepad.
2. Phase 2, as the source of SFT and DAgger targets.
3. Phase 3, as the in the loop reference pose behind the `P_teach` reward term.
4. Section 9, as the quality baseline MMM is compared against.

`step` runs the stateful loop. `reference_pose` answers "what would you do at
this state?" without disturbing that loop, which is what Phase 2 DAgger
labelling and the Phase 3 reward both need.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mmm.data import features as feat
from mmm.data import rotations, spaces
from mmm.data.database import (
    MotionDatabase,
    assemble_pose,
    pose_to_hips_height,
    pose_to_root_velocity,
    pose_to_rotations,
)
from mmm.data.fk import forward_kinematics
from mmm.data.skeleton import HIPS, LEFT_FOOT, RIGHT_FOOT
from mmm.teacher.inertialize import Inertializer
from mmm.teacher.search import KnnIndex, SearchResult
from mmm.teacher.trajectory import TrajectoryPredictor

__all__ = ["ControllerState", "StepRecord", "KnnController"]


@dataclass
class ControllerState:
    """Everything the controller carries between frames.

    Kept as a separate object so a state can be snapshotted, restored, or
    constructed from scratch for a stateless `reference_pose` query.
    """

    row: int
    origin: np.ndarray
    forward: np.ndarray
    local_rotations: np.ndarray
    hips_height: float
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    frames_since_search: int = 0

    def copy(self) -> "ControllerState":
        return ControllerState(
            row=self.row,
            origin=self.origin.copy(),
            forward=self.forward.copy(),
            local_rotations=self.local_rotations.copy(),
            hips_height=self.hips_height,
            linear_velocity=self.linear_velocity.copy(),
            angular_velocity=self.angular_velocity.copy(),
            frames_since_search=self.frames_since_search,
        )


@dataclass
class StepRecord:
    """What one frame of the controller did.

    Exposed rather than kept private because the rest of the project needs the
    intermediate quantities, not only the final pose: Phase 2 trains on the
    query and the candidates, and the overlays draw them.
    """

    pose: np.ndarray
    local_rotations: np.ndarray
    origin: np.ndarray
    forward: np.ndarray
    searched: bool
    jumped: bool
    query: np.ndarray | None = None
    candidates: np.ndarray | None = None
    distances: np.ndarray | None = None
    chosen_row: int = -1


class KnnController:
    """Classic KNN motion matching.

    Args:
        database: the Matching and Animation Databases.
        index: a built `KnnIndex` over that database. Passed in rather than
            built here so several controllers can share one index.
        search_interval: frames between searches. 10 is the classic value, and
            making it configurable is what gives the learned re grounding
            experiment a fixed interval baseline to beat.
        k: how many candidates to retrieve. The teacher plays the best one; the
            full top K is recorded as the model's context.
        transition_cost: a candidate must beat the continuation of the current
            frame by this much, in normalized feature distance, before a jump
            happens. Without it the controller jumps on nearly every search to
            a fractionally better row and the motion shimmers. Measured over
            600 frames of forward walking on four LaFAN1 clips: 0.0 gives 18
            jumps in 60 searches, 0.5 gives 6, and 2.0 gives none at all but
            drops the character to a third of the requested speed because it
            refuses to leave the clip it is on. Hence 0.5 by default.
    """

    def __init__(
        self,
        database: MotionDatabase,
        index: KnnIndex | None = None,
        search_interval: int = 10,
        k: int = 4,
        transition_cost: float = 0.5,
        trajectory: TrajectoryPredictor | None = None,
        inertializer: Inertializer | None = None,
        exclude_clips=None,
        root_source: str = "hybrid",
    ):
        self.database = database
        self.index = index if index is not None else KnnIndex(database)
        self.search_interval = search_interval
        self.k = k
        self.transition_cost = transition_cost
        self.exclude_clips = set(exclude_clips or ())

        if root_source not in ("hybrid", "simulation", "animation"):
            raise ValueError(
                "root_source must be 'hybrid', 'simulation' or 'animation'; "
                f"got {root_source!r}"
            )
        # "hybrid": facing from the spring damper, translation from the
        #   animation. The default: the only mode that is both steerable and
        #   free of added foot sliding.
        # "simulation": the spring damper owns position and facing too. Exactly
        #   controllable but slides the feet, since the root then moves at the
        #   requested speed rather than the animation's.
        # "animation": integrate the retrieved clip's own root velocity, the
        #   textbook description of motion matching. Faithful feet, but the
        #   heading random walks, because a 10 frame playback commits to part
        #   of a turn the search only constrained at 20 frames and beyond.
        self.root_source = root_source

        skeleton = database.skeleton
        self.skeleton = skeleton
        self.n_joints = skeleton.n_joints
        self.frame_time = database.frame_time

        self.left_foot = skeleton.index(LEFT_FOOT)
        self.right_foot = skeleton.index(RIGHT_FOOT)
        self.hips = skeleton.index(HIPS)

        self.trajectory = trajectory if trajectory is not None else TrajectoryPredictor(
            frame_time=self.frame_time
        )
        self.inertializer = inertializer if inertializer is not None else Inertializer(
            self.n_joints, frame_time=self.frame_time
        )

        self.state: ControllerState | None = None
        self.reset()

    # -- lifecycle --------------------------------------------------------

    def reset(self, row: int = 0, origin=None, forward=None) -> None:
        """Place the character at a database row.

        Starting from a row that is marked invalid, i.e. one of the last frames
        of a clip, would leave the controller with no future trajectory, so the
        row is advanced to the next valid one.
        """
        valid = self.database.matching.valid
        while row < len(valid) - 1 and not valid[row]:
            row += 1

        pose = self.database.animation.poses[row]
        linear, angular = pose_to_root_velocity(pose, self.n_joints)

        self.state = ControllerState(
            row=int(row),
            origin=np.zeros(3, np.float32) if origin is None else np.asarray(origin, np.float32).copy(),
            forward=np.array([0.0, 1.0], np.float32) if forward is None else np.asarray(forward, np.float32).copy(),
            local_rotations=pose_to_rotations(pose, self.n_joints).astype(np.float32),
            hips_height=float(pose_to_hips_height(pose, self.n_joints)),
            linear_velocity=np.asarray(linear, np.float32).copy(),
            angular_velocity=np.asarray(angular, np.float32).copy(),
            frames_since_search=0,
        )
        self.trajectory.reset(facing=self.state.forward)
        self.inertializer.reset()
        self._previous_world_positions = None
        self._local_velocities = np.zeros((self.n_joints, 3), np.float32)
        self._track_velocities()

    # -- the loop ---------------------------------------------------------

    def step(self, stick: np.ndarray, facing_stick: np.ndarray | None = None) -> StepRecord:
        """Advance one frame.

        Args:
            stick: `(2,)` movement request in world XZ, magnitude `[0, 1]`.
            facing_stick: `(2,)` optional separate facing request.

        Returns:
            A `StepRecord`. On a search frame it also carries the query, the
            top K candidates and their distances.
        """
        state = self.state
        self.trajectory.update(stick, facing_stick)

        searched = state.frames_since_search >= self.search_interval
        jumped = False
        query = candidates = distances = None
        chosen = -1

        if searched:
            query, result = self._search(state)
            candidates = result.rows
            distances = result.distances
            chosen = self._choose(state, result, query)
            if chosen != state.row:
                self._jump(state, chosen)
                jumped = True
            state.frames_since_search = 0

        # Advance one row of the Animation Database, then read the pose.
        state.row = int(self.database.next_frame_of([state.row])[0])
        source_pose = self.database.animation.poses[state.row]
        source_rotations = pose_to_rotations(source_pose, self.n_joints)
        source_linear, source_angular = pose_to_root_velocity(source_pose, self.n_joints)

        blended_rotations, linear, angular = self.inertializer.apply(
            source_rotations, source_linear, source_angular
        )
        self.inertializer.step()

        state.local_rotations = blended_rotations
        state.hips_height = float(pose_to_hips_height(source_pose, self.n_joints))

        if self.root_source == "hybrid":
            # Facing from the simulation, translation from the animation.
            #
            # Facing has to come from the spring damper or it is not
            # controllable at all: see `TrajectoryPredictor.step_root`.
            # Translation has to come from the animation or the feet slide,
            # because a pose's feet were captured while its own root advanced
            # by its own velocity, and moving the root at the *requested*
            # speed instead leaves the planted foot behind. Measured against
            # the 7.7 cm/s of planted foot motion in the raw mocap, taking
            # both from the simulation cost 23 to 43 cm/s of sliding; taking
            # translation from the animation returns it to the floor while
            # keeping facing exact.
            #
            # The animation's velocity is in character space, so applying it in
            # the simulated frame reproduces the clip's own root motion
            # relative to the body while the player still chooses which way
            # the body points.
            state.linear_velocity = np.asarray(linear, np.float32)
            new_forward = self.trajectory.facing.copy()
            step = spaces._rotate_out_of_frame(
                np.array(
                    [state.linear_velocity[0], state.linear_velocity[2]], np.float32
                )[None]
                * self.frame_time,
                new_forward[None],
            )[0]
            state.origin = state.origin.copy()
            state.origin[0] += step[0]
            state.origin[2] += step[1]
            state.origin[1] = 0.0
            state.forward = new_forward
            # Report the yaw rate that actually turned the frame, so that
            # integrating the logged demonstration reproduces the logged path.
            state.angular_velocity = np.array(
                [0.0, self.trajectory.facing_rate, 0.0], np.float32
            )
        elif self.root_source == "simulation":
            # The spring damper owns position as well as facing. Exactly
            # controllable, but it moves the root at the requested speed rather
            # than the animation's, which slides the feet. Kept for ablation.
            state.origin, state.forward = self.trajectory.step_root(
                state.origin, state.forward
            )
            state.linear_velocity, state.angular_velocity = (
                self.trajectory.realised_root_velocity(state.forward)
            )
        else:
            state.linear_velocity = np.asarray(linear, np.float32)
            state.angular_velocity = np.asarray(angular, np.float32)
            state.origin, state.forward = spaces.integrate_root(
                state.origin,
                state.forward,
                state.linear_velocity,
                state.angular_velocity,
                self.frame_time,
            )
        state.frames_since_search += 1
        self._track_velocities()

        pose = assemble_pose(
            blended_rotations, state.hips_height, state.linear_velocity, state.angular_velocity
        )

        return StepRecord(
            pose=pose,
            local_rotations=blended_rotations,
            origin=state.origin.copy(),
            forward=state.forward.copy(),
            searched=searched,
            jumped=jumped,
            query=query,
            candidates=candidates,
            distances=distances,
            chosen_row=chosen,
        )

    # -- search -----------------------------------------------------------

    def character_space_positions(self, state: ControllerState) -> np.ndarray:
        """Joint positions in character space, `(n_joints, 3)`.

        Because a stored pose is already character space -- the hips rotation is
        yaw stripped -- running forward kinematics from `(0, hips_height, 0)`
        lands directly in the space `compute_features` measured the rows in.
        Heights are therefore above the ground, which is what the foot position
        features and foot contact both depend on.
        """
        root = np.array([[0.0, state.hips_height, 0.0]], np.float32)
        positions, _ = forward_kinematics(self.skeleton, state.local_rotations[None], root)
        return positions[0]

    def world_positions(self, state: ControllerState) -> np.ndarray:
        """Joint positions in world space, for drawing and for measuring
        velocity the same way the database rows measured it."""
        local = self.character_space_positions(state)
        frame = spaces.RootFrame(origin=state.origin[None], forward=state.forward[None])
        return spaces.from_character_space(local[None], frame)[0]

    def _track_velocities(self) -> None:
        """Update the per joint velocity estimate. Called once every frame.

        This must run every frame, not only on search frames. A search happens
        every 10 frames, so differencing positions inside the query builder
        would divide a 10 frame displacement by one frame time and report
        velocities roughly an order of magnitude too large. Those two blocks
        are 12 of the 27 features, so the search then compares mostly noise:
        measured, it put 87% of the total distance into the two velocity blocks
        and left the character wandering off the requested heading.

        Velocities are differenced in *world* space and then rotated into the
        character frame, exactly as `compute_features` built the database
        columns. Differencing character space positions directly would fold the
        frame's own rotation into the result and bias every retrieval taken
        while turning.
        """
        state = self.state
        local_positions = self.character_space_positions(state)
        frame = spaces.RootFrame(origin=state.origin[None], forward=state.forward[None])
        world = spaces.from_character_space(local_positions[None], frame)[0]

        previous = self._previous_world_positions
        if previous is not None:
            world_velocities = (world - previous) / self.frame_time
            self._local_velocities = spaces.direction_to_character_space(
                world_velocities[None], frame
            )[0].astype(np.float32)
        self._previous_world_positions = world.copy()

    def _current_query(self, state: ControllerState) -> np.ndarray:
        """Build the query from the character's current state plus the joystick.

        Every quantity is produced the same way `compute_features` produced the
        corresponding database column, which is what makes the comparison
        valid. The velocities come from `_track_velocities`, which runs every
        frame; see its docstring for why that matters.
        """
        local_positions = self.character_space_positions(state)
        local_velocities = self._local_velocities

        foot_positions = np.stack([local_positions[self.left_foot], local_positions[self.right_foot]])
        foot_velocities = np.stack([local_velocities[self.left_foot], local_velocities[self.right_foot]])
        hip_velocity = local_velocities[self.hips]

        desired_positions, desired_directions = self.trajectory.predict(state.forward)

        return feat.build_query(
            foot_positions=foot_positions,
            foot_velocities=foot_velocities,
            hip_velocity=hip_velocity,
            desired_positions=desired_positions,
            desired_directions=desired_directions,
        )

    def _search(self, state: ControllerState) -> tuple[np.ndarray, SearchResult]:
        query = self._current_query(state)
        normalized = self.database.matching.normalization.apply(query)
        result = self.index.query(
            normalized,
            k=self.k,
            exclude_clips=self.exclude_clips or None,
        )
        return query, result

    def _choose(self, state: ControllerState, result: SearchResult, query: np.ndarray) -> int:
        """Pick a row from the candidates, preferring to stay put.

        Continuing the current clip costs nothing, so a candidate has to beat
        the frame already playing by `transition_cost` before it is worth the
        jump. Without this the controller jumps on every search to a
        fractionally better row and the motion shimmers.

        The query is passed in rather than rebuilt: `_current_query` advances
        the foot velocity history, so calling it twice in one frame would
        report zero velocity on the second call.
        """
        best_row = int(result.rows[0])
        if self.transition_cost <= 0.0:
            return best_row

        current_next = int(self.database.next_frame_of([state.row])[0])
        normalized_query = self.database.matching.normalization.apply(query)
        current_row = self._normalized_rows()[current_next]
        current_distance = float(np.linalg.norm(current_row - normalized_query))

        if result.distances[0] + self.transition_cost < current_distance:
            return best_row
        return state.row

    def _normalized_rows(self) -> np.ndarray:
        """The normalized feature rows, computed once and reused.

        `MatchingDatabase.normalized()` rebuilds the whole array, which is tens
        of megabytes; doing that inside a per frame comparison would dominate
        the controller's cost.
        """
        cached = getattr(self, "_rows_cache", None)
        if cached is None or len(cached) != self.database.n_rows:
            cached = self.database.matching.normalized()
            self._rows_cache = cached
        return cached

    def _jump(self, state: ControllerState, target_row: int) -> None:
        """Start an inertialization transition to a new row."""
        target_pose = self.database.animation.poses[target_row]
        target_rotations = pose_to_rotations(target_pose, self.n_joints)
        target_linear, target_angular = pose_to_root_velocity(target_pose, self.n_joints)

        self.inertializer.transition(
            current_rotations=state.local_rotations,
            target_rotations=target_rotations,
            current_linear=state.linear_velocity,
            target_linear=np.asarray(target_linear, np.float32),
            current_angular=state.angular_velocity,
            target_angular=np.asarray(target_angular, np.float32),
        )
        state.row = int(target_row)

    # -- stateless query, for Phases 2 and 3 ------------------------------

    def reference_pose(
        self,
        pose: np.ndarray,
        desired_positions: np.ndarray,
        desired_directions: np.ndarray,
        foot_velocities: np.ndarray | None = None,
        hip_velocity: np.ndarray | None = None,
    ) -> tuple[np.ndarray, SearchResult]:
        """What the teacher would play at an arbitrary state.

        This is what Phase 2 DAgger labels with and what the Phase 3 `P_teach`
        reward term compares against. It does not touch the controller's own
        playback state, so it can be called at every step of an MMM rollout.

        Because a pose is already in character space, no world position or
        facing is needed: the state is fully described by the pose plus the
        desired path, which is exactly the information the model itself has.

        Args:
            pose: `(pose_dim,)` the state to ask about, in the stored layout.
            desired_positions: `(n_horizons, 2)` character space.
            desired_directions: `(n_horizons, 2)` character space.
            foot_velocities: `(2, 3)` optional. Omitted means zero, which
                biases retrieval towards planted feet; pass the real value
                during a rollout, where the caller has the previous frame.
            hip_velocity: `(3,)` optional, defaulting to the pose's own stored
                root linear velocity.

        Returns:
            The teacher's pose `(pose_dim,)` and the search result behind it.
        """
        local_rotations = pose_to_rotations(pose, self.n_joints)
        hips_height = float(pose_to_hips_height(pose, self.n_joints))
        linear, _ = pose_to_root_velocity(pose, self.n_joints)

        root = np.array([[0.0, hips_height, 0.0]], np.float32)
        positions, _ = forward_kinematics(self.skeleton, local_rotations[None], root)
        foot_positions = np.stack([positions[0, self.left_foot], positions[0, self.right_foot]])

        query = feat.build_query(
            foot_positions=foot_positions,
            foot_velocities=np.zeros((2, 3), np.float32) if foot_velocities is None else foot_velocities,
            hip_velocity=linear if hip_velocity is None else hip_velocity,
            desired_positions=desired_positions,
            desired_directions=desired_directions,
        )
        normalized = self.database.matching.normalization.apply(query)
        result = self.index.query(normalized, k=self.k, exclude_clips=self.exclude_clips or None)

        # The teacher's answer is the frame after the best match: the match
        # describes the state it is in, and what it would *do* is the next one.
        next_row = int(self.database.next_frame_of([result.best])[0])
        return self.database.animation.poses[next_row].copy(), result
