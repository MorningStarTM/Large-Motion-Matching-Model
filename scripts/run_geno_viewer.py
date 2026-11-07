"""
Run the motion matching teacher on Daniel Holden's Geno character.

The same controller and the same database as `scripts/run_obstacle_teacher.py`;
only the rendering differs. A skinned character with a shadow and ambient
occlusion makes foot sliding and ground penetration obvious, which is the whole
reason GenoView exists and the reason it is worth having beside the wireframe
viewer rather than instead of it.

    # obstacle aware, with the recovered studio layout
    python scripts/run_geno_viewer.py --studio

    # plain locomotion, no obstacles
    python scripts/run_geno_viewer.py --no-obstacles

    # render 120 frames and write a PNG, without an interactive session
    python scripts/run_geno_viewer.py --studio --frames 120 --screenshot out.png

`--screenshot` is how this was developed: a renderer cannot be checked by
reading its source, and a self test that only proves it did not crash proves
very little about whether the character is inside out.

The character, the shaders and the format are Daniel Holden's, MIT licensed.
See NOTICE.md and licenses/GenoViewPython-LICENSE.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmm.obstacles.controller import ObstacleController
from mmm.obstacles.database import augment
from mmm.obstacles.geometry import BoxField
from mmm.teacher.controller import KnnController
from mmm.viewer import session as session_wiring

from scripts.run_obstacle_teacher import DEFAULT_CLIPS, build


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Motion matching on the Geno character.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bvh-dir", default=os.path.join("data", "bvh", "bvh"))
    parser.add_argument(
        "--layout", default=os.path.join("data", "obstacles", "lafan1_studio.json")
    )
    parser.add_argument("--clips", nargs="*", default=list(DEFAULT_CLIPS))
    parser.add_argument("--max-frames", type=int, default=8000)
    parser.add_argument(
        "--studio", action="store_true", help="start with the recovered layout"
    )
    parser.add_argument(
        "--no-obstacles",
        action="store_true",
        help="use the plain KnnController, for comparison",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="stop after this many frames instead of running interactively",
    )
    parser.add_argument("--screenshot", default=None, help="write a PNG before exiting")
    parser.add_argument(
        "--walk",
        action="store_true",
        help="drive the character forward instead of reading input. Use with "
        "--frames: with no input the controller stays on row 0, and row 0 of "
        "a LaFAN1 clip is the calibration T-pose.",
    )
    parser.add_argument("--skeleton", action="store_true", help="overlay the bones")
    parser.add_argument("--search-interval", type=int, default=10)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--transition-cost", type=float, default=None)
    session_wiring.add_arguments(parser)
    parser.add_argument(
        "--character",
        default=None,
        help="path to Geno.bin; defaults to the copy in "
        "mmm/viewer/geno/resources. Renamed from --model, which now means "
        "the trained checkpoint.",
    )

    args = parser.parse_args(argv)

    layout = args.layout if os.path.isfile(args.layout) else None
    if args.layout and layout is None:
        print(
            f"note: {args.layout} not found, so every clip is treated as flat "
            "ground. Run scripts/recover_obstacles.py to produce it."
        )

    # A model retrieves nothing, so the Matching Database it would search is
    # never read. Building it anyway cost 23 seconds of startup against 3 for
    # the one clip the seed pose comes from, so the model takes the short
    # path. Measured, not assumed.
    if args.model:
        database, studio = session_wiring.load_model_scene(
            args.bvh_dir, args.clips, args.max_frames, layout
        )
        augmented, motions = database, database.motions
        print(f"model scene: {database}")
    else:
        database, augmented, motions, studio = build(
            args.bvh_dir, args.clips, args.max_frames, layout
        )

    kwargs = {"search_interval": args.search_interval, "k": args.k}
    if args.transition_cost is not None:
        kwargs["transition_cost"] = args.transition_cost

    if args.no_obstacles and not args.model:
        controller = KnnController(database, **kwargs)
        field = BoxField()
        drawn = database
        description = "KNN teacher (no obstacles)"
    else:
        field = BoxField(list(studio)) if args.studio else BoxField()
        controller, drawn, description = session_wiring.build_controller(
            args, database, augmented, field, kwargs
        )
    print(f"driven by: {description}")

    from mmm.viewer.geno.app import GenoViewerApp, GenoViewerConfig

    config = GenoViewerConfig(
        width=args.width,
        height=args.height,
        show_skeleton=args.skeleton,
        title="MMM - Geno viewer",
    )
    viewer = GenoViewerApp(
        drawn,
        controller,
        field=field,
        config=config,
        model_path=args.character,
    )

    if args.frames is None:
        print(
            "\ncontrols: WASD or stick to move, mouse to turn.\n"
            "  space pause   K skeleton overlay   B boxes   M release mouse\n"
        )
    viewer.run(
        max_frames=args.frames,
        screenshot=args.screenshot,
        auto_stick=(0.0, 1.0) if args.walk else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
