"""
Shared frame-validation helpers for the ALFA loop phases.

The real `arc_agi` SDK has been observed returning FrameDataRaw objects
whose .frame attribute is an empty list ([]) mid-episode on certain
games (notably ls20 step 151 during long action sequences). Accessing
`.frame[0]` on such an object raises IndexError and crashes the loop.

Every phase that reads from env.get_observation() or env.step() must
guard against this case. The canonical guard lives here so all callers
share one definition.
"""


def has_valid_grid(obs) -> bool:
    """
    True when `obs` has a non-empty .frame list whose first element is a
    usable grid (not None). Guards every phase-loop iteration against the
    real arc_agi SDK quirk of returning frame=[] mid-episode.

    Returns False for:
      - obs is None
      - obs has no .frame attribute
      - obs.frame is an empty list
      - obs.frame[0] is None

    Returns True otherwise. Callers should only access obs.frame[0] after
    this check passes.
    """
    if obs is None:
        return False
    frame_list = getattr(obs, "frame", None)
    if not frame_list:
        return False
    return frame_list[0] is not None
