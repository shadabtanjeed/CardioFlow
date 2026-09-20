"""
The back-and-forth time schedule that drives trajectory correction.

Restora-Flow reuses RePaint's `get_schedule_jump`: walk t from noise to data one
step at a time, but every so often jump *backwards* and re-walk a stretch. Each
backward jump is where trajectory correction happens (look ahead to a one-shot
estimate of the clean clip, then re-noise to the earlier time); each forward step
is an ordinary Euler step of the ODE with mask fusion applied first.

`correction_steps = 0` degenerates to a plain forward sweep with no correction at
all, which is the natural ablation baseline for the correction mechanism itself.
"""


def jump_schedule(ode_steps: int, correction_steps: int) -> list[int]:
    """
    RePaint's jump schedule, specialised to `jump_length=1` as Restora-Flow uses it.

    Returns integer times descending from `ode_steps - 1` to -1, with each level
    revisited `correction_steps` extra times.
    """
    if ode_steps < 2:
        raise ValueError(f'ode_steps must be at least 2, got {ode_steps}.')
    if correction_steps < 0:
        raise ValueError(f'correction_steps must be >= 0, got {correction_steps}.')

    remaining = {j: correction_steps for j in range(0, ode_steps - 1)}

    t = ode_steps
    times = []
    while t >= 1:
        t -= 1
        times.append(t)
        if remaining.get(t, 0) > 0:
            remaining[t] -= 1
            t += 1
            times.append(t)
    times.append(-1)
    return times


def time_pairs(ode_steps: int, correction_steps: int) -> list[tuple[float, float]]:
    """
    The schedule as consecutive (t_last, t_cur) pairs normalised onto [0, 1],
    ordered from noise (t = 0) to data (t = 1).

    t_last < t_cur  -> forward: mask fusion, then an Euler step.
    t_last > t_cur  -> backward: trajectory correction.
    """
    times = jump_schedule(ode_steps, correction_steps)
    low, high = min(times), max(times)
    normalised = [(t - low) / (high - low) for t in times]
    normalised.reverse()
    return list(zip(normalised[:-1], normalised[1:]))
