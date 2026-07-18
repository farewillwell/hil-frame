# HIL Frame

`hil_frame` is a Human-in-the-Loop robot learning framework with four separated modules:

- `control`: human phase, human action, policy action chunks, chunk-to-step arbitration, safety fallback, stop/reset requests.
- `env`: reset, current observation, step, render, success, and action specification.
- `learner`: policy inference, policy server, weight manager, and algorithm extension points.
- `data`: canonical trajectory schema, local pending writer, reliable upload, remote store, transition view, and sampling.

Configuration files are parallel, not overrides. `configs/system.yaml` only contains global identity, role, endpoints, seed, and logging. Module configs only contain their own parameters.

Human control is explicit phase control:

- `IDLE`: human does not control the robot.
- `ACTIVE`: human owns control, even when action is zero.
- `STOPPED`: human requested stop and reset.

Keyboard defaults are `Enter` to enter control, `Backspace` to exit control, `F12` to stop/reset, and `Esc` to quit. Motion keys are `W/S`, `D/A`, `Q/E`, `I/K`, `L/J`, `U/O`, `X/Z`.

Control authority is never inferred from action norm. A zero human action while `ACTIVE` means the human intentionally holds still.

The local control loop uses:

```python
obs = env.get_observation()
controller.submit_observation(obs, context)
control_output = controller.get_action(obs, context)
result = env.step(control_output.action)
builder.append_step(obs, control_output, result)
```

Each trajectory step stores the pre-action observation and the environment returned `applied_action`. Canonical trajectory data does not store `next_observation`, `intervention_active`, `intervention_onset`, or `intervention_end`. It stores only `action_source`: `policy`, `human`, `safety`, or `zero_fallback`.

Transitions are a remote logical view. `TransitionView` uses the next step observation as `next_observation`, and uses `final_observation` for the last transition. Human segments are derived from consecutive `action_source == HUMAN` ranges.

Policy communication is latest-only: local main never blocks for policy inference; old unsent observations can be replaced, and old proposals are rejected by `trajectory_id` and `obs_id`. Human entering `ACTIVE` clears any remaining policy chunk. Human exit never restores old chunk; the controller waits for a new proposal and outputs zero fallback.

Trajectory upload uses full trajectory files, not per-step uploads. Local data has only `active/` and `pending/`. A pending file is deleted only after durable ACK with matching trajectory id, checksum, and `stored=True`. Pickle payloads are compressed with gzip; load them only in trusted environments.

Run dummy local:

```bash
python run_local.py \
  --system-config configs/system.yaml \
  --control-config hil_frame/control/config.yaml \
  --env-config hil_frame/env/config.yaml \
  --data-config hil_frame/data/config.yaml
```

Run remote servers:

```bash
python run_remote.py \
  --system-config configs/system.yaml \
  --learner-config hil_frame/learner/config.yaml \
  --data-config hil_frame/data/config.yaml \
  --policy-backend zero
```

LIBERO is selected by setting `backend: libero` in `hil_frame/env/config.yaml`. OpenPI is reserved at `hil_frame/learner/policy/openpi.py` and raises `NotImplementedError` until a real adapter is installed. SpaceMouse and real robot interfaces are present as explicit skeletons. New algorithms go under `hil_frame/learner/algorithms/`.
