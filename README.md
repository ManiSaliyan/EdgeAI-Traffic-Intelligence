# ATLAS Pro — Adaptive Traffic Signal Control with Reinforcement Learning

An RL-based traffic light controller for a single 4-way intersection, designed
around a fixed-cycle India-style signal pattern (one direction gets green at a
time, in a strict rotation) rather than the paired NS/EW phasing common in
Western signal design. Built and validated against real Indian intersection
research (Mangaluru's Nanthur Junction), with a genuine emergency-preemption
system and a metacognitive safety layer (MUSE) that can hand control to a
simple fixed timer when the RL policy's own confidence is low or the whole
junction is gridlocked.

## What this is, in one paragraph

A Dueling Double DQN agent watches a 22-dimensional state (queue, speed, wait,
vehicle count, and emergency-presence for each of the 4 approaches, plus the
current phase and elapsed time) and every 5 simulated seconds chooses one of
3 actions — **Maintain**, **Advance to the next direction in the fixed
N→E→S→W cycle**, or **Extend** the current green beyond its normal cap. The
agent is trained and evaluated in real SUMO simulations (not a toy
gridworld), and every performance claim in this repo comes from actual
`sumo`/`traci` runs with a real emissions/fuel model, not estimates.

## Why a fixed cycle instead of free phase selection

Real Indian intersections — including the specific one this project models
(Mangaluru) — are frequently controlled this way in practice, and it
sidesteps a large class of unsafe or illegal phase combinations by
construction. The tradeoff is that the agent can only control *timing*
(how long each direction holds green), not *sequence* — which turned out to
still leave substantial room for real, measured improvement over a fixed
timer (see Results).

## Architecture

- **Algorithm**: Dueling Double DQN, with Noisy Networks (state-dependent
  exploration, no epsilon-greedy), Prioritized Experience Replay, and 5-step
  returns.
- **Network**: `[512, 256, 256, 128]` shared trunk, split into Value and
  Advantage streams (`algoritmos_avanzados.py`).
- **State (22-D)**: per direction (N, S, E, W) — queue length, mean speed,
  waiting time, vehicle count, emergency-present flag (5 features × 4
  directions), plus current phase index and elapsed-episode fraction. A
  6th per-direction feature (lane occupancy) was measured and removed —
  it correlated 0.987–1.000 with vehicle count and added nothing.
- **Actions (3)**: `0=Maintain, 1=Advance, 2=Extend`.
- **Network geometry**: 3 lanes in + 3 lanes out per direction (6 lanes per
  road), built via `netconvert` from plain node/edge definitions rather than
  hand-edited XML (`intersection_net_3lane.xml`).

## Reward function

```
reward = queue_length_weight   * total_queue_all_directions          (time-scaled)
       + wait_time_weight      * total_wait_all_directions           (time-scaled)
       + throughput_weight     * vehicles_arrived_this_decision
       + speed_weight          * mean_speed_all_directions           (time-scaled)
       + emergency_weight      * emergency_vehicles_still_waiting
       + emergency_cleared_bonus  (when a waiting emergency vehicle clears)
       + phase_change_penalty     (flat, only on non-emergency switches)
       + premature_switch_penalty_weight * queue_of_direction_being_abandoned
```

The last term was added after diagnosing that a heavy direction was being
under-served relative to its real demand share (24% of green time vs. 31% of
real traffic) because the global wait penalty didn't distinguish "left while
still busy" from "left after clearing." It's scaled by exactly how many
vehicles were still queued in the direction the agent is switching away from.

`queue`/`wait`/`speed` terms are scaled by `delta_time/10.0` so that changing
the decision interval doesn't silently inflate or shrink the accumulated
episode reward — this was a real bug, found and fixed mid-project.

## Scenarios

Traffic volumes are calibrated against real published data, not guessed:
**"A Study on Rotary Intersection at Mangaluru — Nanthur Junction"** (IJCESR
2017) found the real junction's saturation ratio Y=1.117 (demand exceeds
capacity), and several Indian intersection studies (Vellore, Chennai) show
two-wheelers consistently at 60–82% of traffic volume. Vehicle composition
here: 65% motorcycle, 20% car, 8% auto-rickshaw, 7% bus/truck
(`generar_escenarios_mangalore.py`).

| Scenario | Character | Real-world basis |
|---|---|---|
| `simple` | Moderate daytime, non-peak | Baseline volume |
| `noche` | Very light, night | ~15% of daytime volume |
| `hora_punta` | Peak-hour, deliberately oversaturated | Calibrated to reproduce Nanthur's real Y>1 congestion |
| `evento` | Moderate base + a temporary directional surge | Simulates a stadium/temple event letting out |
| `traffic_real` | User-supplied real traffic data | Not synthetic — actual uploaded demand |

## Emergency handling

The environment itself (not the RL agent) detects an emergency vehicle
waiting more than `emergency_wait_threshold` seconds in a non-green
direction and forcibly preempts the fixed cycle to serve it immediately —
this is a hard safety guarantee independent of whatever the RL policy or
MUSE fallback strategy currently active would otherwise choose.

## MUSE — metacognitive safety layer

Wraps the trained agent and can override it when:
- **Competence is low + an emergency is present** → prioritize the emergency
- **The situation is highly novel** (state far outside training
  distribution) → fall back to a simple queue-comparison rule, or flag for
  human review at the most extreme novelty
- **All 4 directions are simultaneously heavy (gridlock)** → switch to a
  genuine fixed-timer cycle (ignores queue state entirely) until the
  gridlock clears, then hand back to RL automatically

**Usage note**: MUSE requires calling both `muse.act(state)` **and**
`muse.observe(state, action, reward, next_state, done)` every step — only
calling `act()` silently prevents its internal calibration (including the
gridlock detector) from ever engaging. `--muse` must be passed explicitly to
`ejecutar_entrenamiento_pro.py`; it is off by default.

## Results (representative — see `testbench_rl_vs_static.py` for exact,
reproducible numbers)

Trained agent vs. a fixed-duration timer, same traffic, same random seed,
real SUMO `tripinfo` and emissions output (not hand-calculated):

| Scenario | Waiting time | CO2 per vehicle | Throughput |
|---|---|---|---|
| Simple | ~35–70% better | ~12–40% better | on par or slightly higher |
| Hora Punta (oversaturated) | ~33–55% better | ~21–32% better | **higher** — more vehicles served, not just faster per vehicle |
| Traffic Real (real data) | ~42–75% better | ~19–44% better | on par or slightly higher |
| Noche | ~37–75% better | ~18–47% better | on par |

Ranges reflect different checkpoints and different static-timer durations
tested against (20s, 50s) — always genuinely computed, never estimated.

## How to run it

```bash
# Train (single scenario, or comma-separated for interleaved rotation
# across scenarios — this is what actually prevents catastrophic
# forgetting when training on more than one traffic pattern):
python ejecutar_entrenamiento_pro.py --scenario simple,noche,hora_punta,evento --episodes 300

# With MUSE active:
python ejecutar_entrenamiento_pro.py --scenario simple,noche,hora_punta,evento --episodes 300 --muse

# Compare a trained checkpoint against a fixed timer:
python testbench_rl_vs_static.py --scenario hora_punta --model modelos/best_pro_<name>.pt --green_time 50
```

## Key files

| File | Purpose |
|---|---|
| `ejecutar_entrenamiento_pro.py` | Environment + training loop, scenario rotation |
| `algoritmos_avanzados.py` | Dueling DDQN agent (Noisy Nets, PER, N-step) |
| `muse_metacognicion.py` | Metacognitive fallback layer |
| `train_config.yaml` | All hyperparameters and reward weights |
| `intersection_net_3lane.xml` | The intersection network (3 lanes in/out per direction) |
| `generar_escenarios_mangalore.py` | Generates the 4 research-calibrated scenario route files |
| `testbench_rl_vs_static.py` | RL vs. fixed-timer comparison, real SUMO metrics |

## Known limitations — read before trusting this in production

- **Single intersection only.** Nothing here coordinates multiple signals
  (green waves, network-wide routing). An experimental multi-intersection
  extension was prototyped and works mechanically, but is not part of the
  maintained project.
- **Training is not fully converged.** Most checkpoints sit around
  60,000–120,000 gradient steps; performance was still visibly improving at
  that point in every scenario tested.
- **Only `traffic_real` reflects genuine field data.** The other four
  scenarios are calibrated against published research, not measured
  directly at this specific junction.
- **No formal safety guarantees.** This is a Q-learning system — it has no
  worst-case behavioral bounds the way a classical control-theoretic
  approach would. MUSE's gridlock/emergency fallbacks are a practical
  mitigation, not a proof.
- **Turn-movement split (70/20/10 through/left/right)** is a standard
  traffic-engineering assumption, not measured data for this junction.
