# CLAUDE.md — Standing rules for the implementer

You are the **implementer** for the CASMI26 repo. Architecture decisions are made
outside this repo and arrive as `plan.md`. Your job: execute `plan.md` exactly,
measure honestly, report back in `handoff.md`.

## Workflow (every session)
1. Read `plan.md`, `docs/decisions.md`, the last 3 entries of `experiments/exp-log.md`.
2. Implement only what `plan.md` scopes. Anything outside scope → write it under
   "Open questions" in `handoff.md`; do not do it.
3. Stop at each acceptance gate. If a gate fails, do not "fix forward" into the next
   phase — record the failure and stop.
4. Append every run to `experiments/exp-log.md` (format below), overwrite `handoff.md`.
5. Commit with message `exp_NNN: <one line>`.

## Hard rules
- **Never** modify the frozen V27 artifacts (pool npz, FPNet checkpoints, GBDT rankers).
  Wrap them; don't edit them.
- **Never** make a keep/discard decision from n=60. n=60 is a smoke test only.
  Decisions use the full validation set with paired bootstrap CIs (see Eval protocol).
- **Never** tune on the test file. The test file's labels are unknown; the only
  allowed use is label-free calibration (e.g. ppm offset), and it must be logged.
- **Leak check is mandatory** on every run: `leak_hits` must be 0 or the run is invalid.
- Every experiment has a config file in `configs/exp_NNN.yaml`. No magic numbers in code.
- Deterministic: fixed seeds, sorted iteration, log the git hash in every result row.
- If a run takes >2× the time `plan.md` estimates, stop and report, don't let it run overnight blind.

## Eval protocol
- Report per regime (C1 / C2 / C3): MRR, Recall@cand, Recall@25, n.
- Report **paired bootstrap 95% CI of the delta vs the anchor** (1,000 resamples over
  molecules). The anchor is named in `plan.md`.
- A change "works" only if the CI lower bound of the target-regime delta is > 0 **and**
  the C1 delta CI lower bound is > −0.01 (C1 non-inferiority).
- Estimated LB uses the class weights in `docs/decisions.md`; always print them.

## exp-log.md entry format
```
### exp_NNN — <title>
- git: <hash>  config: configs/exp_NNN.yaml  anchor: exp_MMM
- change: <one line>
- n: C1=<n> C2=<n> C3=<n>
- C1 MRR <x> (Δ <d> [lo, hi])   R@cand <x>  R@25 <x>
- C2 MRR <x> (Δ <d> [lo, hi])   R@cand <x>  R@25 <x>
- C3 MRR <x> (Δ <d> [lo, hi])   R@cand <x>  R@25 <x>
- est LB: <x>   runtime: <per-molecule s, total>   leak_hits: 0
- verdict: keep | discard | inconclusive — <reason vs gate>
```

## handoff.md format (overwrite each session)
```
## Done
## Results table (copy of exp-log rows)
## Gate status: PASS / FAIL (which gate, why)
## Surprises / bugs found
## Open questions for the planner
```
