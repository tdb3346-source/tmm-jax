# TMM-JAX Log

## Kill-checks (must pass before any result is trusted)

1. light either bounces off  moves through or absorbs inside and they must sum to 1
2. glass should output 0.04 which is 4% that glass does 
3. zero thickness layer should give same result as if the layer wasn't there at all

## PHASE 0 COMPLETE — all 5 steps, 2 problems

Validated inverse-modeling machinery on toy line problem (inv-modeling),
then ported to TMM thin-film optics (tmm-jax).

Step 1 likelihood landscape    ✅
Step 2 gradient descent          ✅
Step 3 emcee posterior           ✅
Step 4 coverage test             ✅ (line: m=0.915, b=0.920, both PASS — machinery calibrated)
Step 5 port to TMM               ✅ (multi-modal posterior, prediction confirmed)

### Step 5 FINDING (identifiability):
Reflectance spectrum does NOT uniquely determine film thickness.
Three thicknesses (~67, ~150, ~365 nm) all produce near-identical
reflectance — data is consistent with all three, cannot distinguish them.
Gradient descent HID this (reported one valley, depending on start).
Posterior REVEALED it (three humps, all at once, from spread-out walkers).
Honest answer to "what thickness?" = "67 OR 150 OR 365", not one number.

Key technical choice: walkers initialized SPREAD across prior (uniform 20-450),
NOT clustered at one valley — this is what let the posterior see all three modes.

### PARKED for Phase 1 (do NOT do now):
- Multi-modal coverage test on d (standard percentile interval breaks on
  multi-modal posterior — needs per-mode treatment). Phase 1 question, not Phase 0.