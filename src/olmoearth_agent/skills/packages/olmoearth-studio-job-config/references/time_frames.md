# Temporal context — three modes, picked by what the label represents

Studio's "What time frame is important context for this model?" step offers three modes. The decision is about *the label*, not the imagery. Pick by the nature of the thing you're predicting; the imagery follows.

## Mode A: a period of time
The label describes a **span**, not a moment.

**Pick when**: crop type for a year, mangrove extent for a season, ecosystem type for a year, annual land cover — and **stable landscape properties** (geology, aquifer vulnerability, soil class): any one scene "contains" them, but the seasonal stack is the signal, so they are Mode A with 12 months, not a single moment.

**Then choose period length**:

| Length | Use for |
|--------|---------|
| 3 months | Crop growth stages, irrigation cycles, quarterly inference |
| 6 months | Ecosystem state shifts, wet/dry transitions, vegetation structure change |
| 12 months | Annual land cover, long-term ecosystem condition |
| Custom (1–12) | Domain-specific cycles (e.g., a 4-month sugarcane ratoon) |

**And start months**: which calendar months can the period begin in?

- Northern-hemisphere agriculture: March or April (spring) — period covers growing season
- Southern-hemisphere agriculture: September or October
- Year-round / continuous targets (land cover, mangrove): all 12 months as valid starts
- Aligned to fiscal/management cycles: pick the start of the cycle (e.g., July for some US agencies)

If unsure, default to all 12 months. If the user mentions any seasonal logic, narrow the list.

## Mode B: single moment with before-and/or-after context
The label is about a **specific date** but needs surrounding imagery to be predictable.

**Pick when**:

- Soil moisture or drought indicators — need *before* context (preceding months show drying trend)
- Flood damage or forest-loss cause — need *after* context (following months show recovery / persistence)
- Phenology-anchored events — need both
- Post-event change detection — *after* only, with a small offset gap to avoid the event itself

**Set independently**:

- Before context: 0–12 months
- After context: 0–12 months
- At least one must be > 0

**Offset gap (days)**: ignore imagery very close to the observation date. Typical uses:

- Flood mapping: 3–7 day before-offset to skip cloud-saturated day-of imagery
- Forest-loss attribution: 30 day after-offset to skip immediate-aftermath haze
- Default: 0 (no offset)

## Mode C: a single moment
The label is about **this image, right now**. Transient or moving targets.

**Pick when**:

- Vessel detection (ships move every hour)
- Oil slick (slick visible in one scene, gone the next)
- Active fire / wake / smoke plume
- Anomaly detection where the *moment* is the signal

**Set observation window** (symmetric, in hours):

| Window | Use for |
|--------|---------|
| ±12 h (1 day) | **Default.** Vessel detection, oil slicks where S1 + S2 both have a daily revisit somewhere. |
| ±24 h | When the target is sparser in time. |
| ±48–60 h | When imagery is sparse (e.g., specific S1 footprint at high latitudes). |

**Critical**: training-label timestamps must match what you'll predict on. If labels say "2024-03-15 14:32 UTC" and inference imagery is the closest S2 pass (could be 12 h off), your training and inference temporal alignment must match. Pitfall: train on perfectly-aligned labels (±0 h) but deploy on imagery that's hours stale → distribution shift.

## Disambiguation prompts

When the user is vague, ask:

1. "Does your label describe a state over time (e.g., 'this is a corn field' — true for a whole season) or a moment (e.g., 'a ship was here at 14:32')?"
2. If state-over-time → Mode A. "How long is that span?"
3. If moment → "Does the model need to see *before* or *after* to make the prediction? (Soil moisture trend needs before. Flood cause needs after. Oil slick at the moment needs neither.)"
4. If before/after needed → Mode B with appropriate context.
5. If neither → Mode C, observation window ±12 h default.

## Anti-patterns

- **Single moment for a stable landscape property** — geology, aquifer vulnerability, soil class don't change with the scene; a ±36 h window throws away the seasonal signal. Mode A, 12 months.
- **12-month period for vessel detection** — vessels are transient; using a 12-month composite blurs them out. Mode C.
- **Mode C for crop type** — crop type isn't a moment; you need within-season phenology. Mode A, 3–12 months.
- **Mode B with 0 before / 0 after** — Studio requires at least one to be non-zero. If both are 0, switch to Mode C.
- **Offset gap > period length** — nonsensical; the model would see no imagery.
- **Annual period start = December for Northern-hemisphere agriculture** — the period would cover the dormant season and miss the growing season. Pick March/April for NH, Sep/Oct for SH.
