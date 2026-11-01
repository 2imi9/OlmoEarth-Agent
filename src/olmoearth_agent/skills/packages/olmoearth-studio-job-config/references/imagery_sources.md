# Image sources — start with S2, add S1 only when it helps

Studio's "Which satellite imagery sources should the model use?" step lists three modalities (Landsat shown but not yet selectable). The default and right answer for most tasks is **Sentinel-2 alone**.

| Source | Available in Studio | Strengths | Weaknesses |
|--------|---------------------|-----------|------------|
| **Sentinel-2 (optical)** | Yes | 10–60 m multispectral, 5-day revisit, broad spectral range (RGB + NIR + SWIR + red-edge) | Blocked by clouds; no signal at night |
| **Sentinel-1 (radar)** | Yes | Cloud-penetrating, day/night, sensitive to surface roughness, moisture, structure | Speckle noise; spectrum-blind (no color); harder to interpret |
| **Landsat (optical)** | Not yet | Long historical record (Landsat 8: 2013–; thermal band) | 30 m resolution, 16-day revisit, similar cloud limitation to S2 |

## When to add Sentinel-1

Add S1 to S2 only if at least one of these applies:

1. **Cloud cover problem** — task region has >30% annual cloud-cover climatology (tropics, monsoon-affected, polar). S1 fills the gaps.
2. **Texture / structure signal** — the target is detectable by *roughness* not *color*: oil slicks (smooth water = dark), ship wakes (turbulence trail), soil moisture (dielectric constant), water-surface roughness, ice-surface change.
3. **Forest structure or biomass** — radar penetrates canopy; useful for above-ground biomass and forest disturbance.
4. **Night / dark targets** — vessels at night, active fire heat (though Landsat thermal is better).
5. **You've trained S2-only and accuracy plateaued** — S1 is your next dial; add it and re-train.

Don't add S1 because "more bands feels safer". Every added modality:
- Increases training time meaningfully (per Studio's own warning)
- Adds noise that the model has to filter out
- Increases the chance of a temporal alignment issue (S1 and S2 don't pass on the same day)

## When Landsat would help (when it lands in Studio)

- Pre-2015 history (Landsat 8: 2013 onward, Landsat 9: 2021)
- Long-term trend tasks (decadal land-cover change, deforestation history)
- Heat / thermal signals (urban heat islands, fire scars, surface temperature) — Landsat has a thermal band, S2 doesn't
- Cross-sensor consistency: if downstream consumer expects 30 m Landsat data

Until Landsat is available in Studio, route the user to fetch Landsat from Planetary Computer or Google Earth Engine separately and use S2 for the Studio job.

## Per-task quick lookup

| Task | Recommended modalities |
|------|------------------------|
| Crop type | S2 |
| Mangrove extent | S2 (S1 helps in tidal areas) |
| Land cover | S2 |
| Soil moisture | S2 + S1 (S1 is *the* moisture signal) |
| Tree height / canopy | S2 (S1 helps for structure) |
| Biomass | S2 + S1 |
| Vessel detection | S2 + S1 (S1 catches small wakes at night) |
| Solar array detection | S2 (high reflectance is the signal) |
| Oil slick | S1 primary, S2 secondary (slick = smooth water = dark S1) |
| Flood mapping | S2 + S1 (clouds during flood events; S1 essential) |
| Drought indicators | S2 (NDVI, EVI carry the signal) |
| Fire scar / burn area | S2 (SWIR band carries the signal) |
| Active fire detection | S2 (better with thermal — switch to Landsat when available) |
| Wetlands | S2 + S1 |
| Ice / snow extent | S2 (S1 helps for sea ice texture) |
| Embeddings (general purpose) | S2 alone |
| Embeddings (water/wetland clustering) | S2 + S1 |

## Anti-patterns

- **Add S1 without checking if S2 alone solves it** — costs training time you don't need to spend.
- **S1-only for a spectral task** — radar can't see color. Crop type from S1 alone is much harder than from S2.
- **Add S1 because the user said "it's cloudy"** — confirm with cloud-cover climatology numbers, not anecdotes. A region with 40 % monthly cloud cover may still produce a perfectly good seasonal composite from S2.
- **Mix S1 and S2 with a time frame that doesn't match revisit cadence** — S2 has 5-day revisit, S1 ~6-day at mid-latitudes. Mode C with ±12 h may miss one or the other; widen to ±48 h or accept missing pairs.
