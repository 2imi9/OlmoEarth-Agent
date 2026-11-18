# Limits

What the OlmoEarth Agent cannot tell you, in short. Each line holds for both apps unless it names one. The details
are in [Usage.md](Usage.md).

- Without labels, the tools tell you where to look, not how wrong the map is. The windows they put first (a review
  set) are chosen to hold errors, not drawn at random, so labelling only them overstates the error rate.
- Every interval (the range, on the panels) and every certified zone assumes the labels are right.
- Some tools read a Studio result on a grid of sampled points (at most 16 x 16), so their figures cover those points,
  not every pixel. For a map with scores, reading its own files (`olmoearth_scores_from_studio`, as the MCP App's
  Check job does) covers every valid window.
- Comparing results uses no ground truth: a series over dates estimates movement, not verified change on the ground.
- A certificate adds no guarantee: its labels, the map's date and the single draw are the issuer's word.
- By default a host that is not on the allowed list (the egress allowlist) is only logged, not refused. Beyond your
  own machine, set `OLMOEARTH_EGRESS=enforce` (in the MCP App, in its [env block](mcp-app-setup.md#options)). The
  list does not cover the olmoearth-inferencex command line the tools start ([security](Usage.md#security)).
- MCP App: nothing checks Claude's replies against the tool results. Its own limits:
  [mcp-app-setup.md](mcp-app-setup.md#limits).
