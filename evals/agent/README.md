# Whole-agent rubric evals (scenario seeds)

Shippy-style agent evaluation ("Evaluating an agent, not a model"): score the
**whole agent** — tool selection, trajectory order, guardrail adherence,
knowing where to stop — not a single skill's one-shot output (that is what
`evals/skillopt/` does). This directory currently holds the **scenarios and
rubric format only**; the runner + LLM judge are the planned August build.

## Planned pipeline (August)

1. **Runner** drives `LeadAgent.run_stream()` on the exact build under test
   (live local LLM; Studio live or mocked per scenario `requires.studio`) and
   records the full event trajectory (thinking / tool_call / tool_result /
   final).
2. **Judge** — an LLM (Claude or the local Qwen) grades each rubric criterion
   0-1 **with written reasoning**, from the trajectory + final answer.
3. **Score** — weighted aggregate vs the scenario's `pass_threshold`.
4. **Gate** — results feed `evals/skillopt/scripts/regression_gate.py`
   (same summary shape: a `hard` pass-rate over scenarios, a `soft` mean
   weighted score), so a regressing build doesn't ship.

## Scenario format (`scenarios/*.json`)

```jsonc
{
  "id": "kebab-case-unique-id",
  "brief": "the user brief handed to LeadAgent",
  "requires": { "studio": "mock" | "live", "multi_turn": false },
  "history": [],            // optional prior turns for multi-turn scenarios
  "rubric": [
    {
      "criterion": "short name",
      "weight": 0.4,        // weights sum to 1.0 per scenario
      "guidance": "what the judge should check in the trajectory/answer"
    }
  ],
  "pass_threshold": 0.75
}
```

Rubric weights are per-scenario on purpose (Shippy: "every task is graded on
what actually matters for it") — a run-order scenario weights trajectory
correctness heavily; a guardrail scenario weights the boundary itself.

Guardrail scenarios mirror `harness/soul.md`'s **Guardrails** section, so the
soul artifact is *tested*, not just versioned.
