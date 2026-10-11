# Reviewer prompts

Launch each reviewer as a **fresh sub-agent** so it does not inherit the author's reasoning.
Give it only the file paths listed. Paste its reply verbatim into the oversight log, then
answer each point (accept + change / reject + reason).

## plan

> You are an adversarial reviewer for a data-mining plan. Read `<RUN>/profile.md`
> (facts about the dataset) and `<RUN>/plan.json` (the decisions), and
> `<SKILL_DIR>/references/decision-rules.md` (the house rules).
> Do not run any code. Report, as a numbered list with severity (blocker / major / minor):
> 1. Any profiler flag that the plan does not address.
> 2. Any decision whose `why` is not supported by the evidence in profile.md, or that
>    contradicts the house rules without saying why.
> 3. Possible leakage: anything fitted on data it should not see, any column that could
>    encode the outcome.
> 4. Whether the two models form an informative comparison, and whether search spaces,
>    budgets and stopping criteria are reasonable for this data size.
> 5. Missing ablations: decisions that are asserted rather than tested.
> Be concrete (column names, numbers). Say "no issues" for a category if there are none.
> Keep it under 400 words.

## report

> You are verifying a generated two-page report. Files: `<RUN>/report.tex`,
> `<RUN>/numbers_audit.md`, `<RUN>/results.json`, `<RUN>/profile.json`, `<RUN>/plan.json`.
> Do not edit files. Report, as a numbered list with severity:
> 1. **Numbers**: verify every literal number listed in numbers_audit.md against the JSON
>    files; independently re-derive at least three table values from results.json
>    (name the JSON path you used). List any mismatch.
> 2. **Claims**: any sentence that overstates the evidence (significance claimed without
>    p < 0.05 and a CI excluding 0; causal language; generalisation beyond the data).
> 3. **Coverage** against the rubric: data exploration and cleaning, feature
>    selection/engineering, model training (configuration, tuning, stopping criteria),
>    evaluation and comparison, findings/discussion. Name anything missing or shallow.
> 4. **Reasoning gaps**: decisions stated without justification, alternatives not
>    discussed, ablation results that contradict the text.
> 5. Header metadata present: matric, name, "IN6227-Assignment-1", variant, model name
>    and version, LLM interface, repository link.
> Keep it under 500 words.
