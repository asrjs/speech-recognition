# CRITICAL EDITING, DEBUGGING & REGRESSION SAFETY RULES

These rules apply to ALL coding agents, regardless of model quality.

The objective is not merely to tell agents to "be careful".
The repository must remain safe even when a weaker model makes poor decisions.

Prefer surgical edits, measurable evidence, deterministic validation,
and rollback over autonomous improvisation.

==================================================

1. SURGICAL EDITING ONLY

Never repair a failed patch by immediately stacking another blind replacement
on top of it.

Prefer:

READ -> understand exact code/API -> make ONE minimal localized edit ->
inspect diff -> validate -> continue

over:

patch -> patch again -> syntax breaks -> patch again -> rewrite file

After EVERY source edit:

1. inspect `git diff` for the affected file;
2. run the appropriate syntax/type/static check;
3. run the smallest relevant targeted test;
4. only then continue.

Where patch tooling supports a preflight/check mode, use it before applying
non-trivial patches.

Do not assume an edit succeeded merely because the editing tool returned
success.

==================================================

2. FAILED EDIT = REVERT BEFORE RETRY

If an edit causes syntax failure, type failure, malformed source, duplicated
blocks, unexpected diff, or unrelated changes: REVERT THAT EDIT before
attempting another solution.

Do not build additional fixes on top of a known-bad working tree.

A failed patch is evidence that an assumption may be wrong.
Re-read the source and reconsider the assumption before editing again.

==================================================

3. MAXIMUM TWO FAILED EDIT ATTEMPTS

Maximum two failed edit attempts on the same file/location for the same issue.

After the second failure: STOP EDITING. Then:

1. restore the file to the last known-good state;
2. inspect the exact relevant source;
3. inspect the real API/function signature or implementation;
4. identify which assumption was incorrect;
5. formulate a new plan;
6. only then resume editing.

Do NOT continue guess-and-patch loops.

==================================================

4. NO FULL-FILE REWRITE AS ERROR RECOVERY

Never delete, truncate, regenerate, or completely rewrite an existing tracked
source file merely because previous patches accumulated errors.

In particular, do NOT use rm/Remove-Item, delete+recreate, whole-file
overwrite, or a generated replacement file as an easy recovery mechanism for
failed edits.

Whole-file rewrites are allowed only when: the task explicitly requires one,
the file is generated/disposable, or a deliberate architectural rewrite has
been justified after inspection.

"Too many patch problems" is NOT sufficient justification.

When an existing file becomes messy because of your own edits: REVERT.
Do not reconstruct it from memory.

==================================================

5. INSPECT APIS - NEVER GUESS-AND-PATCH

If uncertain about argument names, tensor shapes, return types, state
structures, device placement, library behavior, or version-specific APIs:
inspect the actual implementation first.

Use, where appropriate: function signatures, source code, runtime
introspection, package version, existing tests, authoritative documentation,
known working call sites.

Do NOT repeatedly edit code based on guessed APIs.

Example - wrong:

    guess input_ids -> fail
    guess targets -> fail
    guess another state layout -> fail

Correct:

    inspect decoder.forward signature/source
    -> understand targets/state contract
    -> make one correct patch

==================================================

6. PRESERVE THE ORIGINAL SUCCESS CRITERIA

Never lower, redefine, simplify, or silently change the task's acceptance
criteria because an intermediate step is difficult.

Difficulty is not permission to move the goalposts.

If the requested task is "establish parity through features -> encoder ->
caches -> predictor state -> joiner logits -> tokens -> final text", then
proving only the encoder is NOT completion of the task.

It is: PARTIAL PASS - ENCODER ONLY. Unresolved stages must remain explicitly
unresolved.

Never convert "predictor/joiner failed" into "let us simplify and ship
encoder evidence" unless the USER explicitly changes the scope.

==================================================

7. PARITY MEANS NUMERICAL PARITY

For model-porting/parity work:

"same shape", "reasonable magnitude", "looks correct", "runs successfully"
ARE NOT parity.

A component may only be declared numerically matched after comparing the
reference and target outputs using appropriate quantitative metrics.

Where applicable record: tensor shape, dtype, device, max absolute error,
mean absolute error, relative error, cosine similarity, allclose result with
explicit atol/rtol, first divergent index/time step, representative values.

Example: reference shape [1, 4, 640] and target shape [1, 4, 640] only proves
interface/shape compatibility. It does NOT prove numerical parity.

==================================================

8. MAINTAIN AN EXPLICIT PARITY LADDER

For ASR/model-porting work, track each rung independently:

[ ] preprocessing / features
[ ] encoder first chunk
[ ] encoder continuation
[ ] encoder cache/state
[ ] predictor output
[ ] predictor hidden/state
[ ] joiner logits
[ ] decoding decisions
[ ] token sequence
[ ] timestamps, if applicable
[ ] final text

Each rung should have: reference implementation result, target implementation
result, comparison metric, tolerance, PASS/FAIL/UNTESTED, and an evidence
artifact/log.

Find the EARLIEST divergence.

Do not keep debugging later stages when an earlier stage is already known to
diverge unless there is an explicit reason.

==================================================

9. DIAGNOSTIC CODE MUST ITSELF BE TRUSTWORTHY

Do not declare evidence from a diagnostic/parity script until the script
itself has passed basic validation: syntax check, imports resolve, expected
environment/package versions confirmed, sanity test, deterministic inputs
where practical, output artifacts actually written, device/dtype mismatches
resolved.

A broken diagnostic script cannot prove that production code is broken.

==================================================

10. DO NOT CONFUSE PROGRESS WITH COMPLETION

Agents may report intermediate progress. They must label it accurately.

Allowed:

    Encoder numerical parity: PASS.
    Predictor: BLOCKED by unresolved NeMo state contract.
    Joiner: UNTESTED.
    Overall parity: INCOMPLETE.

Not allowed:

    Encoder succeeded. Updating goal and shipping.

when predictor/joiner/token/text parity were part of the requested goal.

==================================================

11. PROTECT KNOWN-GOOD WORK

Before non-trivial or risky edits: inspect git status, identify unrelated
user changes, preserve them, establish the relevant last-known-good state.

Never discard unrelated user work.
Never use broad reset/clean commands that may destroy user changes.
When possible, create small reversible checkpoints for risky work.

==================================================

12. DIFF / BLAST-RADIUS CHECK

After editing, inspect not only whether the intended lines changed but whether
anything ELSE changed.

A targeted fix should normally produce a targeted diff.

If a small bug fix unexpectedly modifies many unrelated lines, multiple
unrelated files, formatting across a file, generated artifacts, configuration,
or unrelated tests: STOP and investigate before continuing.

Large unexpected diffs are a regression signal.

==================================================

13. TEST THE SMALLEST THING FIRST

Validation ladder:

syntax/type check
  -> small targeted test
  -> component test
  -> integration test
  -> full regression suite
  -> benchmark/performance validation

Do not repeatedly launch an expensive full pipeline to discover basic syntax
or API errors that a smaller test could catch immediately.

==================================================

14. PERFORMANCE WORK REQUIRES BASELINES

For optimization tasks: never claim an optimization without a known baseline,
same or equivalent workload, correctness validation, measured before/after
result, and a regression check.

Do not trade correctness for speed unless explicitly requested.

Track latency, throughput/RTFx, memory/VRAM, initialization/loading cost where
relevant, and numerical/accuracy impact.

A faster incorrect implementation is a regression.

==================================================

15. ROOT-CAUSE BEFORE WORKAROUND

Prefer identifying why something fails over accumulating compatibility hacks.

When an error appears: OBSERVE -> localize -> inspect actual contract -> form
hypothesis -> run discriminating test -> fix root cause -> validate.

Do not create a growing stack of special cases merely to make the latest error
disappear.

==================================================

16. TOOL FAILURE SAFETY

Tool-call failures are not permission to repeat indefinitely.

If the same tool/edit approach fails twice: STOP using that approach.

Investigate invalid arguments, model/tool schema mismatch, harness/provider
incompatibility, shell/platform quoting, path assumptions, and
environment/version mismatch.

Do not generate hundreds of increasingly speculative tool calls.

==================================================

17. MODEL/HARNESS ISSUES MUST BE DISTINGUISHED FROM CODE ISSUES

When behavior is suspicious, consider whether the failure belongs to:
repository/application code, dependency/API contract, environment/version,
tool/harness integration, or model behavior.

Do not automatically mutate repository code to compensate for a harness or
tool-protocol problem.

Where practical, reproduce suspicious behavior using a minimal isolated test.

==================================================

18. DO NOT EDIT TESTS TO MAKE IMPLEMENTATION PASS

While debugging implementation correctness, do not weaken assertions,
tolerances, expected outputs, parity thresholds, or benchmark requirements
merely to obtain a passing result.

If the test itself is proven incorrect, document the evidence before changing
it.

Never modify both implementation and its acceptance criterion in one opaque
step.

==================================================

19. FAIL CLOSED ON UNCERTAINTY

If critical evidence is unavailable or contradictory: STOP and report the
blocker. Do not assume success.

Examples: reference output unavailable, API contract unresolved, diagnostic
script unreliable, expected tensor semantics unclear, test environment
inconsistent, unexplained large diff, benchmark cannot be reproduced.

Use BLOCKED / UNVERIFIED instead of "probably correct".

==================================================

20. ESCALATION RULE

When repeated local attempts do not resolve the problem: do NOT respond by
making increasingly broad edits.

Escalate reasoning, not blast radius.

Re-read goal, relevant implementation, reference implementation, APIs, and
previous evidence. Then produce: what is proven, what remains unknown, the
earliest unresolved divergence, likely hypotheses, and the smallest
discriminating experiment.

Only continue after choosing the experiment most likely to reduce uncertainty.

==================================================

21. COMPLETION REPORT

Before marking a technical task complete, report: original success criteria,
which criteria passed, which remain unresolved, files changed, validation/tests
run, numerical evidence where applicable, performance before/where applicable,
known limitations, and remaining risks.

Never represent partial progress as complete work.

==================================================

22. CORE OPERATING PRINCIPLE

The safest workflow should also be the easiest workflow:

READ -> VERIFY ASSUMPTIONS -> PLAN -> MINIMAL EDIT -> DIFF ->
SYNTAX/TYPE CHECK -> TARGETED TEST -> MEASURE -> CONTINUE

Never:

GUESS -> PATCH -> PATCH THE PATCH -> BREAK FILE -> REWRITE FILE ->
LOWER SUCCESS CRITERIA -> SHIP

A weaker model must be contained by the workflow.

Correctness must come from evidence and automated validation, not from the
agent believing that its own work is correct.
