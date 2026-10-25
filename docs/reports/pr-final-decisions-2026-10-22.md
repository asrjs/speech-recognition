# PR decisions — 2026-10-22

72 live open PRs fetched individually. HOLD is an explicit no-merge decision, not PASS or completed validation. Prior agent reports contained incorrect claims about #186/#212/#218; this record supersedes those claims. Diff hashes and head SHAs are in JSON. Four earlier PRs (#182/#226/#227/#234) already closed.

| PR | Decision | Evidence / next requirement |
|---|---|---|
| #233 | REJECT AS-IS | Cosmetic/deprecation edits bundle duplicate lint changes; no demonstrated runtime fix. |
| #232 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |
| #231 | REJECT AS-IS | Cosmetic/deprecation edits bundle duplicate lint changes; no demonstrated runtime fix. |
| #230 | REJECT AS-IS | Cosmetic/deprecation edits bundle duplicate lint changes; no demonstrated runtime fix. |
| #229 | HOLD TEST ADAPTATION | Useful test candidate, not yet run here; #229 bundles unrelated lint edits, #191 needs exception-safe spy restoration. |
| #228 | ADAPTED LOCALLY | Disposal idea adapted in 8cd68b26 with failing-before/passing-after reset regression; broader init cases not certified. |
| #220 | ADAPTED LOCALLY | Disposal idea adapted in 8cd68b26 with failing-before/passing-after reset regression; broader init cases not certified. |
| #219 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |
| #218 | HOLD | for-in replacement includes inherited enumerable keys unlike Object.keys; no ownership/semantic regression test or measured gain. |
| #217 | ACCEPTED LOCALLY | Ported and independently tested; full suite passed. Remote integration waits on failing root-import gate. |
| #216 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |
| #215 | ACCEPTED LOCALLY | Ported and independently tested; full suite passed. Remote integration waits on failing root-import gate. |
| #214 | ACCEPTED LOCALLY | Ported and independently tested; full suite passed. Remote integration waits on failing root-import gate. |
| #213 | ACCEPTED LOCALLY | Ported and independently tested; full suite passed. Remote integration waits on failing root-import gate. |
| #212 | HOLD | Internal single-chunk ownership change only; NOT the stream-cache bug. Require ownership and benchmark validation. |
| #211 | REJECT | Cache retains yielded mutable buffer after stream completion; preserve independent snapshot. |
| #210 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |
| #209 | REJECT | Cache retains yielded mutable buffer after stream completion; preserve independent snapshot. |
| #208 | ADAPTED LOCALLY | Disposal idea adapted in 8cd68b26 with failing-before/passing-after reset regression; broader init cases not certified. |
| #207 | REJECT | Cache retains yielded mutable buffer after stream completion; preserve independent snapshot. |
| #206 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |
| #205 | ADAPTED LOCALLY | Disposal idea adapted in 8cd68b26 with failing-before/passing-after reset regression; broader init cases not certified. |
| #204 | HOLD ONE CONSOLIDATED PATCH | TypedArray.set idea plausible; duplicate variants should not all merge; numerical equivalence and measured benefit not established. |
| #203 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |
| #202 | REJECT | Fallback confidence 0.9 invents evidence; unknown confidence must remain unknown. |
| #201 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |
| #200 | HOLD ONE CONSOLIDATED PATCH | TypedArray.set idea plausible; duplicate variants should not all merge; numerical equivalence and measured benefit not established. |
| #199 | REJECT | Cache retains yielded mutable buffer after stream completion; preserve independent snapshot. |
| #198 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |
| #197 | HOLD ONE CONSOLIDATED PATCH | TypedArray.set idea plausible; duplicate variants should not all merge; numerical equivalence and measured benefit not established. |
| #196 | HOLD ONE CONSOLIDATED PATCH | TypedArray.set idea plausible; duplicate variants should not all merge; numerical equivalence and measured benefit not established. |
| #195 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |
| #194 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |
| #193 | REJECT | Cache retains yielded mutable buffer after stream completion; preserve independent snapshot. |
| #192 | HOLD ONE CONSOLIDATED PATCH | TypedArray.set idea plausible; duplicate variants should not all merge; numerical equivalence and measured benefit not established. |
| #191 | HOLD TEST ADAPTATION | Useful test candidate, not yet run here; #229 bundles unrelated lint edits, #191 needs exception-safe spy restoration. |
| #190 | HOLD ONE CONSOLIDATED PATCH | TypedArray.set idea plausible; duplicate variants should not all merge; numerical equivalence and measured benefit not established. |
| #189 | HOLD ONE CONSOLIDATED PATCH | TypedArray.set idea plausible; duplicate variants should not all merge; numerical equivalence and measured benefit not established. |
| #188 | HOLD | streamToBytes and path normalization changes; NOT proven cache-write alias bug. Compare current implementation before port. |
| #187 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |
| #186 | CLOSE EMPTY | Fresh gh pr diff returned no changes; no build-defect claim. |
| #185 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |
| #184 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |
| #183 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |
| #181 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |
| #180 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |
| #167 | REJECT AS-IS | Stale alias variants: current full suite resolves imports; no demonstrated missing resolution needing this patch. |
| #166 | REJECT AS-IS | Stale alias variants: current full suite resolves imports; no demonstrated missing resolution needing this patch. |
| #165 | REJECT AS-IS | Stale alias variants: current full suite resolves imports; no demonstrated missing resolution needing this patch. |
| #164 | REJECT AS-IS | Stale alias variants: current full suite resolves imports; no demonstrated missing resolution needing this patch. |
| #163 | HOLD ONE CONSOLIDATED PATCH | TypedArray.set idea plausible; duplicate variants should not all merge; numerical equivalence and measured benefit not established. |
| #162 | REJECT AS-IS | Stale alias variants: current full suite resolves imports; no demonstrated missing resolution needing this patch. |
| #161 | REJECT AS-IS | Stale alias variants: current full suite resolves imports; no demonstrated missing resolution needing this patch. |
| #160 | HOLD ONE CONSOLIDATED PATCH | TypedArray.set idea plausible; duplicate variants should not all merge; numerical equivalence and measured benefit not established. |
| #159 | HOLD ONE CONSOLIDATED PATCH | TypedArray.set idea plausible; duplicate variants should not all merge; numerical equivalence and measured benefit not established. |
| #158 | HOLD ONE CONSOLIDATED PATCH | TypedArray.set idea plausible; duplicate variants should not all merge; numerical equivalence and measured benefit not established. |
| #157 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |
| #156 | REJECT AS-IS | Stale alias variants: current full suite resolves imports; no demonstrated missing resolution needing this patch. |
| #155 | REJECT AS-IS | Stale alias variants: current full suite resolves imports; no demonstrated missing resolution needing this patch. |
| #154 | HOLD ONE CONSOLIDATED PATCH | TypedArray.set idea plausible; duplicate variants should not all merge; numerical equivalence and measured benefit not established. |
| #153 | REJECT AS-IS | Stale alias variants: current full suite resolves imports; no demonstrated missing resolution needing this patch. |
| #152 | REJECT AS-IS | Stale alias variants: current full suite resolves imports; no demonstrated missing resolution needing this patch. |
| #151 | REJECT AS-IS | Stale alias variants: current full suite resolves imports; no demonstrated missing resolution needing this patch. |
| #150 | HOLD ONE CONSOLIDATED PATCH | TypedArray.set idea plausible; duplicate variants should not all merge; numerical equivalence and measured benefit not established. |
| #149 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |
| #148 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |
| #147 | HOLD FOR TARGETED ADAPTATION | Keep canonical graph-name fix candidate; reject machine-specific /app fixture path. Converter runtime not validated. |
| #146 | HOLD TEST ADAPTATION | Useful test candidate, not yet run here; #229 bundles unrelated lint edits, #191 needs exception-safe spy restoration. |
| #145 | REJECT | Exported synchronous loader becomes async; benchmark/callers need awaiting. Breaking API change rejected as written. |
| #144 | HOLD TEST ADAPTATION | Useful test candidate, not yet run here; #229 bundles unrelated lint edits, #191 needs exception-safe spy restoration. |
| #143 | HOLD TEST ADAPTATION | Useful test candidate, not yet run here; #229 bundles unrelated lint edits, #191 needs exception-safe spy restoration. |
| #142 | HOLD | No merge approval: behavior/performance evidence is insufficient; evaluate as a separate narrow change. |

Validation: 1845 passed /36 skipped; lint 0 errors/13 warnings; typecheck/build/Node smoke/package smoke passed. Build needed sandbox permission retry. Root import still exit 2 (78 vs <45). No push or merge because user conditioned publication on absence of problems.
