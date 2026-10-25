# PR 238: rejected and closed

PR: https://github.com/asrjs/speech-recognition/pull/238

Reviewed head: `37d7b774d0bf86340c2ae8f45b3328605fc79dc9`.
The parent independently reran the source-extracted reproduction on 2026-10-25.

The change removes async callbacks from disposal maps. In the public Parakeet
`assetHandles` path, a synchronous exception from the first handle now aborts
the map before the second handle is disposed. The previous async callback
converts that exception to a rejection while invoking the remaining callbacks.
`ResolvedAssetHandle.dispose()` explicitly allows both synchronous and async
implementations. This is a demonstrated cleanup behavior regression, not a
style preference.

The removals of `Promise.resolve(handle.dispose())` are a distinct, mostly
cosmetic change: their argument was already evaluated synchronously. They do
not fix the regression above, and the PR supplies no measured performance
benefit that justifies adopting this broad cleanup separately now.

Decision: closed without merging; no PR source changes adopted. The GitHub
response confirms `state=closed`, `merged=false`, `merged_at=null`. Its
`merge_commit_sha` field is a generated merge candidate, not proof of a merge.

Evidence: [repro.json](evidence/bot-pr-238-2026-10-25/repro.json).
The reproduction verifies calls `[first, second]` before the change and
`[first]` after it, with the same propagated error in both cases. Full PR tests
and remote CI were not certified; rejection rests on the targeted regression.
