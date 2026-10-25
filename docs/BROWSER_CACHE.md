# Browser model cache

ASRJS uses IndexedDB database `asrjs-cache-db`, store `asset-cache`, schema version 2. The name belongs to the library, not an application, framework or package
release. React, Svelte, Vue and plain JavaScript consumers use the same default
asset cache.

## What can share a cache

Applications share this database when they use the same protocol, hostname,
port and browser profile. Different URL paths are fine. `localhost` and
`127.0.0.1`, different ports, different profiles and different sites have separate
storage. A shared library name does not override these browser rules.

The local companion workspace can serve all seven applications at one origin:

```sh
# In frameworks/, after the four demo apps have been built:
npm run build:suite
npm run preview:suite
```

Use `http://127.0.0.1:4190/` and its application links. Standalone `dev:react`,
`dev:svelte`, `dev:vue` and independent demo servers remain useful for development,
but each different origin gets its own storage.

## Artifact identity

For recognized built-in Hugging Face `main` requests, the key is derived from
repository, revision and normalized file path. Loading the same file through a
model-family helper, the generic loader or an exact Hugging Face resolve URL
therefore reuses the same entry. Encoder, decoder, quantization and external-data
files remain distinct when their file paths differ. Backend or framework names
are not added to a file's identity.

Existing generated keys are read as aliases without copying their bytes into a
second entry. Caller-provided keys are left alone. Caller-provided fallback-key
migration retains its explicit behavior. Non-main revisions keep their existing
strict keys; they are not merged with old entries of uncertain revision.
Signed/query URLs are not stripped or treated as equivalent to public URLs.

FireRed's default browser cache writes new entries to the shared database under
its `firered:` resource keys. An existing `firered-vad-web-cache` is read in place
when the browser supports database enumeration. Older browsers without that API
use the shared cache for new downloads. Legacy data is neither copied nor deleted.

Local file/folder inputs are supplied by the user and are not automatically
copied into this persistent model cache. Custom providers can implement their
own cache policy.

## Different websites

A universally shared, permission-free model database across unrelated websites
is not available to a normal browser library. Storage partitioning also means
that embedding a common iframe or using the same CDN does not establish one
shared store. Storage Access API can expose unpartitioned IndexedDB in some
browsers after access is granted, but it is not a portable default.

ASRJS remains installation-free. No local daemon or browser extension is
required. Cross-site sharing is not advertised as supported. See
[IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API),
[storage partitioning](https://developer.mozilla.org/en-US/docs/Web/Privacy/Guides/State_Partitioning)
and [permission-based IndexedDB access](https://developer.mozilla.org/en-US/docs/Web/API/StorageAccessHandle/indexedDB).

## Existing disk usage

This change does not clear caches belonging to other origins or remove existing
model copies. Browser site-storage controls let the user remove a chosen site's
data. Never clear all browser storage as part of model loading.

Sharing persistent entries does not share live model sessions or GPU memory.
Each application owns its loaded model. Concurrent first-time requests can still
fetch before another tab has finished caching; cross-tab single-flight download
coordination is not established by this change. Browser quotas and eviction
remain controlled by the browser.
