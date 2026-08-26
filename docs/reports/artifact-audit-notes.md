# Artifact audit notes

The local ONNX auditor is intentionally artifact-only. With `--recursive`, it
walks nested model directories but skips `.git`, `.hg`, `.svn`, and
`node_modules`; these directories are repository/dependency state rather than
model inputs and can contain large unrelated LFS objects. When `--output` is
inside the model directory, that report is also excluded from its own input
inventory so repeated audits remain reproducible.
