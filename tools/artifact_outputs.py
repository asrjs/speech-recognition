"""Model-neutral preflight for new diagnostic artifacts; preserve source data."""

from pathlib import Path


def require_new_outputs(outputs, *, protected_inputs=(), protected_directories=()):
    """Reject existing targets, aliased outputs, inputs and acquisition directories.

    Call before inference/downloads; use exclusive file creation at write time
    as well, since another process can create a target after this check.
    """
    original_paths = [Path(value) for value in outputs if value is not None]
    paths = [path.resolve() for path in original_paths]
    inputs = {Path(value).resolve() for value in protected_inputs if value is not None}
    directories = [Path(value).resolve() for value in protected_directories]
    if len(set(paths)) != len(paths):
        raise ValueError("Output paths must be distinct")
    for original, path in zip(original_paths, paths):
        if path in inputs or any(path == directory or directory in path.parents for directory in directories):
            raise ValueError(f"Output must not overwrite a protected input or source directory: {path}")
        if path.exists() or original.is_symlink():
            raise FileExistsError(f"Refusing existing output: {path}")


def write_new_text(path, text):
    """Exclusive UTF-8 receipt write; never truncate a prior file after a race."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
