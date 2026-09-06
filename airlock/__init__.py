"""Airlock -- a calibrated privacy gate in front of hosted language models."""

# Host-compatibility shim, deliberately first. Where the interpreter's system
# libstdc++ is older than the one pyarrow's bundled libarrow requires, pyarrow
# can only be imported before torch loads the system copy. transformers imports
# scikit-learn, which imports pyarrow, so the order decides whether the process
# starts at all. A no-op where pyarrow is absent or where the two agree.
try:  # pragma: no cover - environment dependent
    import pyarrow  # noqa: F401
except Exception:  # pragma: no cover
    pass

__version__ = "0.1.0"
