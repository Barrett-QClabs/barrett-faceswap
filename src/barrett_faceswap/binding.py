"""Run an ONNX session, optionally through IO binding.

CUDA graph capture is a provider option set when the session is created.
When the live session has a CUDA provider, feeds go through one reused
IO binding. CPU sessions, including CI, stay on ``session.run``.
"""

from __future__ import annotations

import logging
import weakref

import numpy as np

from barrett_faceswap.i18n import t

logger = logging.getLogger(__name__)
_warned = False
_bindings: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def session_provider_names(session) -> list[str]:
    """Friendly provider names currently attached to ``session``."""

    names = []
    for item in session.get_providers():
        names.append(item if isinstance(item, str) else str(item))
    return names


def cuda_available(session) -> bool:
    return "CUDAExecutionProvider" in session_provider_names(session)


def run_feeds(session, feeds: dict[str, np.ndarray], *, cuda_graph: bool) -> list:
    """Execute ``feeds``. CUDA sessions reuse one IO binding; CPU uses ``session.run``."""

    global _warned
    if cuda_available(session):
        try:
            return _run_bound(session, feeds)
        except Exception:
            logger.debug("IO binding failed", exc_info=True)
            return session.run(None, feeds)
    if cuda_graph and not _warned:
        logger.info(t("execution.cuda_graph_skipped"))
        _warned = True
    return session.run(None, feeds)


def _run_bound(session, feeds: dict[str, np.ndarray]) -> list:
    binding = _reuse_binding(session)
    for name, value in feeds.items():
        binding.bind_cpu_input(name, np.ascontiguousarray(value))
    for item in session.get_outputs():
        binding.bind_output(item.name, "cuda")
    session.run_with_iobinding(binding)
    return binding.copy_outputs_to_cpu()


def _reuse_binding(session):
    try:
        binding = _bindings.get(session)
    except TypeError:
        return session.io_binding()
    if binding is None:
        binding = session.io_binding()
        try:
            _bindings[session] = binding
        except TypeError:
            return binding
        return binding
    binding.clear_binding_inputs()
    binding.clear_binding_outputs()
    return binding
