"""Run an ONNX session, optionally through IO binding.

CUDA graph capture is a provider option set when the session is created.
This helper only changes how inputs are fed. Without a CUDA provider it uses
the ordinary ``session.run`` path, which is what CI executes.
"""

from __future__ import annotations

import logging

import numpy as np

from barrett_faceswap.i18n import t

logger = logging.getLogger(__name__)
_warned = False


def session_provider_names(session) -> list[str]:
    """Friendly provider names currently attached to ``session``."""

    names = []
    for item in session.get_providers():
        names.append(item if isinstance(item, str) else str(item))
    return names


def cuda_available(session) -> bool:
    return "CUDAExecutionProvider" in session_provider_names(session)


def run_feeds(session, feeds: dict[str, np.ndarray], *, cuda_graph: bool) -> list:
    """Execute ``feeds``. IO binding is used only when CUDA is really present."""

    global _warned
    if not cuda_graph or not cuda_available(session):
        if cuda_graph and not _warned:
            logger.info(t("execution.cuda_graph_skipped"))
            _warned = True
        return session.run(None, feeds)
    binding = session.io_binding()
    for name, value in feeds.items():
        binding.bind_cpu_input(name, np.ascontiguousarray(value))
    for item in session.get_outputs():
        binding.bind_output(item.name, "cuda")
    session.run_with_iobinding(binding)
    return binding.copy_outputs_to_cpu()
