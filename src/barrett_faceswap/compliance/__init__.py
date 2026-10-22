"""Compliance package.

Importing these modules does not turn enforcement on. The pipeline calls
:class:`barrett_faceswap.compliance.policy.CompliancePolicy` around every run.
The policy enforces consent, watermarking, the content filter, and the audit
log only when ``compliance.enabled`` is true or the CLI passes
``--enable-compliance``.
"""

from __future__ import annotations

from barrett_faceswap.compliance.policy import CompliancePolicy

__all__ = ["CompliancePolicy"]
