# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Actionable, pre-grasp navigation rejection; no grasp has been attempted."""


class GraspWorkspaceError(RuntimeError):
    def __init__(self, diagnostics: dict):
        self.diagnostics = dict(diagnostics)
        super().__init__(
            "Could not reach a collision-checked grasp workspace: "
            + str(self.diagnostics.get("status", "navigation_failed"))
        )
