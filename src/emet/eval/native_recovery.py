# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Explicit evaluator-only refusal between completed native transfers.

This injects an admission failure, not a contact or sensor failure. The wrapped
provider still owns all planning, commands and measured effect verification.
"""

from emet.controller.task.tamp.task_session import TaskExecutionResult


class RefuseAfterOneTransfer:
    def __init__(self, provider, audit_path):
        self.provider, self.audit_path = provider, audit_path
        self.completed = 0
        self.records = []

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def execute(self, operator):
        if self.completed == 1 and not self.records:
            before = self.audit_path.read_bytes() if self.audit_path.exists() else b""
            receipt = TaskExecutionResult(False, False, "task_motion_budget_exhausted")
            after = self.audit_path.read_bytes() if self.audit_path.exists() else b""
            self.records.append({"kind": "injected_between_transfer_admission_refusal",
                                 "completed_before_refusal": self.completed,
                                 "no_commands": before == after})
            return receipt
        receipt = self.provider.execute(operator)
        if isinstance(receipt, TaskExecutionResult) and receipt.success or receipt is True:
            self.completed += 1
        return receipt


def replan_after_boundary(call, rejection, checks, instruction, consumed_ref):
    """A separate preauthorized scripted request, only after the expected stop.

    Rediscovery must admit the measured scene before fresh planning. In the
    native provider this also rejects held or unsettled objects. Never continue
    after an unexpected receipt or an incorrectly reusable failed handle.
    """
    if not checks or not all(row["passed"] for row in checks):
        return None
    rejection("execute_task_plan", ["unknown_plan"], plan_ref=consumed_ref)
    if not all(row["passed"] for row in checks):
        return None
    if call("scene_tasks")["status"] != "ok":
        return None
    plan = call("plan_task", instruction=instruction, strategy="beam")
    if plan["status"] != "ok":
        return None
    return call("execute_task_plan", plan_ref=plan["data"]["plan_ref"])
