# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Evaluator-only fault seam before invoking the actual run-agent CLI.

The only replacement is the native provider constructor. Motion, feedback,
dispatch, prompts and the model remain production code. No hardware mode.
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

from emet.controller.task.tamp import native_transfer
from emet.eval.native_recovery import RefuseAfterOneTransfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fault-record', type=Path, required=True)
    args, cli = parser.parse_known_args()
    if '--state-only-sim' not in cli or '--native-task-config' not in cli:
        parser.error('fault injection requires explicit state-only native simulation')
    audit = Path(os.environ['EMET_PHYSICAL_AUDIT'])
    original = native_transfer.NativeTableTransferProvider
    def provider(*a, **kw):
        return RefuseAfterOneTransfer(original(*a, **kw), audit, record_path=args.fault_record)
    native_transfer.NativeTableTransferProvider = provider
    sys.argv = ['emet.app.run_agent', *cli]
    runpy.run_module('emet.app.run_agent', run_name='__main__')


if __name__ == '__main__':
    main()
