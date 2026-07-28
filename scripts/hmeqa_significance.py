#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
"""Shim — prefer: uv run emet hmeqa significance"""
from emet.eval.hmeqa_significance import main

if __name__ == "__main__":
    raise SystemExit(main())
