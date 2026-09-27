#!/usr/bin/env python3
"""Reproduce the strict taxonomy bootstrap from a compact prediction table.

This stable public entry point delegates to the shared strict implementation,
which validates duplicate prediction keys and exact taxonomy/majority coverage
before dataset-equal run-cluster resampling.
"""
from __future__ import annotations

from cluster_taxonomy_bootstrap import main


if __name__ == "__main__":
    main()
