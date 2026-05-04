"""Create a Chroma collection ingested with the FIXED-SIZE chunking baseline.

In the original workshop, "custom" meant the AWS Lambda namespace-aware chunker —
the *improved* alternative to fixed-size. In the local stack, the default KB
created by Module 01 already uses the equivalent (markdown_header chunker), so
this script supplies the *fixed-size baseline* needed by evaluate_chunking.py
to compare our default against the dumb-but-fast strategy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modules.local.kb import LocalKB  # noqa: E402

print("Creating fixed-size baseline collection (so we can compare against the smart chunkers)...")
kb = LocalKB("workshop-kb-fixed-size", chunker="fixed_size")
kb.reset()
n = kb.ingest_data_dir()
print(f"✅ Fixed-size baseline 'workshop-kb-fixed-size' ready: {n} chunks")
