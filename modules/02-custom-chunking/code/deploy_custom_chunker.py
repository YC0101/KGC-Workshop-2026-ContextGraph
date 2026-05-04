"""In the local stack there is no Lambda to deploy.

The "custom chunker" is the markdown_header strategy in modules/local/chunking.py;
it runs in-process. The default KB created by modules/01-knowledge-base-setup
already uses it, so this script is a no-op kept for parity with the original
workshop step list.
"""
print("Local stack: custom chunker is markdown_header in modules/local/chunking.py")
print("It runs in-process — no Lambda deployment needed.")
print("The default workshop-kb collection already uses this strategy.")
