import sys

MIN_PYTHON = (3, 11)

if sys.version_info < MIN_PYTHON:
    sys.stderr.write(
        "DisputeDesk requires Python %s.%s+ (found %s).\n"
        "Use the project venv:\n"
        "  python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'\n"
        % (MIN_PYTHON[0], MIN_PYTHON[1], sys.version.split()[0])
    )
    sys.exit(1)
