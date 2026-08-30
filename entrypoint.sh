#!/bin/sh
set -e

if [ "$TRANSPORT" = "mcp" ]; then
    exec python mcp_server.py
else
    exec uvicorn disputedesk:app --host 0.0.0.0 --port 8000
fi