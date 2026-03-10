"""
depthforge.web
==============
Local Flask preview server for DepthForge.

Usage
-----
    from depthforge.web.app import create_app, run_server

    # Start the server:
    run_server(host="127.0.0.1", port=5000)

    # Or get the app object for testing:
    app = create_app()
    client = app.test_client()
"""

from depthforge.web.app import create_app, run_server

__all__ = ["create_app", "run_server"]
