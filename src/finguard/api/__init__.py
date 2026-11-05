"""FinGuard API and dashboard (Phase 5, Project Report v3.0 sections 4, 9 and 10).

* ``service``   the endpoints' logic and their JSON-schema contract (``ROUTES``), framework-free;
* ``app``       the ASGI app: FastAPI (docs at ``/docs``), or Starlette when FastAPI is absent;
* ``world``     where evidence comes from: the real data, model and index, or a synthetic demo;
* ``synthetic`` the synthetic demo world (CI deployment gate, first look without the data);
* ``client``    standard-library client: Docker health check and the timed alert-to-report demo;
* ``static/dashboard.html``  the analyst dashboard: alert queue, case view, assistant, cases.

Run ``python -m finguard.api --help``.
"""
