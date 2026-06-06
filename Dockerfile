# Multi-stage build with a pre-deploy test gate.
#
#   base    — runtime deps + source (shared by test and runtime)
#   test    — adds pytest and runs the suite; a failure here FAILS the build
#   runtime — the lean production image; depends on `test` via COPY --from so a
#             normal `docker compose build ai-router` is forced to run the tests
#             first. pytest is NOT present in the final image.

FROM python:3.12-slim AS base
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p logs data

# ── Test gate ─────────────────────────────────────────────────────────────────
FROM base AS test
RUN pip install --no-cache-dir pytest==8.3.4
RUN python -m pytest -q
RUN touch /app/.tests-passed

# ── Runtime ───────────────────────────────────────────────────────────────────
FROM base AS runtime
# This COPY makes `runtime` depend on `test`, so the build cannot succeed unless
# the test stage (and therefore the suite) passed. The marker file is inert.
COPY --from=test /app/.tests-passed /app/.tests-passed
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
