"""llm-traffic-replay: replay YOUR production traffic shape against an LLM endpoint.

A self-contained load generator and measurement client for evaluating LLM
serving endpoints (provisioned throughput or any OpenAI-compatible API)
under realistic traffic: heavy-tailed prompt sizes, constructed prompt-cache
hit ratios, and bursty arrivals.

Design principles:
  1. Reported, not assumed. Achieved cache rate, achieved arrival rate, and
     token-targeting error are printed next to every latency table.
  2. Instrument validated first. The bundled mock server has a known latency
     model; `python -m traffic_replay validate` proves the measurement path
     before it points at anything real.
  3. Zero exotic dependencies. Python 3.10+, numpy. The HTTP client is
     standard library, so it runs anywhere.
"""

__version__ = "0.5.1"
