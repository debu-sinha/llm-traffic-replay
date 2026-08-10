"""llm-traffic-replay: replay a traffic shape against a streamed LLM endpoint.

A self-contained load generator and measurement client for evaluating LLM
serving endpoints that implement the tested streamed Chat Completions subset.
Route, authentication, request controls, SSE framing, response identity, and
usage fields still require endpoint-specific conformance validation. Traffic
shapes may be production-derived or explicitly synthetic and can include
heavy-tailed prompt sizes, cache-eligible prefix reuse, and bursty arrivals.

Design principles:
  1. Reported, not assumed. Intended prefix reuse is separated from
     endpoint-reported cached tokens; achieved arrival rate and token-targeting
     error accompany the latency evidence.
  2. Instrument validated first. The bundled mock server has a known latency
     model; `python -m traffic_replay validate` checks the local
     sampling-to-report path against that oracle before it points at anything
     real. It does not validate a provider dialect or production network.
  3. Small runtime dependency set. Python 3.10+, NumPy, and a standard-library
     HTTP client keep deployment requirements explicit.
"""

from .adapters import (
    DEFAULT_ENDPOINT_ADAPTER,
    EndpointAdapter,
    EventMilestones,
    ResponseOutcome,
    endpoint_adapter_contract,
    get_endpoint_adapter,
    list_endpoint_adapters,
    register_endpoint_adapter,
)
from .capabilities import (
    CAPABILITY_SCHEMA_VERSION,
    CapabilityCatalog,
    CapabilityProfile,
    load_capability_profile,
    seal_capability_profile,
)

__version__ = "0.7.0"

__all__ = [
    "DEFAULT_ENDPOINT_ADAPTER",
    "EndpointAdapter",
    "EventMilestones",
    "ResponseOutcome",
    "CAPABILITY_SCHEMA_VERSION",
    "CapabilityCatalog",
    "CapabilityProfile",
    "endpoint_adapter_contract",
    "get_endpoint_adapter",
    "list_endpoint_adapters",
    "load_capability_profile",
    "register_endpoint_adapter",
    "seal_capability_profile",
]
