from prometheus_client import Counter, Histogram

REQUEST_COUNTER = Counter("docsum_requests_total", "Total requests to document summarizer", ["endpoint"])
REQUEST_LATENCY = Histogram("docsum_request_latency_seconds", "Request latency seconds", ["endpoint"]) 

def record_request(endpoint: str, elapsed: float):
    REQUEST_COUNTER.labels(endpoint=endpoint).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
