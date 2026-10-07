#!/usr/bin/env python3
"""
Unified runner for Layer A background daemons:
- GPUPoller (NVML GPU hardware metrics)
- SGLangMetricsScraper (Engine throughput, TTFT, queue depth)
"""
import os
import sys
import time
import signal

# Ensure Layer_A root is on sys.path so sibling modules (emitter, gpu_poller, etc.) can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from emitter import DualWriteEmitter
from gpu_poller import GPUPoller
from sglang_scraper import SGLangMetricsScraper

def main():
    otel_endpoint = os.environ.get("OTEL_ENDPOINT", "http://localhost:4318")
    pg_dsn = os.environ.get("PG_DSN", None)
    sglang_url = os.environ.get("SGLANG_URL", "http://localhost:18000")
    poll_interval = float(os.environ.get("POLL_INTERVAL", "2.0"))

    print("==================================================")
    print("🛰️  Starting TeleDuct Layer A Telemetry Daemons")
    print(f"   - OTel Endpoint : {otel_endpoint}")
    print(f"   - PG DSN        : {'Configured' if pg_dsn else 'None (OTel only)'}")
    print(f"   - SGLang URL    : {sglang_url}")
    print(f"   - Poll Interval : {poll_interval}s")
    print("==================================================")

    emitter = DualWriteEmitter(otel_endpoint=otel_endpoint, pg_dsn=pg_dsn)
    
    poller = GPUPoller(emitter, interval=poll_interval)
    poller.start()

    scraper = SGLangMetricsScraper(emitter, sglang_url=sglang_url, interval=poll_interval + 1.0)
    scraper.start()

    print("✅ Daemons active and streaming. Press Ctrl+C to stop.")

    def handle_exit(sig, frame):
        print("\n🛑 Stopping daemons...")
        poller.stop()
        scraper.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()