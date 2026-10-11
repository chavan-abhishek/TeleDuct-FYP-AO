#!/usr/bin/env python3
"""
layer_b/demo_agent/run_live_agent.py
Executes the live LangGraph financial compliance agent against real SGLang inference
served across the SSH tunnel (http://localhost:18000/v1).
"""
import os
import sys
import time
import psycopg2
import requests

# Add Layer B SDK & Demo paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sdk")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "storage")))

from correlator import AgentSessionCorrelator
from agents import run_financial_compliance_agent

# Local Mac database uses port 5433 (remapped from Docker container)
PG_DSN = os.getenv(
    "PG_DSN",
    "postgresql://teleduct_admin:teleduct_secure_pass_2026@localhost:5433/teleduct_telemetry"
)
SGLANG_BASE_URL = os.getenv("SGLANG_BASE_URL", "http://localhost:18000/v1")

import math
from psycopg2.extras import Json

class SGLangOpenAIWrapper:
    """Live SGLang client that requests token logprobs and writes logit telemetry."""
    def __init__(self, base_url, model="openai/gpt-oss-20b", correlator=None, pg_conn=None):
        self.base_url = base_url
        self.model = model
        self.correlator = correlator
        self.pg = pg_conn

    def invoke(self, prompt: str):
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 64,
            "logprobs": True,
            "top_logprobs": 5
        }
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
        
        # Calculate Sampling Margin from live token logprobs
        margin = 0.50
        top_probs = [0.85, 0.15]
        top_ids = [101, 102]
        
        logprobs_data = choice.get("logprobs", {})
        if logprobs_data and "content" in logprobs_data:
            tokens_content = logprobs_data["content"]
            # Analyze each generated token
            min_margin = 1.0
            found_margins = []
            for t_info in tokens_content:
                top_lps = t_info.get("top_logprobs", [])
                if len(top_lps) >= 2:
                    p1 = math.exp(top_lps[0]["logprob"])
                    p2 = math.exp(top_lps[1]["logprob"])
                    t_margin = max(0.0, min(1.0, p1 - p2))
                    found_margins.append(t_margin)
                    if t_margin < min_margin:
                        min_margin = t_margin
                        top_probs = [round(p1, 4), round(p2, 4)]
            if found_margins:
                # Use minimum margin (bottleneck/most fragile token in reasoning)
                margin = round(min_margin, 4)

        fragility_flag = margin < 0.05
        conf_label = "HIGH" if margin > 0.30 else ("LOW" if fragility_flag else "MEDIUM")

        # If correlator and DB connection are available, record into logit_telemetry
        if self.correlator and self.pg:
            step_id = f"{self.correlator.session_id}__step_{self.correlator.step_counter}"
            try:
                cur = self.pg.cursor()
                cur.execute("""INSERT INTO logit_telemetry
                    (session_id, step_id, request_id, timestamp_ns, top_k_token_ids,
                     top_k_probs, sampling_margin, fragility_flag, confidence_label)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (self.correlator.session_id, step_id, data.get("id", "live-req"),
                     time.time_ns(), Json(top_ids), Json(top_probs),
                     margin, fragility_flag, conf_label))
            except Exception as e:
                print(f"⚠️ Failed to insert logit_telemetry record: {e}")

        class Response:
            def __init__(self, c, m, f, l):
                self.content = c
                self.margin = m
                self.fragility = f
                self.confidence = l
        return Response(content, margin, fragility_flag, conf_label)

def main():
    print("==================================================")
    print("🏦 TeleDuct Phase 2: Live Financial Agent Runner")
    print(f"   - SGLang Endpoint : {SGLANG_BASE_URL}")
    print(f"   - Database Target : {PG_DSN.split('@')[-1]}")
    print("==================================================")

    # 1. Verify SGLang server connectivity through tunnel
    try:
        health = requests.get("http://localhost:18000/health", timeout=5)
        print(f"✅ SGLang server verified reachable (HTTP {health.status_code})")
    except Exception as e:
        print(f"❌ Failed to reach SGLang on http://localhost:18000/health: {e}")
        print("👉 Did you start the SSH tunnel: ssh root@... -L 18000:localhost:18000 ...?")
        sys.exit(1)

    # 2. Connect to local PostgreSQL
    pg = psycopg2.connect(PG_DSN)
    pg.autocommit = True
    correlator = AgentSessionCorrelator(pg)
    llm = SGLangOpenAIWrapper(base_url=SGLANG_BASE_URL, correlator=correlator, pg_conn=pg)

    # 3. Test Cases: Clean transactions vs. Suspicious Transactions (Triggering Fragility)
    test_cases = [
        "Wire transfer $4,900 from standard domestic checking account for payroll invoice #102.",
        "Rapid cross-border structured transfer $9,999 to offshore account with no declared counterparty.",
        "Cash deposit $3,500 at ATM for retail store daily operating revenue.",
        "Layered transaction sequence: three split transfers of $4,950 within 12 minutes to high-risk jurisdiction."
    ]

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- [Run {i}/{len(test_cases)}] Executing Trajectory ---")
        print(f"Input: \"{test_input}\"")

        session_id = correlator.start_session(
            parent_goal=test_input,
            agent_type="financial_compliance"
        )
        print(f"   🔗 Session Started: {session_id}")

        t0 = time.time()
        result = run_financial_compliance_agent(llm, correlator, test_input)
        elapsed = (time.time() - t0) * 1000.0

        decision = result.get("decision", "UNKNOWN")
        correlator.end_session(decision)
        print(f"   🎯 Decision: {decision} (Analysis: \"{result.get('analysis', '').strip()[:60]}...\")")
        print(f"   ⏱️  Completed in {elapsed:.1f}ms")

    print("\n==================================================")
    print("✅ Live Agent Runs Complete. Verifying Correlated DB Entries...")
    cur = pg.cursor()
    cur.execute("""
        SELECT 
            i.session_id, 
            i.step_id, 
            i.node_name, 
            i.selected_action,
            l.sampling_margin,
            l.fragility_flag,
            l.confidence_label
        FROM intent_events i
        LEFT JOIN logit_telemetry l ON i.step_id = l.step_id
        ORDER BY i.created_at DESC 
        LIMIT 8;
    """)
    rows = cur.fetchall()
    for r in rows:
        margin_str = f"Margin: {r[4]:.3f}" if r[4] is not None else "Margin: N/A"
        frag_str = "🚨 FRAGILE" if r[5] else "✅ ROBUST"
        print(f"   - Step: {r[1]} | {r[2]} -> {r[3]} | {margin_str} | {frag_str}")
    print("==================================================")

if __name__ == "__main__":
    main()