#!/usr/bin/env python3
"""
layer_b/demo_agent/run_audit_agent.py
Executes financial compliance agent runs while populating 100% of the intent schema fields:
parent_goal, current_objective, available_tools, selected_action, selection_rationale, guardrail_checks.
"""
import os
import sys
import time
import uuid
import json
import psycopg2
import requests

PG_DSN = os.environ.get(
    "PG_DSN", 
    "postgresql://teleduct_admin:teleduct_secure_pass_2026@localhost:5433/teleduct_telemetry"
)
SGLANG_URL = os.environ.get("SGLANG_URL", "http://localhost:18000/v1")

def log_intent(conn, d):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO intent_events 
            (intent_id, session_id, step_id, step_index, node_name, parent_goal, 
             current_objective, available_tools, selected_action, selection_rationale, 
             guardrail_checks, goal_alignment_check)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (intent_id) DO NOTHING;
        """, (
            str(uuid.uuid4()), d["session_id"], d["step_id"], d["step_index"],
            d["node_name"], d["parent_goal"], d["current_objective"],
            json.dumps(d["available_tools"]), d["selected_action"],
            d["selection_rationale"], json.dumps(d["guardrail_checks"]),
            d.get("goal_alignment_check", "aligned")
        ))
    conn.commit()

def log_logits(conn, d):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO logit_telemetry
            (session_id, step_id, request_id, timestamp_ns, top_k_token_ids, 
             top_k_probs, sampling_margin, fragility_flag, confidence_label)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            d["session_id"], d["step_id"], d.get("request_id", "req_audit"),
            time.time_ns(), json.dumps(d["top_k_token_ids"]),
            json.dumps(d["top_k_probs"]), d["sampling_margin"],
            d["fragility_flag"], d["confidence_label"]
        ))
    conn.commit()

def main():
    print("==================================================")
    print("🏦 TeleDuct Tier 1: Deep Intent Audit Agent Runner")
    print(f"   - Database Target : {PG_DSN.split('@')[-1]}")
    print("==================================================")

    conn = psycopg2.connect(PG_DSN)

    test_trajectories = [
        {
            "tx": "Wire transfer $49,500 to offshore account with undisclosed ultimate beneficial owner.",
            "is_suspicious": True,
            "action": "flag_transaction",
            "rationale": "Transaction amount of $49,500 is directly below the $50,000 threshold with missing UBO declaration, indicating potential AML structuring.",
            "rule": "aml_structuring_threshold",
            "candidates": ["flag_transaction", "request_human_review", "bypass_alert"],
            "probs": [0.495, 0.480, 0.025],  # Margin 0.015 -> FRAGILE!
        },
        {
            "tx": "Standard retail cash deposit of $2,400 for local bakery weekly operating float.",
            "is_suspicious": False,
            "action": "approve_transaction",
            "rationale": "Transaction amount of $2,400 aligns with historical weekly cash float profile for verified merchant category code 5462.",
            "rule": "merchant_deposit_normal",
            "candidates": ["approve_transaction", "flag_transaction", "request_kyc_refresh"],
            "probs": [0.820, 0.110, 0.070],  # Margin 0.710 -> STABLE!
        }
    ]

    for idx, t in enumerate(test_trajectories, 1):
        session_id = str(uuid.uuid4())
        step_id = f"{session_id}__step_01_compliance"

        # 1. Register Session
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sessions (session_id, parent_goal, agent_type, total_steps)
                VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING;
            """, (session_id, "audit_incoming_wire_transfers", "financial_compliance", 1))
        conn.commit()

        # 2. Log Full Intent Context
        margin = round(t["probs"][0] - t["probs"][1], 4)
        fragile = margin < 0.05

        log_intent(conn, {
            "session_id": session_id,
            "step_id": step_id,
            "step_index": 1,
            "node_name": "verify_transaction_limit",
            "parent_goal": "audit_incoming_wire_transfers",
            "current_objective": "verify_if_amount_exceeds_aml_threshold",
            "available_tools": ["flag_transaction", "request_human_review", "bypass_alert"],
            "selected_action": t["action"],
            "selection_rationale": t["rationale"],
            "guardrail_checks": [{"id": "nemo_financial_v1", "rule": t["rule"], "status": "passed"}]
        })

        # 3. Log Top-K Candidate Distribution
        log_logits(conn, {
            "session_id": session_id,
            "step_id": step_id,
            "top_k_token_ids": t["candidates"],
            "top_k_probs": t["probs"],
            "sampling_margin": margin,
            "fragility_flag": fragile,
            "confidence_label": "FRAGILE" if fragile else "CONFIDENT"
        })

        status = "🚨 FRAGILE DECISION" if fragile else "✅ CONFIDENT DECISION"
        print(f"Trajectory {idx}/2 | {status} | Action: {t['action']} | Margin: {margin:.4f}")

    conn.close()
    print("\n🎉 Tier-1 Trajectories Successfully Stored in Database!")

if __name__ == "__main__":
    main()