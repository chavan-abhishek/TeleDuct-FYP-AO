-- infra/local_hub/views_tier1_audit.sql
-- Tier-1 Analytical Views: Full Intent Context, Candidate Logits, and Thermal Telemetry

-- 1. Full Chain of Intent Audit View (Satisfies EU AI Act Articles 12 & 13)
CREATE OR REPLACE VIEW v_chain_of_intent_audit AS
SELECT 
    i.session_id,
    i.step_id,
    i.step_index,
    i.node_name,
    i.parent_goal,
    i.current_objective,
    i.available_tools,
    i.selected_action,
    i.selection_rationale,
    i.guardrail_checks,
    l.sampling_margin,
    l.fragility_flag,
    h.gpu_vram_used_gb,
    h.gpu_sm_utilization,
    h.gpu_power_watts,
    h.gpu_temperature_c,
    i.created_at AS step_timestamp
FROM intent_events i
LEFT JOIN logit_telemetry l ON i.step_id = l.step_id
LEFT JOIN LATERAL (
    SELECT gpu_vram_used_gb, gpu_sm_utilization, gpu_power_watts, gpu_temperature_c
    FROM hardware_telemetry
    ORDER BY timestamp_ns DESC
    LIMIT 1
) h ON true
ORDER BY i.created_at DESC;

-- 2. Top-K Candidate Token Logit Distribution View (Unpacks JSONB candidate probabilities)
CREATE OR REPLACE VIEW v_candidate_token_distribution AS
SELECT 
    l.session_id,
    l.step_id,
    l.sampling_margin,
    l.fragility_flag,
    tok.ordinality AS candidate_rank,
    tok.value::text AS candidate_token,
    prob.value::text::double precision AS candidate_prob,
    TO_TIMESTAMP(l.timestamp_ns / 1000000000.0) AS event_time
FROM logit_telemetry l,
LATERAL jsonb_array_elements_text(l.top_k_token_ids) WITH ORDINALITY AS tok(value, ordinality)
JOIN LATERAL jsonb_array_elements_text(l.top_k_probs) WITH ORDINALITY AS prob(value, ordinality) 
    ON tok.ordinality = prob.ordinality
ORDER BY l.timestamp_ns DESC, tok.ordinality ASC;

-- 3. Hardware Thermal & Power Health Summary
CREATE OR REPLACE VIEW v_hardware_thermal_summary AS
SELECT 
    ROUND(AVG(gpu_temperature_c)::numeric, 1) AS mean_temp_c,
    MAX(gpu_temperature_c) AS peak_temp_c,
    ROUND(AVG(gpu_power_watts)::numeric, 1) AS mean_power_w,
    MAX(gpu_power_watts) AS peak_power_w,
    ROUND(AVG(gpu_vram_used_gb)::numeric, 2) AS mean_vram_gb,
    COUNT(CASE WHEN gpu_temperature_c >= 82 THEN 1 END) AS thermal_throttle_warnings
FROM hardware_telemetry
WHERE timestamp_ns >= (EXTRACT(EPOCH FROM (NOW() - INTERVAL '15 minutes')) * 1000000000);