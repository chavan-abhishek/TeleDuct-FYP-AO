-- infra/local_hub/views_phase3_moe.sql
-- Analytical views for Phase 3 MoE routing telemetry & expert load balance

-- 1. Full Correlation View: Joins Agent Intent, Token Logits, and MoE Routing
CREATE OR REPLACE VIEW v_moe_step_correlation AS
SELECT 
    COALESCE(m.session_id, i.session_id) AS session_id,
    COALESCE(m.step_id, i.step_id) AS step_id,
    COALESCE(i.node_name, 'moe_inference') AS node_name,
    i.selected_action,
    l.sampling_margin AS token_sampling_margin,
    l.fragility_flag AS token_fragile,
    m.layer_idx,
    m.num_experts_total,
    m.experts_per_token,
    m.avg_routing_margin,
    m.routing_fragile,
    m.expert_load,
    COALESCE(TO_TIMESTAMP(m.timestamp_ns / 1000000000.0), i.created_at, NOW()) AS event_time
FROM moe_telemetry m
FULL OUTER JOIN intent_events i ON m.step_id = i.step_id
LEFT JOIN logit_telemetry l ON COALESCE(m.step_id, i.step_id) = l.step_id
ORDER BY event_time DESC;

-- 2. Flattened Expert Load Breakdown View (Unpacks JSONB into tabular expert fractions)
CREATE OR REPLACE VIEW v_moe_expert_load_breakdown AS
SELECT 
    m.id AS moe_record_id,
    m.session_id,
    m.step_id,
    m.layer_idx,
    kv.key::int AS expert_id,
    kv.value::text::double precision AS load_fraction,
    m.avg_routing_margin,
    m.routing_fragile,
    TO_TIMESTAMP(m.timestamp_ns / 1000000000.0) AS event_time
FROM moe_telemetry m,
LATERAL jsonb_each(m.expert_load) AS kv(key, value)
ORDER BY m.timestamp_ns DESC, m.layer_idx ASC, expert_id ASC;

-- 3. Layer-Wise MoE Routing Health Summary
CREATE OR REPLACE VIEW v_moe_layer_health_summary AS
SELECT 
    layer_idx,
    COUNT(*) AS total_evaluations,
    ROUND(AVG(avg_routing_margin)::numeric, 4) AS mean_routing_margin,
    ROUND(MIN(avg_routing_margin)::numeric, 4) AS min_routing_margin,
    COUNT(CASE WHEN routing_fragile = true THEN 1 END) AS fragile_routing_count,
    ROUND((COUNT(CASE WHEN routing_fragile = true THEN 1 END)::numeric / COUNT(*)::numeric) * 100, 2) AS fragile_rate_pct
FROM moe_telemetry
GROUP BY layer_idx
ORDER BY layer_idx ASC;

-- 4. Overloaded Experts Detection View (Identifies experts taking > 30% of total tokens)
CREATE OR REPLACE VIEW v_moe_bottleneck_experts AS
SELECT 
    layer_idx,
    expert_id,
    COUNT(*) AS overload_occurrences,
    ROUND(AVG(load_fraction)::numeric, 4) AS avg_peak_load,
    MAX(load_fraction) AS max_recorded_load
FROM v_moe_expert_load_breakdown
WHERE load_fraction > 0.30
GROUP BY layer_idx, expert_id
ORDER BY overload_occurrences DESC;