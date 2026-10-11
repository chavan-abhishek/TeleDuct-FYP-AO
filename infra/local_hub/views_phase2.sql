-- infra/local_hub/views_phase2.sql
-- Analytical views for Phase 2 real-time logit interception & correlation

-- 1. Correlation View: Links agent reasoning node with token margin metrics
CREATE OR REPLACE VIEW v_step_fragility_correlation AS
SELECT 
    i.session_id,
    i.step_id,
    i.step_index,
    i.node_name,
    i.current_objective,
    i.selected_action,
    l.sampling_margin,
    l.fragility_flag,
    l.confidence_label,
    l.top_k_probs,
    l.timestamp_ns,
    COALESCE(TO_TIMESTAMP(l.timestamp_ns / 1000000000.0), i.created_at) AS event_time
FROM intent_events i
LEFT JOIN logit_telemetry l ON i.step_id = l.step_id
ORDER BY i.created_at DESC;

-- 2. Session Summary View: Aggregates fragility risk per agent run
CREATE OR REPLACE VIEW v_session_fragility_summary AS
SELECT 
    s.session_id,
    s.parent_goal,
    s.agent_type,
    s.final_decision,
    s.total_steps,
    COUNT(l.id) AS total_logit_records,
    COUNT(CASE WHEN l.fragility_flag = true THEN 1 END) AS fragile_steps_count,
    ROUND(AVG(l.sampling_margin)::numeric, 4) AS avg_sampling_margin,
    ROUND(MIN(l.sampling_margin)::numeric, 4) AS min_sampling_margin,
    CASE 
        WHEN COUNT(CASE WHEN l.fragility_flag = true THEN 1 END) > 0 THEN 'AT_RISK'
        ELSE 'ROBUST'
    END AS trajectory_health
FROM sessions s
LEFT JOIN logit_telemetry l ON s.session_id = l.session_id
GROUP BY s.session_id, s.parent_goal, s.agent_type, s.final_decision, s.total_steps
ORDER BY s.started_at DESC;

-- 3. Live Fragility Alert View: Immediate feed of low-margin token decisions
CREATE OR REPLACE VIEW v_fragile_steps_alert AS
SELECT 
    l.id,
    l.session_id,
    l.step_id,
    l.sampling_margin,
    l.confidence_label,
    l.top_k_token_ids,
    l.top_k_probs,
    TO_TIMESTAMP(l.timestamp_ns / 1000000000.0) AS alert_time
FROM logit_telemetry l
WHERE l.fragility_flag = true
ORDER BY l.timestamp_ns DESC;