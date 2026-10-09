# TeleDuct Project Status

## Current Milestone Achieved (Oct 2026)
- **Phase 1 Complete:** We have successfully established the foundational architecture. 
- **End-to-End Handshake Verified:** Telemetry emitted from the GPU (Layer A) travels securely over an SSH tunnel and correctly inserts into the local PostgreSQL database (Layer B) on the Mac.
- **Model Loading:** The `openai/gpt-oss-20b` model successfully loads from the persistent `/workspace` cache and serves on port 18000.

## Architectural Decision Records (ADRs)
- **Dependency Management (Layer A):** We avoid using a virtual environment (`venv`) for PyTorch and SGLang installations on RunPod instances. Mixing torch binaries via pip caused C++ crashes. We rely on the container's native system Python.
- **Layer A <-> Layer B Communication:** We abandoned raw TCP (ngrok) handshakes. All secure communication (including OTel, Postgres, and SGLang API requests) is routed through standard SSH tunnels.
- **Mac Postgres Port Conflict:** We remapped the local Docker PostgreSQL instance to port **5433** to avoid conflicts with native Mac Postgres installations.

---

## 🚀 Daily Resume Guide (How to get back to this exact state)
If you delete your RunPod to save credits, your `/workspace` network volume is preserved. When you return for the next session, follow these exact steps to instantly recreate the Phase 1 state.

### 1. Start the Mac Infrastructure
Open a terminal on your Mac and ensure the Docker containers are running:
```bash
cd ~/Documents/VAST/TeleDuct/infra/local_hub
docker compose up -d
```

### 2. Spawn the RunPod & Establish the Tunnel
1. On RunPod, deploy a new L4 (or 3090/4090) pod using your existing Network Volume. Use the `lmsysorg/sglang:latest` image.
2. Click **Connect** and get the IP and Port (e.g., `213.173.105.13` and `16264`).
3. **On your Mac**, open a terminal and run this command to SSH in AND establish the tunnels:
```bash
ssh root@213.173.105.13 -p 16264 \
    -L 18000:localhost:18000 \
    -R 4318:localhost:4318 \
    -R 5432:localhost:5433 \
    -i ~/.ssh/id_rsa
```

### 3. Start the GPU Server
Once logged into the RunPod via the SSH command above, execute:
```bash
# 1. Navigate to the preserved workspace
cd /workspace/repo

# 2. Re-install telemetry dependencies into the new pod's system python
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http pynvml psycopg2-binary requests

# 3. Start the SGLang server (reads weights from the preserved /workspace cache)
./Layer_A/scripts/start_layer_a.sh
```

### 4. Verify Phase 1 is Active
Open a second RunPod terminal (or just wait for SGLang to say HEALTHY, then stop it and restart it) and run:
```bash
cd /workspace/repo
export OTEL_ENDPOINT="http://localhost:4318"
export PG_DSN="postgresql://teleduct_admin:teleduct_secure_pass_2026@localhost:5432/teleduct_telemetry"
python3 Layer_A/scripts/test_remote_to_local_pipeline.py
```
If this succeeds, you are perfectly synced and ready to begin Phase 2!
