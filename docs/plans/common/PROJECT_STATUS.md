# TeleDuct Project Status

## Current Milestone Achieved (Oct 2026)
- **SGLang Server Configuration:** Successfully resolved the dependency hell (CUDA, PyTorch, and SGLang ABI mismatches) by using the system Python on the official SGLang/RunPod container.
- **Model Loading:** The `openai/gpt-oss-20b` model successfully loaded and the server is HEALTHY on port 18000 using the NVIDIA L4 GPU.
- **Next Step Unlocked:** We are now ready to verify the end-to-end telemetry and inference pipeline.

## Architectural Decision Records (ADRs)
- **Dependency Management (Layer A):** We avoid using a virtual environment (`venv`) for PyTorch and SGLang installations on RunPod instances. Mixing torch binaries via pip caused `common_ops` and `undefined symbol` C++ crashes. We now rely on the container's native system Python and `pip install "sglang[all]"`.
- **Layer A <-> Layer B Communication:** We are abandoning raw TCP handshakes. All secure communication, handshakes, and port forwarding (including the OTel pipeline and API requests) will be routed through standard SSH tunnels.

## Next Session Action Items
1. Spin up a new RunPod instance using the guide in `docs/plans/p2_member_GPU/RUNPOD_SETUP_GUIDE.md`.
2. Establish the SSH tunnel from the local Mac to the RunPod instance.
3. Verify the Remote-to-Local pipeline (OTel metrics) via the SSH tunnel.
4. Test actual inference requests from Layer B (Mac) to Layer A (RunPod) over the SSH tunnel.
