# Next Session Execution Plan (Member 1 & Member 2)

Because we are abandoning raw TCP and `ngrok`, the workflow between Member 1 (Mac / Coordinator) and Member 2 (GPU) is much simpler and drastically more secure.

**The Magic of SSH Tunnels:**
Member 1 (Mac) doesn't need to send ANY URLs to Member 2 (GPU). 
Instead, Member 1 initiates a single SSH connection to Member 2's RunPod. This connection automatically maps the Mac's database/telemetry ports to the RunPod's `localhost`, AND maps the RunPod's LLM port to the Mac's `localhost`.

Here is exactly what needs to happen in the next session:

---

## 👨‍💻 Member 2 (GPU Person) 

### Step 1: Start the Pod
1. Log into RunPod and spawn a new NVIDIA L4 (or RTX 3090/4090) pod using the `lmsysorg/sglang:latest` image.
2. Go to the "My Pods" page and click the **Connect** button on the new pod.
3. You will see an SSH command that looks something like this:
   `ssh root@213.173.105.13 -p 16264`
4. **SEND THIS ENTIRE COMMAND TO MEMBER 1.** 
   - `213.173.105.13` is the IP Address.
   - `16264` is the SSH Port.

### Step 2: Setup the Server
1. Follow the exact instructions in `docs/plans/p2_member_GPU/RUNPOD_SETUP_GUIDE.md` to clone the repo, install the pip dependencies (without a `venv`), and start the SGLang server.
2. Verify the server is running by seeing `✅ SGLang server is HEALTHY on port 18000!` in the logs.
3. Because Member 1 is handling the SSH tunnel, your environment variables for PostgreSQL and OpenTelemetry are just `localhost`! You don't need any ngrok URLs.

---

## 👨‍💻 Member 1 (Mac / Coordinator)

### Step 1: Start Local Services
1. Ensure your Docker containers for OpenTelemetry Collector (port 4318) and PostgreSQL (port 5432) are running on your Mac.

### Step 2: Establish the SSH Tunnel (The Handshake)
1. Wait for Member 2 to send you the RunPod SSH command (e.g. `ssh root@213.173.105.13 -p 16264`).
2. Open your Mac's terminal and run this command, injecting the Local and Remote port forwarding flags. 

*(Assuming the IP is `213.173.105.13` and the port is `16264`)*:
```bash
ssh root@213.173.105.13 -p 16264 \
    -L 18000:localhost:18000 \
    -R 4318:localhost:4318 \
    -R 5432:localhost:5433 \
    -i ~/.ssh/id_rsa
```

**What this command does:**
- `-L 18000:localhost:18000`: Forwards port 18000 on your Mac to port 18000 on the RunPod. You can now curl `localhost:18000` on your Mac, and it will talk to the GPU's SGLang server.
- `-R 4318:localhost:4318`: Forwards port 4318 on the RunPod to port 4318 on your Mac. When the GPU sends telemetry to `localhost:4318`, it arrives at your Mac's OTel Collector.
- `-R 5432:localhost:5432`: Forwards port 5432 on the RunPod to port 5432 on your Mac. When the GPU writes to `localhost:5432`, it hits your Mac's Postgres DB.

### Step 3: Test the Pipeline
1. Run `python3 Layer_A/scripts/test_remote_to_local_pipeline.py` on the RunPod. It will send data to `localhost`, which will seamlessly appear on your Mac.
2. Run a test LLM generation script from your Mac to `http://localhost:18000/v1/chat/completions`.
