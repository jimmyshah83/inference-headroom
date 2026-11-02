# Getting started

How to reach the project's 8×H200 node and serve **DeepSeek-V4.1-Flash-1017**
with vLLM for the coding-agent PoC. The node was requested on 2026-11-02 and is
held until **Fri 2026-11-20**. None of the steps below have been run on it yet.
Every command block has a comment above each command saying what it does.

This guide replaces the 1×B300 / Qwen3.8-Flash-Next guide (still in git
history). Why the plan changed: [ADR 0002](adr/0002-coding-agent-poc-deepseek-v4.1-flash-on-8xh200-with-vllm.md).
What counts as done: [PoC plan](poc.md). What to study alongside it:
[learning plan](learning-plan.md).

| | |
|---|---|
| Node | `<instance-id>` (Nebius, platform `gpu-h200-sxm`, preset `8gpu-128vcpu-1600gb`). Fill in when granted |
| Region | `<region>` (H200 runs in `eu-north1`, `eu-west1`, `us-central1`) |
| SSH user | `<ssh-user>` |
| SSH key (on the Mac) | `~/.ssh/<key>` |
| GPUs | 8× NVIDIA H200 SXM: 141 GB HBM3e and 4.8 TB/s each (1,128 GB per node), NVLink 900 GB/s |
| Host | 128 vCPU (Xeon Platinum 8468), 1,600 GiB RAM |
| Engine | vLLM v0.32.0, image `vllm/vllm-openai@sha256:befccf0e…5c5b34` (CUDA 13.0) |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision `1bae0e1088993b9b82de08f59de5be5a34e42170` (~511 GB) |
| Layout | Two replicas, each TP4 with the Engram tables offloaded to host RAM (the vLLM recipe's H200 default) |

## On the Mac

### 1. Pick the Nebius CLI profile

Which profile works depends on where capacity places the node:

- **Token Factory tenant** (for example, the Architects project): the default
  profile `tokenfactory-solutions-architects` should see it.
- **AI Cloud project**: create an AI Cloud profile, as for the old B300 VM. Run
  this in your own terminal, because it opens a browser to log in.

```bash
# Create a profile that signs in through AI Cloud (auth.nebius.com) and
# defaults to the project that owns the node. It becomes the active profile.
nebius profile create --profile ai-cloud \
  --endpoint api.nebius.cloud \
  --federation-endpoint auth.nebius.com \
  --parent-id <project-id>

# List profiles; the active one is marked [default].
nebius profile list
```

To use a different profile for one command, add `--profile <name>` to it.

### 2. Find the node's public IP

```bash
# Read the node's details as JSON and print the address of its first network
# interface, with the "/32" suffix removed.
nebius compute instance get --id <instance-id> --format json \
  | jq -r '.status.network_interfaces[0].public_ip_address.address | split("/")[0]'
```

If the node is stopped and started again, the IP can change unless a public IP
allocation is attached.

### 3. Add an SSH shortcut

Add this to `~/.ssh/config`, replacing the placeholders:

```
Host h200
  HostName <public-ip>
  User <ssh-user>
  IdentityFile ~/.ssh/<key>
  IdentitiesOnly yes
  LocalForward 8000 localhost:8000
  LocalForward 8001 localhost:8001
```

- `IdentitiesOnly yes` offers only this key, so the server doesn't reject the
  login after trying every key in the agent.
- The two `LocalForward` lines make replica A (port 8000) and replica B
  (port 8001) reachable on the Mac while the SSH session is open. The servers
  are never exposed on the public IP.

### 4. Connect

```bash
# Open a shell on the node. This also starts both tunnels.
ssh h200

# Open only the tunnels, with no shell (-N). Use this in a spare terminal.
ssh -N h200
```

## On the node

### 5. Check the GPUs, their links, and NUMA layout

```bash
# Show each GPU's model, the memory the driver exposes, and the driver version.
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# Show how the GPUs connect to each other (expect NV18 = NVLink between every
# pair) and which CPU cores and NUMA node each GPU is closest to.
nvidia-smi topo -m

# Show the CPU sockets and NUMA nodes.
lscpu | grep -E 'Socket|NUMA'

# Confirm Docker is installed.
docker --version
```

Record the output in `~/runs/env.txt` (step 12). The NUMA layout matters for
this model. Each replica keeps its Engram tables in pinned host memory and
reads rows from them every step. A replica on GPUs 0–3 should use the socket
those GPUs attach to. That is one of the PoC experiments (O3 in the
[PoC plan](poc.md)).

### 6. Let your user run Docker

If `docker run` fails with `permission denied while trying to connect to the
docker API`, add your user to the `docker` group:

```bash
# Add the current user to the docker group, so docker works without sudo.
sudo usermod -aG docker $USER

# Apply the new group in this shell only. Alternatively, log out and SSH back in.
newgrp docker
```

Membership in the `docker` group is effectively root on the node. That is fine
on a single-user node.

### 7. Check that containers can see all 8 GPUs

```bash
# Start a throwaway Ubuntu container with GPU access and list the GPUs it
# sees. --rm deletes the container afterwards.
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L
```

You should see `GPU 0` through `GPU 7`, each `NVIDIA H200`.

If it fails with `could not select device driver "" with capabilities:
[[gpu]]`, the NVIDIA Container Toolkit is missing. Install it:

```bash
# Add NVIDIA's signing key, so apt trusts the toolkit packages.
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add NVIDIA's apt repository, signed with that key.
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit.
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Register the NVIDIA runtime in Docker's config.
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker so it picks up the runtime.
sudo systemctl restart docker
```

Then run the `nvidia-smi -L` test again.

### 8. Check memory, disk, and clocks

```bash
# Host RAM in GiB. Each replica pins its own copy of the Engram tables:
# 94 GiB by the recipe's count, up to 183 GiB if the whole table set is
# pinned. Budget ~370 GiB for two replicas, plus page cache for weight loading.
free -g | head -2

# CPU cores. The engines, the load generator and the SWE-bench containers share them.
nproc

# Free space per filesystem. Plan for ~800 GB: the checkpoint (~511 GB),
# the vLLM image (9.2 GB compressed), and SWE-bench task images (~100 GB for 50 tasks).
df -h

# Disks and where they are mounted, to choose where the checkpoint goes.
lsblk -o NAME,SIZE,MOUNTPOINT

# Power limit and maximum clocks. They set the compute ceiling in the
# analytical model, so record them with every run.
nvidia-smi --query-gpu=index,power.limit,clocks.max.sm,clocks.max.mem --format=csv
```

If the boot disk is too small, attach a network disk of at least 1 TB for the
Hugging Face cache and point `HF_HOME` at it (step 9).

### 9. Pull the pinned vLLM image and download the model

```bash
# The pinned v0.32.0 image (CUDA 13.0). Pinning by digest means a later tag
# can't change the engine in the middle of the PoC.
export IMG=vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34

# The model and its exact revision (full hash; a short hash may not resolve).
export MODEL=deepseek-ai/DeepSeek-V4.1-Flash-1017
export REV=1bae0e1088993b9b82de08f59de5be5a34e42170

# Hugging Face cache on the large disk chosen in step 8.
export HF_HOME=/path/on/big/disk/hf

# vLLM's compile and autotune cache. Keeping it on disk makes later starts faster.
export VLLM_CACHE=/path/on/big/disk/vllm-cache
mkdir -p $HF_HOME $VLLM_CACHE

# Download the image (one time, ~9.2 GB compressed).
docker pull $IMG

# Print the vLLM version inside the image, to confirm it is 0.32.0.
docker run --rm --entrypoint python3 $IMG -c "import vllm; print(vllm.__version__)"

# Download the checkpoint into HF_HOME with the hf CLI that ships in the
# image. ~511 GB, one time.
docker run --rm --entrypoint hf -v $HF_HOME:/root/.cache/huggingface $IMG \
  download $MODEL --revision $REV
```

Put the `export` lines in `~/.bashrc` so later sessions have them. If the image
has no `hf` command, install it on the host with `pip install -U huggingface_hub`
and run `hf download $MODEL --revision $REV` directly.

### 10. Start the two replicas

Each replica is one vLLM server on 4 GPUs. The baseline is the vLLM recipe's
H200 configuration with tool calling and reasoning turned on and the vision
encoder skipped. **Speculative decoding is off in the baseline.** Turning on
DSpark is an experiment (O4 in the [PoC plan](poc.md)).

```bash
# Define a helper that starts one replica as a background container.
#   $1 container name, $2 GPU list, $3 port. Any further arguments are
#   passed through to "vllm serve" as extra flags.
#   --gpus "device=..."   give the container only these GPUs
#   --ipc=host            share host memory for inter-process communication
#   -p 127.0.0.1:...      publish the port on the node's loopback only
#   HF_HUB_OFFLINE=1      use the downloaded files and never re-download
#   VLLM_ENGINE_READY_TIMEOUT_S=3600   the first start can take a long time
serve() {
  local name=$1 gpus=$2 port=$3; shift 3
  docker run -d --name "$name" --gpus "\"device=$gpus\"" --ipc=host \
    -p 127.0.0.1:$port:$port \
    -e HF_HUB_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    -v $HF_HOME:/root/.cache/huggingface -v $VLLM_CACHE:/root/.cache/vllm \
    --entrypoint vllm $IMG serve $MODEL --revision $REV \
      --tensor-parallel-size 4 \
      --tokenizer-mode deepseek_v41 \
      --engram-config '{"cpu_offload":true}' \
      --language-model-only \
      --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
      --reasoning-parser deepseek_v41 \
      --host 0.0.0.0 --port $port "$@"
}

# Start replica A on GPUs 0-3 and replica B on GPUs 4-7.
serve vllm-a 0,1,2,3 8000
serve vllm-b 4,5,6,7 8001

# Follow replica A's startup log. Ctrl-C stops following; the server keeps running.
docker logs -f vllm-a
```

What the flags do:

| Flag | Purpose |
|---|---|
| `--revision` | Load the exact commit pinned above |
| `--tensor-parallel-size 4` | Split each layer across 4 GPUs. Two TP4 replicas fill the node |
| `--tokenizer-mode deepseek_v41` | Encode prompts with DeepSeek's Python encoder. The repo ships no Jinja chat template |
| `--engram-config '{"cpu_offload":true}'` | Keep the Engram n-gram tables (183 GiB) in pinned host RAM instead of HBM. The recipe reports 23.6 GiB moved off each GPU and ~38.5 GiB of KV cache left per GPU. Without the offload, the KV pool collapses. The recipe's figures don't add up to the 183 GiB table size, so measure host and GPU memory yourself |
| `--language-model-only` | Skip the vision encoder. The coding workload is text only, so this frees memory for KV cache |
| `--tool-call-parser deepseek_v41 --enable-auto-tool-choice` | Turn the model's DSML tool-call blocks into OpenAI `tool_calls` |
| `--reasoning-parser deepseek_v41` | Return the thinking in `reasoning_content`, separate from the answer |
| `--host 0.0.0.0 --port …` | Listen on every interface inside the container. Docker exposes the port only on the node's loopback |

vLLM's defaults apply to everything else: prefix caching on, a GPU memory
utilization of 0.9, and the default scheduler token budget. The experiments in
the PoC plan change these one at a time.

Both replicas read the same 511 GB from disk, so the first start is slow. Once
the files are in the page cache, the second replica and later restarts are
faster. A replica is ready when its log shows `Application startup complete`.

Record these from each replica's startup log:

- The KV cache memory available and the number of tokens the KV cache can hold.
- **Which MoE kernel was picked for the MXFP4 experts.** H200 has no FP4 tensor
  cores. vLLM v0.32.0 falls back to W4A16 (4-bit weights, BF16 compute) when no
  W4A4 kernel exists. If so, the compute ceiling for the expert layers is the
  BF16 rate, not FP8. That changes the analytical model.
- The Engram offload message and how much host memory it pinned.
- How long the start took, split into weight loading and CUDA graph capture.

### 11. Test it from the Mac

With `ssh h200` or `ssh -N h200` open:

```bash
# Health check on both replicas. Each returns HTTP 200 once ready.
curl -i localhost:8000/health
curl -i localhost:8001/health

# List the model being served.
curl -s localhost:8000/v1/models

# One request with thinking off. The answer should be 323.
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
  "messages": [{"role": "user", "content": "What is 17*19? Return only the integer."}],
  "chat_template_kwargs": {"thinking": false},
  "max_tokens": 64
}' | jq -r '.choices[0].message.content'

# One tool-call request, the shape a coding agent sends. It should print a
# run_shell call with a command such as "ls".
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
  "messages": [{"role": "user", "content": "List the files in the repository root."}],
  "tools": [{"type": "function", "function": {
    "name": "run_shell",
    "description": "Run a shell command in the repository and return its output",
    "parameters": {"type": "object",
                   "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}],
  "tool_choice": "auto",
  "chat_template_kwargs": {"thinking": false},
  "max_tokens": 1024
}' | jq '.choices[0].message.tool_calls'
```

If you leave thinking unset, it is **on at effort 50**. A small `max_tokens` is
then spent on reasoning and the answer comes back empty with
`finish_reason=length`. That is expected behaviour, not a broken model.

### 12. Record the environment, then stop

```bash
# Append the run environment to a log: UTC time, image digest, model revision,
# vLLM/torch/CUDA versions, GPU settings, and the GPU topology.
mkdir -p ~/runs && {
  date -u +%FT%TZ
  echo "$IMG  model $MODEL rev $REV"
  docker exec vllm-a python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit,clocks.max.sm,clocks.max.mem --format=csv,noheader
  nvidia-smi topo -m
} | tee -a ~/runs/env.txt

# Stop and remove both replicas. The image, checkpoint and vLLM cache stay on disk.
docker stop vllm-a vllm-b && docker rm vllm-a vllm-b
```

When a session ends, stop the node from the Mac so it isn't billed while idle:

```bash
# Stop the node. Bring it back with "nebius compute instance start --id …".
nebius compute instance stop --id <instance-id>
```

Release the node on or before **Fri 2026-11-20**. If the PoC finishes early,
tell capacity, so the node goes back to the pool.

## Next

- Fill in the node table once the node is granted.
- Run the baseline and the quality smoke tests: Week 1 of the [PoC plan](poc.md).
- Start the analytical model with the memory and roofline worksheet in the
  [learning plan](learning-plan.md).
