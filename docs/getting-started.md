# Getting started

How to reach the B300 node and serve **DeepSeek-V4-Flash-0731** with vLLM for
the coding-agent PoC. Every command block has a comment above each command
saying what it does.

Why this setup: [ADR 0003](adr/0003-deepseek-v4-flash-0731-on-1xb300-with-vllm.md).
What counts as done: [PoC plan](poc.md). The 8×B200 guide for
DeepSeek-V4.1-Flash is in git history.

| | |
|---|---|
| Node | One B300 VM. SSH alias `b300-usn1-1` on the Mac |
| SSH user | `jimmyshah-sshkey` |
| SSH key (on the Mac) | `~/.ssh/id_ed25519_nebius` |
| GPU | 1× NVIDIA B300: 288 GB HBM3e (measure the usable amount in step 3), up to 8 TB/s |
| Engine | vLLM v0.33.0, image `vllm/vllm-openai@sha256:d825baf9…a84c6e` (CUDA 13.0) |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731`, revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` (166.9 GB) |
| Layout | One server on the GPU, with the vLLM recipe's single-GPU settings: TP1 with expert parallel, the DeepGEMM MegaMoE kernel and an fp8 KV cache. Speculative decoding is off |

## On the Mac

### 1. Add an SSH shortcut

`~/.ssh/config` should have this entry. Replace `<public-ip>` with the
node's address. `IdentityFile` is the private key, the file without `.pub`:

```
Host b300-usn1-1
  HostName <public-ip>
  User jimmyshah-sshkey
  IdentityFile ~/.ssh/id_ed25519_nebius
  IdentitiesOnly yes
  LocalForward 8000 localhost:8000
```

- `IdentitiesOnly yes` offers only this key, so the server doesn't reject the
  login after trying every key in the agent.
- `LocalForward` makes the server (port 8000) reachable on the Mac while the
  SSH session is open. The server is never exposed on the public IP.
- If the node is stopped and started again, its public IP can change. Update
  `HostName` if the connection times out.

### 2. Connect

```bash
# Open a shell on the node. This also starts the tunnel.
ssh b300-usn1-1

# Open only the tunnel, with no shell (-N). Use this in a spare terminal.
ssh -N b300-usn1-1
```

## On the node

### 3. Check the GPU

```bash
# Show the GPU's model, the memory the driver exposes, and the driver version.
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# Show the CPU model, sockets and NUMA nodes.
lscpu | grep -E 'Model name|Socket|NUMA'

# Confirm Docker is installed.
docker --version
```

You should see one `NVIDIA B300`. Note `memory.total`: it's the usable HBM,
which the KV cache size in step 8 depends on. Step 10 writes all of this to
`~/runs/env.txt`.

### 4. Let your user run Docker

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

### 5. Check that containers can see the GPU

```bash
# Start a throwaway Ubuntu container with GPU access and list the GPUs it
# sees. --rm deletes the container afterwards.
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L
```

You should see `GPU 0: NVIDIA B300`.

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

### 6. Check memory, disk, and clocks

```bash
# Host RAM in GiB. Nothing is offloaded to host memory, but the page cache
# speeds up loading the 167 GB checkpoint on later starts.
free -g | head -2

# CPU cores. The engine, the load generator and the SWE-bench containers share them.
nproc

# Free space per filesystem. Plan for ~300 GB: the checkpoint (167 GB), the
# vLLM image (9.7 GB compressed), and SWE-bench task images (~100 GB for 50 tasks).
df -h

# Disks and where they are mounted, to choose where the checkpoint goes.
lsblk -o NAME,SIZE,MOUNTPOINT

# Power limit and maximum clocks. They set the compute ceiling in the
# analytical model, so record them with every run.
nvidia-smi --query-gpu=index,power.limit,clocks.max.sm,clocks.max.mem --format=csv
```

Put the checkpoint on a real disk with enough space. Step 7 uses your home
directory; point `HF_HOME` elsewhere if home is too small. Don't use:

- `tmpfs` mounts such as `/dev/shm` and `/run`. They live in RAM and
  disappear on reboot.
- `/mnt/cloud-metadata`, which holds the VM's cloud metadata.

### 7. Pull the pinned vLLM image and download the model

Save the PoC's settings in `~/.bashrc` once, so every new SSH session and tmux
window has them. Steps 8–10 depend on them.

```bash
# Append the settings to ~/.bashrc. The quoted 'EOF' keeps $HOME as written, so
# it expands each time a shell starts.
cat >> ~/.bashrc <<'EOF'

# DeepSeek-V4-Flash PoC on 1×B300 (getting-started step 7)
# The pinned v0.33.0 image (CUDA 13.0). vLLM's recipe needs >= 0.25.0 for the
# 0731 checkpoint. Pinning by digest means a later tag can't change the engine
# in the middle of the PoC.
export IMG=vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e
# The model and its exact revision (full hash; a short hash may not resolve).
export MODEL=deepseek-ai/DeepSeek-V4-Flash-0731
export REV=7872f01b1d1fe23eabc4c98b48bffcef5a386062
# The Hugging Face cache and vLLM's compile and autotune cache. HF_HOME needs
# ~200 GB free; if your home disk doesn't have that, point it at one that does.
export HF_HOME=$HOME/hf
export VLLM_CACHE=$HOME/vllm-cache
EOF

# Load the settings into this shell and create the cache folders.
source ~/.bashrc
mkdir -p $HF_HOME $VLLM_CACHE

# Download the image (one time, ~9.7 GB compressed).
docker pull $IMG

# Print the vLLM version inside the image, to confirm it is 0.33.0.
docker run --rm --entrypoint python3 $IMG -c "import vllm; print(vllm.__version__)"
```

If `~/.bashrc` already has an older block like this, that's fine: the new
lines come later, so they win.

Then download the checkpoint, 167 GB in 74 files, with the `hf` tool that
ships in the image. Use a Hugging Face read token: on the B200 node the
download ran at about 490 MB/s without one and about 1.6 GB/s with one.

```bash
# Start a tmux session, so the download survives a dropped SSH connection.
# Ctrl-b d detaches; "tmux attach -t dl" comes back.
tmux new -s dl

# Paste a read token from huggingface.co/settings/tokens. read -s keeps it off
# the screen and out of your shell history. Don't put it in ~/.bashrc: vLLM runs
# offline in step 8 and doesn't need it.
read -rsp 'HF token: ' HF_TOKEN && export HF_TOKEN; echo

# Check that the token works and reaches the container. It prints your Hugging
# Face username; "Not logged in" or an error means a wrong or missing token.
docker run --rm -e HF_TOKEN --entrypoint hf $IMG auth whoami

# Download the checkpoint into HF_HOME. -e HF_TOKEN passes the token in, and the
# -v mount puts the files on the host, where step 8's container finds them.
docker run --rm -e HF_TOKEN --entrypoint hf -v $HF_HOME:/root/.cache/huggingface $IMG \
  download $MODEL --revision $REV
```

The progress bar counts finished files. The largest file is only 3.7 GB, so it
shouldn't sit still for long. Don't press Ctrl-C to check on it; detach
instead.

If the download stops before it finishes, delete the partial files before you
run it again. The downloader reserves each file's full size up front, so a
partial file can look complete.

```bash
# Delete partial downloads. The files are owned by root, because the container
# runs as root.
sudo rm -f $HF_HOME/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/blobs/*.incomplete
```

Then run the download command again. It skips the files that already finished.

Check the download before step 8:

```bash
# Count the weight shards. It should print 48.
ls $HF_HOME/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/$REV/ \
  | grep -c '^model-.*-of-00048.safetensors$'

# Check every large file against its hash. Each one is named after its own
# SHA-256, so the check needs no other input. It runs 8 files at a time. It
# prints "all files OK", or the name of each file that fails.
cd $HF_HOME/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/blobs
ls | grep -E '^[0-9a-f]{64}$' \
  | xargs -P 8 -I{} sh -c 'echo "{}  {}" | sha256sum -c --quiet' && echo "all files OK"
cd -
```

### 8. Start the server

One vLLM server on the GPU, with the settings from the vLLM recipe's
single-GPU GB300 entry for this checkpoint. The GB300 is the same chip family
as the B300. Every flag was checked against the v0.33.0 source
([ADR 0003](adr/0003-deepseek-v4-flash-0731-on-1xb300-with-vllm.md)).
**Speculative decoding is off in the baseline.** Turning it on is an
experiment (D1 in the [PoC plan](poc.md#decode)).

Save the start command as a script on the node, `~/serve.sh`. The experiments
restart the server many times, and a script is still there after you
reconnect. It reads the settings from step 7, and it stops with an error if
one of them is missing.

```bash
# Write the script. The quoted 'EOF' keeps $IMG, $MODEL and the others as
# written, so the script reads them from ~/.bashrc's exported settings each
# time it runs.
cat > ~/serve.sh <<'EOF'
#!/usr/bin/env bash
# Start the vLLM server as a background container (getting-started step 8).
# Usage: ~/serve.sh NAME GPUS PORT [extra "vllm serve" flags...]
#   --gpus "device=..."   give the container only these GPUs
#   --ipc=host            share host memory for inter-process communication
#   -p 127.0.0.1:...      publish the port on the node's loopback only
#   HF_HUB_OFFLINE=1      use the downloaded files and never re-download
#   VLLM_ENGINE_READY_TIMEOUT_S=7200   the first start can take a long time
#   VLLM_USE_RUST_FRONTEND=1   from the recipe's DeepSeek-V4-Flash configuration
# set -u stops the script if a setting or argument is missing.
set -euo pipefail
name=$1 gpus=$2 port=$3; shift 3
docker run -d --name "$name" --gpus "\"device=$gpus\"" --ipc=host \
  -p 127.0.0.1:$port:$port \
  -e HF_HUB_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  -e VLLM_USE_RUST_FRONTEND=1 \
  -v $HF_HOME:/root/.cache/huggingface -v $VLLM_CACHE:/root/.cache/vllm \
  --entrypoint vllm $IMG serve $MODEL --revision $REV \
    --trust-remote-code \
    --tensor-parallel-size 1 --enable-expert-parallel \
    --moe-backend deep_gemm_mega_moe \
    --tokenizer-mode deepseek_v4 \
    --attention-config '{"indexer_kv_dtype":"mxfp4"}' \
    --kv-cache-dtype fp8 --block-size 256 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
    --host 0.0.0.0 --port $port "$@"
EOF

# Make the script executable.
chmod +x ~/serve.sh

# Start the server on GPU 0, port 8000: the port the sweep script and step 9 use.
# The container runs in the background (docker run -d), so it keeps running
# after you disconnect.
~/serve.sh vllm-a 0 8000

# Follow the startup log. Ctrl-C stops following; the server keeps running.
docker logs -f vllm-a
```

The server is ready when the log shows `Application startup complete`.

What the flags do:

| Flag | Purpose |
|---|---|
| `--revision` | Load the exact commit pinned in step 7 |
| `--trust-remote-code` | The recipe sets it. The model's `config.json` has no `auto_map`, so no code from the repo runs |
| `--tensor-parallel-size 1 --enable-expert-parallel` | The whole model on one GPU. Expert parallel splits nothing across one GPU, but the MoE kernel below won't start without it |
| `--moe-backend deep_gemm_mega_moe` | DeepGEMM's fused MoE kernel. v0.33.0 requires an SM100-family GPU, FP4 experts, `sqrtsoftplus` routing and normalized top-k, and this checkpoint and GPU meet all four |
| `--tokenizer-mode deepseek_v4` | vLLM's built-in DeepSeek-V4 prompt encoder. The repo has no Jinja chat template |
| `--attention-config '{"indexer_kv_dtype":"mxfp4"}'` | Store the sparse-attention indexer's cache in MXFP4 |
| `--kv-cache-dtype fp8 --block-size 256` | An FP8 KV cache in 256-token blocks, as in the recipe. Switching to auto is experiment D4 |
| `--tool-call-parser deepseek_v4 --enable-auto-tool-choice` | Turn the model's tool-call blocks into OpenAI `tool_calls` |
| `--reasoning-parser deepseek_v4` | Return the thinking in `reasoning_content`, separate from the answer |
| `--reasoning-config` | Turns on vLLM's thinking-budget control. The empty strings mean it uses the parser's own start and end markers |
| `--host 0.0.0.0 --port …` | Listen on every interface inside the container. Docker exposes the port only on the node's loopback |

vLLM's defaults apply to everything else: prefix caching on, a GPU memory
utilization of 0.9, the default scheduler token budget and CUDA graph sizes.
The experiments in the PoC plan change these one at a time.

To change a flag, remove the server and start it again with the flag added.
Docker won't reuse a container name while a container with that name exists,
even a stopped one.

```bash
# Remove the server, then start it again with a shorter maximum context.
# Any arguments after the port go to "vllm serve".
docker rm -f vllm-a
~/serve.sh vllm-a 0 8000 --max-model-len 393216
```

**If it doesn't start:**

1. **The log says the KV cache can't hold a full-length request.** The
   default context is 1M tokens. Add `--max-model-len 393216`, as above. That
   covers the 256K traces and is the recipe's minimum for `max` reasoning
   effort.
2. **The MegaMoE kernel errors out.** Remove `--moe-backend
   deep_gemm_mega_moe` and `--enable-expert-parallel` from `~/serve.sh`, so
   vLLM picks a MoE kernel itself.
3. **v0.33.0 can't load the model at all.** Change `IMG` in `~/.bashrc` to
   `vllm/vllm-openai:v0.28.0`, the image the recipe uses on B300, and pin its
   digest once it works.

#### CUDA graphs

In decode, each step runs one token per request, so each kernel does little
work. Launching hundreds of kernels from the CPU can then take longer than
running them. A CUDA graph records one step's kernel launches at startup and
replays them all with a single launch.

- **A graph has a fixed shape.** vLLM captures one graph for each size in
  `cudagraph_capture_sizes` (counted in tokens) and pads each step up to the
  next captured size. A 9-request decode step runs in the 16-token graph, for
  example. A step larger than the biggest size runs without a graph.
- **Full graphs for decode, piecewise graphs for prefill.** In v0.33.0's
  default mode, `FULL_AND_PIECEWISE`, a step where every request is decoding
  replays one graph of the whole forward pass. A step with prefill in it runs
  attention outside the graph.
- **Breakable graphs, not torch.compile.** v0.33.0 turns on breakable CUDA
  graphs for this model (`VLLM_USE_BREAKABLE_CUDAGRAPH=1`, listed for
  `DeepseekV4ForCausalLM` in `vllm/config/vllm.py`). On V4.1 that also set the
  compile mode to `NONE`, so torch.compile was off. Check the startup log for
  the same here.
- **Graphs cost memory and startup time.** That memory would otherwise go to
  the KV cache. The log reports both.

Experiment D2 changes `--max-num-seqs` and the capture sizes. D1 turns on
speculative decoding, which makes each decode step 1 + drafts tokens per
request, so it needs larger capture sizes.

**Record these from the startup log:**

- The KV cache memory, and how many tokens the KV cache can hold.
- **Which MoE kernel was picked.** The baseline asks for DeepGEMM MegaMoE. The
  kernel's math rate (FP4 or FP8) sets the compute ceiling for the experts in
  the analytical model.
- **Which attention backend was picked.**
- How long the start took, split into weight loading and CUDA graph capture,
  and how much GPU memory the CUDA graphs took.
- **The scheduler and CUDA graph settings the defaults resolved to.** The
  baseline leaves them unset, and vLLM picks some defaults by GPU, so these
  are what experiments P2 and D2 change from: `max_num_batched_tokens`,
  `max_num_seqs`, `cudagraph_capture_sizes` and the compile mode.

```bash
# Print the log lines that hold the scheduler and CUDA graph settings. The
# engine config line is long; the values are inside it, and so is the compile
# mode ('mode': <CompilationMode...>).
docker logs vllm-a 2>&1 \
  | grep -E 'max_num_seqs|max_num_batched_tokens|cudagraph_capture_sizes|max_cudagraph_capture_size'

# Print warnings and the compile and CUDA graph lines, leaving out the long
# config lines. "cuda ?graph" matches both "cudagraph" and "CUDA graph".
docker logs vllm-a 2>&1 \
  | grep -iE 'warn|compil|cuda ?graph|capturing|piecewise|eager' \
  | grep -vE 'non-default args|Launching Rust frontend|Initializing a V1'

# Print the kernel choices and the KV cache size.
docker logs vllm-a 2>&1 \
  | grep -iE 'moe backend|megamoe|mega_moe|attention backend|using .*backend|kv cache'
```

Run the searches again after `Application startup complete`, and record which
CUDA graphs were captured, how long capture took, and how much memory the
graphs use.

### 9. Test it from the Mac

With `ssh b300-usn1-1` or `ssh -N b300-usn1-1` open:

```bash
# Health check. It returns HTTP 200 once the server is ready.
curl -i localhost:8000/health

# List the model being served.
curl -s localhost:8000/v1/models

# One request with thinking off. The answer should be 323.
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
  "messages": [{"role": "user", "content": "What is 17*19? Return only the integer."}],
  "chat_template_kwargs": {"thinking": false},
  "max_tokens": 64
}' | jq -r '.choices[0].message.content'

# One tool-call request, the shape a coding agent sends. It should print a
# run_shell call with a command such as "ls".
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
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

If you leave thinking unset, v0.33.0 turns it **on at effort `high`**. A small
`max_tokens` is then spent on reasoning, and the answer comes back empty with
`finish_reason=length`. That's expected, not a broken model. To choose the
effort, pass `"chat_template_kwargs": {"thinking": true, "reasoning_effort":
"low"}` (or `"high"`, or `"max"`).

The Rust frontend may log that it ignores `--enable-auto-tool-choice`. The
tool-call request above checks that tool calls still come back as
`tool_calls`.

### 10. Record the environment, then stop

```bash
# Append the run environment to a log: UTC time, image digest, model revision,
# vLLM/torch/CUDA versions, GPU settings, and the CPU.
mkdir -p ~/runs && {
  date -u +%FT%TZ
  echo "$IMG  model $MODEL rev $REV"
  docker exec vllm-a python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit,clocks.max.sm,clocks.max.mem --format=csv,noheader
  lscpu | grep -E 'Model name|Socket|NUMA'
} | tee -a ~/runs/env.txt

# Stop and remove the server. The image, checkpoint and vLLM cache stay on disk.
docker stop vllm-a && docker rm vllm-a
```

When a session ends, stop the node from the Mac so it isn't billed while idle.
The Nebius CLI needs a profile that signs in through AI Cloud
(`auth.nebius.com`) and can see the node's project:

```bash
# Create the profile once. It opens a browser to log in, so run it in your own
# terminal. Skip it if "nebius profile list" already shows ai-cloud.
nebius profile create --profile ai-cloud \
  --endpoint api.nebius.cloud \
  --federation-endpoint auth.nebius.com \
  --parent-id <project-id>

# Stop the node. Bring it back with "nebius compute instance start --id …".
nebius --profile ai-cloud compute instance stop --id <instance-id>
```

## Next

- Run the baseline sweep and put it through the loop: see "The loop" in the
  [PoC plan](poc.md#the-loop).
- Start the analytical model: see "Analytical model and profiles" in the
  [PoC plan](poc.md#analytical-model-and-profiles).
