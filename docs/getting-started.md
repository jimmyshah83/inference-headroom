# Getting started

How to create the project's 8×H200 node, reach it, and serve
**DeepSeek-V4.1-Flash-1017** with vLLM for the coding-agent PoC. Capacity
approved one node on 2026-11-02 in the **Production tf-us1** project (region
`us-central1`), to be held until **Fri 2026-11-20**. The node doesn't exist
yet: you create it, after the network [prerequisites](#prerequisites-a-network-and-subnet-for-the-node).
None of the steps below have been run yet. Every command block has a comment
above each command saying what it does.

This guide replaces the 1×B300 / Qwen3.8-Flash-Next guide (still in git
history). Why the plan changed: [ADR 0002](adr/0002-coding-agent-poc-deepseek-v4.1-flash-on-8xh200-with-vllm.md).
What counts as done: [PoC plan](poc.md).

| | |
|---|---|
| Project | Production tf-us1 (`<project-id>`), in the Token Factory tenant. It is a production project, so the node gets a network of its own |
| Region | `us-central1` |
| Network and subnet | `h200-poc-network` (`<network-id>`) and `h200-poc-subnet` (`<subnet-id>`). Fill in after P2 |
| Node | `<instance-id>` (Nebius, platform `gpu-h200-sxm`, preset `8gpu-128vcpu-1600gb`). Fill in after P4 |
| SSH user | `<ssh-user>` |
| SSH key (on the Mac) | `~/.ssh/<key>` |
| GPUs | 8× NVIDIA H200 SXM: 141 GB HBM3e and 4.8 TB/s each (1,128 GB per node), NVLink 900 GB/s |
| Host | 128 vCPU (Xeon Platinum 8468), 1,600 GiB RAM |
| Engine | vLLM v0.32.0, image `vllm/vllm-openai@sha256:befccf0e…5c5b34` (CUDA 13.0) |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision `1bae0e1088993b9b82de08f59de5be5a34e42170` (~511 GB) |
| Layout | Two replicas, each TP4 with the Engram tables offloaded to host RAM (the vLLM recipe's H200 default). In the PD disaggregation phase, the same two GPU groups become one prefill engine and one decode engine |

## Prerequisites: a network and subnet for the node

Do these once, on the Mac, before you create the node. They follow Nebius's
guide [Isolating Nebius AI Cloud resources in separate networks](https://docs.nebius.com/vpc/networking/isolation#create-net).

Why the node needs its own network:

- **A VM must be created in a subnet, and none was visible in this project.**
  Nebius normally gives each new project a `default-network` and a
  `default-subnet`
  ([VPC overview](https://docs.nebius.com/vpc/overview#default-virtual-networks-resources)).
  On 2026-11-02 this project listed no networks, subnets or IP pools, possibly
  because the account couldn't see them.
- **Isolation from production.** Resources in the same network can reach each
  other over private IP addresses. In a separate network, the PoC node can't
  reach the project's production resources over private addresses, and they
  can't reach it.
- **The choice is permanent.** A VM can't be moved to another network or
  subnet after it is created. To change it, you delete and recreate the VM.

### P1. Check CLI access to the project

You need:

- The Nebius CLI, installed and configured. Check with `nebius version`.
- A profile that can see the project. On 2026-11-02 the `ai-cloud` profile
  (signed in through `auth.nebius.com`) could see Production tf-us1. The
  `tokenfactory-solutions-architects` profile listed no tenants at all.
- At least the `editor` role in the project, for example through the default
  `editors` group. Check this under
  [Administration → IAM](https://console.nebius.com/iam) in the web console.
  Reading the project doesn't prove you can write to it. If P2 fails with a
  permission error, ask the capacity team for editor access to the project.

```bash
# Create a profile that signs in through AI Cloud (auth.nebius.com). Skip this
# if "nebius profile list" already shows ai-cloud. Run it in your own terminal,
# because it opens a browser to log in.
nebius profile create --profile ai-cloud \
  --endpoint api.nebius.cloud \
  --federation-endpoint auth.nebius.com \
  --parent-id <project-id>

# List profiles; the active one is marked [default].
nebius profile list

# The project the node goes in (Production tf-us1).
export PROJECT_ID=<project-id>

# Confirm the profile can read the project. It should print
# "Production tf-us1" and "us-central1".
nebius --profile ai-cloud iam project get --id $PROJECT_ID --format json \
  | jq -r '[.metadata.name, .status.region] | @tsv'

# List the networks, subnets and IP pools the project already has.
# On 2026-11-02 all three lists were empty ("{}"). That may mean the account
# couldn't see them, not that the project had none.
nebius --profile ai-cloud vpc network list --parent-id $PROJECT_ID
nebius --profile ai-cloud vpc subnet list --parent-id $PROJECT_ID
nebius --profile ai-cloud vpc pool list --parent-id $PROJECT_ID
```

The commands pass `--profile` and `--parent-id` explicitly, so nothing is
created by accident in whatever project the profile defaults to.

### P2. Create the network and the subnet

Every network must contain at least one subnet. Name both so the project's
owners can tell what they are for.

```bash
# Create the PoC network in the project and save its ID.
export NETWORK_ID=$(nebius --profile ai-cloud vpc network create \
  --parent-id $PROJECT_ID \
  --name h200-poc-network \
  --format jsonpath='{.metadata.id}')

# Create a subnet in that network and save its ID. By default, a subnet takes
# its private and public IP addresses from the network's pools.
export SUBNET_ID=$(nebius --profile ai-cloud vpc subnet create \
  --parent-id $PROJECT_ID \
  --network-id $NETWORK_ID \
  --name h200-poc-subnet \
  --format jsonpath='{.metadata.id}')

# Print both IDs. If either is empty, the command that should have set it
# failed; read the error it printed.
echo "network $NETWORK_ID  subnet $SUBNET_ID"
```

Like Nebius's example, these commands don't set IP pools or CIDR blocks. P3
shows which address ranges the subnet received.

**Known failure (2026-11-02).** The network create failed with:

```
Error: rpc error: code = PermissionDenied desc = Permission denied to <project-id>
  Permission denied PermissionDenied: service VPC API, resource ID: <project-id>
```

The account could read the project but held no role that allows writing to
it. The only roles granted on the tenant belong to the `admins`, `editors`,
`auditors` and `viewers` groups, and the account wasn't in any of them. The
fix has to come from the capacity team or the project's admins: the `editor`
role on the project, or someone else creating the network and node.

### P3. Check that the subnet is ready

```bash
# Print the subnet's state and its private and public address ranges.
nebius --profile ai-cloud vpc subnet get --id $SUBNET_ID --format json \
  | jq '.status | {state, ipv4_private_cidrs, ipv4_public_cidrs}'
```

Expect:

- `"state": "READY"`
- at least one private CIDR block. Nebius documents `10.96.0.0/13` as the
  default private range in `us-central1`
  ([private ranges per region](https://docs.nebius.com/vpc/addressing/available-addresses)).
- at least one public CIDR block

If `ipv4_public_cidrs` is empty, the node can't get a public IP address, and
SSH from the Mac (steps 2–4) won't work. Ask the project's owners for a public
pool rather than creating pools yourself.

### P4. Save the IDs, then create the node in the subnet

```bash
# Save the IDs in a file outside the repo, so a new terminal can load them
# (step 1).
cat > ~/.h200-poc.env <<EOF
export PROJECT_ID=$PROJECT_ID
export NETWORK_ID=$NETWORK_ID
export SUBNET_ID=$SUBNET_ID
EOF
```

When you create the node, put it in `h200-poc-subnet` and give it a public IP
address. **You can't change the subnet afterwards.**

- **Web console:** in the VM's network settings, choose `h200-poc-subnet` and
  keep a public IP address.
- **CLI:** add this parameter to `nebius compute instance create`:

  ```
  --network-interfaces "[{\"name\": \"eth0\", \"subnet_id\": \"$SUBNET_ID\", \"ip_address\": {}, \"public_ip_address\": {}}]"
  ```

  `"public_ip_address": {}` gives the node a dynamic public IP, which can
  change when the node is stopped and started (step 2). Use
  `{"static": true}` instead to keep the address fixed
  ([creating a VM](https://docs.nebius.com/compute/virtual-machines/manage#create-a-vm)).

Once the node exists, add its ID to `~/.h200-poc.env`:

```bash
# Append the node's ID to the saved IDs.
echo 'export INSTANCE_ID=<instance-id>' >> ~/.h200-poc.env
```

Notes:

- **No GPU cluster is needed.** A GPU cluster connects several VMs over
  InfiniBand ([GPU clusters](https://docs.nebius.com/compute/clusters/gpu/)).
  This PoC is one VM, and its 8 GPUs talk to each other over NVLink.
- **If you use Managed Kubernetes instead of a VM,** pass
  `--control-plane-subnet-id $SUBNET_ID` when you create the cluster.
- **Firewall.** Every network has a default security group that allows all
  ingress and egress traffic, and every VM in the network joins it
  ([security groups](https://docs.nebius.com/vpc/security-groups/overview#default-security-groups)).
  Step 10 publishes vLLM's ports on the node's loopback only, so they stay
  private. Anything else that listens on all interfaces is reachable from the
  internet.

## On the Mac

### 1. Load the IDs and pick the profile

```bash
# Load the project, network, subnet and node IDs saved in P4.
source ~/.h200-poc.env

# List profiles; the active one is marked [default]. It should be ai-cloud.
nebius profile list
```

If `ai-cloud` isn't the active profile, add `--profile ai-cloud` to every
`nebius` command below.

### 2. Find the node's public IP

```bash
# Read the node's details as JSON and print the address of its first network
# interface, with the "/32" suffix removed.
nebius compute instance get --id $INSTANCE_ID --format json \
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
nebius compute instance stop --id $INSTANCE_ID
```

Release the node on or before **Fri 2026-11-20**. If the PoC finishes early,
tell capacity, so the node goes back to the pool. To release it, delete the
node and then the network you created for it:

```bash
# Delete the node. Its GPUs go back to the pool.
nebius compute instance delete --id $INSTANCE_ID

# List the project's disks. Delete any the node left behind, such as a data
# disk for the checkpoint, with "nebius compute disk delete --id …".
nebius compute disk list --parent-id $PROJECT_ID

# Delete the subnet, then the network. The node used the subnet, and the
# subnet belongs to the network, so delete them in this order.
nebius vpc subnet delete --id $SUBNET_ID
nebius vpc network delete --id $NETWORK_ID
```

## Next

- Do the network prerequisites (P1–P4), create the node, and fill in the
  table at the top.
- Run the baseline and the quality smoke tests: Week 1 of the [PoC plan](poc.md).
- Start the analytical model: see "Analytical model and profiles" in the
  [PoC plan](poc.md#analytical-model-and-profiles).
