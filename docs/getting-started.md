# Getting started

How to reach the project's 8×B200 node and serve
**DeepSeek-V4.1-Flash-1017** with vLLM for the coding-agent PoC. On 2026-11-03,
capacity approved a temporary allocation of 8 B200 GPUs in the **Architects
us-central1** project (region `us-central1`). The node runs there. The PoC ends
**Fri 2026-11-20**. Every command block has a comment above each command
saying what it does.

This guide replaces the 1×B300 / Qwen3.8-Flash-Next guide (still in git
history). Why the plan changed: [ADR 0002](adr/0002-coding-agent-poc-deepseek-v4.1-flash-on-8xb200-with-vllm.md).
What counts as done: [PoC plan](poc.md).

| | |
|---|---|
| Project | Architects us-central1 (`project-u00fxtzxpr008w3p2t9663`), in the Token Factory tenant |
| Region | `us-central1` (fabric `us-central1-b`) |
| Network and subnet | The project's `default-network` and `default-subnet` (`vpcsubnet-u00xtfz8eh2vntkzg0`). The node was created in them, so the [prerequisites](#prerequisites-a-network-and-subnet-for-the-node) weren't needed |
| Node | `computeinstance-u00wvygqqd9xavf4sy` (`jimmyshah-b200-instance-1`; Nebius platform `gpu-b200-sxm`, preset `8gpu-160vcpu-1792gb`) |
| SSH user | `jimmyshah-sshkey` |
| SSH key (on the Mac) | `~/.ssh/id_ed25519_nebius` |
| GPUs | 8× NVIDIA B200: HBM3e, 183,359 MiB (~179 GiB) each in `nvidia-smi` (~1,432 GiB per node), up to 8 TB/s each, NVLink 5 at 1.8 TB/s per GPU |
| Host | 160 vCPU (2× Xeon Platinum 8580, 2 NUMA nodes), 1,792 GiB RAM |
| Engine | vLLM v0.33.0 (released 2026-11-02), image `vllm/vllm-openai@sha256:d825baf9…a84c6e` (CUDA 13.0, PyTorch 2.14) |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision `1bae0e1088993b9b82de08f59de5be5a34e42170` (~511 GB) |
| Layout | Two replicas, each TP4, with the vLLM recipe's B200 settings: fp8 KV, FlashInfer sparse-MLA attention, and the Engram tables offloaded to host RAM. The recipe's own B200 default is TP2 for concurrency ~1 ([ADR 0002](adr/0002-coding-agent-poc-deepseek-v4.1-flash-on-8xb200-with-vllm.md), decision 3). In the PD disaggregation phase, the same two GPU groups become one prefill engine and one decode engine |

## Prerequisites: a network and subnet for the node

**Not needed for the current node.** It was created in the existing
`default-subnet` of Architects us-central1. These steps are kept for a node in
a project with no usable network, as Production tf-us1 was on 2026-11-02.

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
  --name poc-network \
  --format jsonpath='{.metadata.id}')

# Create a subnet in that network and save its ID. By default, a subnet takes
# its private and public IP addresses from the network's pools.
export SUBNET_ID=$(nebius --profile ai-cloud vpc subnet create \
  --parent-id $PROJECT_ID \
  --network-id $NETWORK_ID \
  --name poc-subnet \
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
cat > ~/.b200-poc.env <<EOF
export PROJECT_ID=$PROJECT_ID
export NETWORK_ID=$NETWORK_ID
export SUBNET_ID=$SUBNET_ID
EOF
```

When you create the node, put it in `poc-subnet` and give it a public IP
address. **You can't change the subnet afterwards.**

- **Web console:** in the VM's network settings, choose `poc-subnet` and
  keep a public IP address.
- **CLI:** add this parameter to `nebius compute instance create`:

  ```
  --network-interfaces "[{\"name\": \"eth0\", \"subnet_id\": \"$SUBNET_ID\", \"ip_address\": {}, \"public_ip_address\": {}}]"
  ```

  `"public_ip_address": {}` gives the node a dynamic public IP, which can
  change when the node is stopped and started (step 2). Use
  `{"static": true}` instead to keep the address fixed
  ([creating a VM](https://docs.nebius.com/compute/virtual-machines/manage#create-a-vm)).

Once the node exists, add its ID to `~/.b200-poc.env`:

```bash
# Append the node's ID to the saved IDs.
echo 'export INSTANCE_ID=<instance-id>' >> ~/.b200-poc.env
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
# The first time only: save the current node's IDs, since P4 wasn't needed.
# Skip this if ~/.b200-poc.env already exists.
cat > ~/.b200-poc.env <<'EOF'
export PROJECT_ID=project-u00fxtzxpr008w3p2t9663
export SUBNET_ID=vpcsubnet-u00xtfz8eh2vntkzg0
export INSTANCE_ID=computeinstance-u00wvygqqd9xavf4sy
EOF

# Load the saved project, subnet and node IDs.
source ~/.b200-poc.env

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

Add this to `~/.ssh/config`, replacing `<public-ip>` with the address from
step 2. `IdentityFile` is the private key, the file without `.pub`:

```
Host b200
  HostName <public-ip>
  User jimmyshah-sshkey
  IdentityFile ~/.ssh/id_ed25519_nebius
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
ssh b200

# Open only the tunnels, with no shell (-N). Use this in a spare terminal.
ssh -N b200
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

Nothing to save yet: step 12 writes this output to `~/runs/env.txt`. For now,
check which NUMA node and CPU cores each GPU is closest to (the `NUMA
Affinity` and `CPU Affinity` columns). The NUMA layout matters for this model.
Each replica keeps its Engram tables in pinned host memory and reads rows from
them every step. A replica on GPUs 0–3 should use the socket those GPUs attach
to. The baseline in step 10 doesn't pin replicas. Pinning them is one of the
PoC experiments (O3 in the [PoC plan](poc.md)).

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

You should see `GPU 0` through `GPU 7`, each `NVIDIA B200`.

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
# the vLLM image (9.7 GB compressed), and SWE-bench task images (~100 GB for 50 tasks).
df -h

# Disks and where they are mounted, to choose where the checkpoint goes.
lsblk -o NAME,SIZE,MOUNTPOINT

# Power limit and maximum clocks. They set the compute ceiling in the
# analytical model, so record them with every run.
nvidia-smi --query-gpu=index,power.limit,clocks.max.sm,clocks.max.mem --format=csv
```

On this node the boot disk is the only real disk: `/dev/vda1`, mounted at `/`.
It is 1.3 TB with 1.2 TB free on 2026-11-03, which is enough for everything
above, so step 9 puts the caches in your home directory. Don't use the other
mounts:

- `tmpfs` mounts such as `/dev/shm` and `/run` live in RAM. A checkpoint there
  would take the host memory the Engram tables need, and it disappears on
  reboot.
- `/mnt/cloud-metadata` holds the VM's cloud metadata. It isn't for data.

If a future node's boot disk is too small, attach a network disk of at least
1 TB and point `HF_HOME` at it.

### 9. Pull the pinned vLLM image and download the model

Save the PoC's settings in `~/.bashrc` once, so every new SSH session and tmux
window has them. Steps 10–12 depend on them.

```bash
# Append the settings to ~/.bashrc. The quoted 'EOF' keeps $HOME as written, so
# it expands each time a shell starts.
cat >> ~/.bashrc <<'EOF'

# DeepSeek-V4.1-Flash PoC (getting-started step 9)
# The pinned v0.33.0 image (CUDA 13.0). Pinning by digest means a later tag
# can't change the engine in the middle of the PoC.
export IMG=vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e
# The model and its exact revision (full hash; a short hash may not resolve).
export MODEL=deepseek-ai/DeepSeek-V4.1-Flash-1017
export REV=1bae0e1088993b9b82de08f59de5be5a34e42170
# The Hugging Face cache and vLLM's compile and autotune cache, on the boot
# disk (/dev/vda1 at /), the only real disk on this node (step 8).
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

Then download the checkpoint, about 510 GB, with the `hf` tool that ships in the
image. Use a Hugging Face read token: on 2026-11-03 the download ran at about
490 MB/s without one and about 1.6 GB/s with one.

```bash
# Start a tmux session, so the download survives a dropped SSH connection.
# Ctrl-b d detaches; "tmux attach -t dl" comes back.
tmux new -s dl

# Paste a read token from huggingface.co/settings/tokens. read -s keeps it off
# the screen and out of your shell history. Don't put it in ~/.bashrc: vLLM runs
# offline in step 10 and doesn't need it.
read -rsp 'HF token: ' HF_TOKEN && export HF_TOKEN; echo

# Check that the token works and reaches the container. It prints your Hugging
# Face username; "Not logged in" or an error means a wrong or missing token.
docker run --rm -e HF_TOKEN --entrypoint hf $IMG auth whoami

# Download the checkpoint into HF_HOME. -e HF_TOKEN passes the token in, and the
# -v mount puts the files on the host, where step 10's containers find them.
docker run --rm -e HF_TOKEN --entrypoint hf -v $HF_HOME:/root/.cache/huggingface $IMG \
  download $MODEL --revision $REV
```

The progress bar counts finished files, so it can sit at `0/83` or `81/83` for
minutes while the largest files download. Shards 47 and 48 hold the two Engram
tables, about 101.5 GB each. Don't press Ctrl-C to check on it; detach instead.

If the download stops before it finishes, delete the partial files before you
run it again. The downloader reserves each file's full size up front, so a
partial file can look complete.

```bash
# Delete partial downloads. The files are owned by root, because the container
# runs as root.
sudo rm -f $HF_HOME/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/blobs/*.incomplete
```

Then run the download command again. It skips the files that already finished.

Check the download before step 10:

```bash
# Count the weight shards. It should print 48.
ls $HF_HOME/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/$REV/ \
  | grep -c '^model-.*-of-00048.safetensors$'

# Check every large file against its hash. Each one is named after its own
# SHA-256, so the check needs no other input. It runs 8 files at a time and takes
# a few minutes. It prints "all files OK", or the name of each file that fails.
cd $HF_HOME/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/blobs
ls | grep -E '^[0-9a-f]{64}$' \
  | xargs -P 8 -I{} sh -c 'echo "{}  {}" | sha256sum -c --quiet' && echo "all files OK"
cd -
```

### 10. Start the two replicas

Each replica is one vLLM server on 4 GPUs. The baseline uses the vLLM recipe's
B200 settings at TP4, with tool calling and reasoning turned on and the vision
encoder skipped. The recipe's B200 reference configuration (TP2, tuned for one
user at a time) is in [ADR 0002](adr/0002-coding-agent-poc-deepseek-v4.1-flash-on-8xb200-with-vllm.md).
**Speculative decoding is off in the baseline.** Turning on DSpark is an
experiment (O4 in the [PoC plan](poc.md)).

Save the start command as a script on the node, `~/serve.sh`. The experiments
restart replicas many times, and a script is still there after you reconnect.
It reads the settings from step 9, and it stops with an error if one of them is
missing.

```bash
# Write the script. The quoted 'EOF' keeps $IMG, $MODEL and the others as
# written, so the script reads them from ~/.bashrc's exported settings each
# time it runs.
cat > ~/serve.sh <<'EOF'
#!/usr/bin/env bash
# Start one vLLM replica as a background container (getting-started step 10).
# Usage: ~/serve.sh NAME GPUS PORT [extra "vllm serve" flags...]
#   --gpus "device=..."   give the container only these GPUs
#   --ipc=host            share host memory for inter-process communication
#   -p 127.0.0.1:...      publish the port on the node's loopback only
#   HF_HUB_OFFLINE=1      use the downloaded files and never re-download
#   VLLM_ENGINE_READY_TIMEOUT_S=7200   the first start can take a long time
#   VLLM_USE_V2_MODEL_RUNNER=1, VLLM_USE_RUST_FRONTEND=1   from the recipe's
#                         B200 configuration
# set -u stops the script if a setting or argument is missing.
set -euo pipefail
name=$1 gpus=$2 port=$3; shift 3
docker run -d --name "$name" --gpus "\"device=$gpus\"" --ipc=host \
  -p 127.0.0.1:$port:$port \
  -e HF_HUB_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_USE_RUST_FRONTEND=1 \
  -v $HF_HOME:/root/.cache/huggingface -v $VLLM_CACHE:/root/.cache/vllm \
  --entrypoint vllm $IMG serve $MODEL --revision $REV \
    --tensor-parallel-size 4 \
    --tokenizer-mode deepseek_v41 \
    --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
    --kv-cache-dtype fp8 \
    --kernel-config '{"enable_flashinfer_autotune":true}' \
    --engram-config '{"cpu_offload":true,"use_thp":true}' \
    --language-model-only \
    --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v41 \
    --host 0.0.0.0 --port $port "$@"
EOF

# Make the script executable.
chmod +x ~/serve.sh

# Start replica A on GPUs 0-3 and replica B on GPUs 4-7. The containers run in
# the background (docker run -d), so they keep running after you disconnect.
~/serve.sh vllm-a 0,1,2,3 8000
~/serve.sh vllm-b 4,5,6,7 8001

# Follow replica A's startup log. Ctrl-C stops following; the server keeps running.
docker logs -f vllm-a
```

What the flags do:

| Flag | Purpose |
|---|---|
| `--revision` | Load the exact commit pinned above |
| `--tensor-parallel-size 4` | Split each layer across 4 GPUs. Two TP4 replicas fill the node |
| `--tokenizer-mode deepseek_v41` | Encode prompts with DeepSeek's Python encoder. The repo ships no Jinja chat template |
| `--attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41",…}'` | The recipe's Blackwell attention backend for the TP layout, with the indexer's keys in MXFP4 and sparse indexer logits. The recipe documents it with TP2, so check that it starts at TP4 (ADR 0002, open question 3) |
| `--kv-cache-dtype fp8` | Store the KV cache in FP8, as the recipe does on Blackwell. Switching back to auto is experiment O7 |
| `--kernel-config '{"enable_flashinfer_autotune":true}'` | Let FlashInfer benchmark its kernel variants at startup and keep the fastest. The results land in the vLLM cache |
| `--engram-config '{"cpu_offload":true,"use_thp":true}'` | Keep the Engram n-gram tables (183 GiB) in pinned host RAM instead of HBM, backed by transparent huge pages. The GPUs read rows through UVA. The recipe has no B200 memory figures. On Hopper it reports 23.6 GiB moved off each GPU, and those figures don't add up to the 183 GiB table size. So measure host and GPU memory yourself |
| `--language-model-only` | Skip the vision encoder. The coding workload is text only, so this frees memory for KV cache. Image input is a later step on one TP8 replica ([PoC plan](poc.md#later-image-input-on-one-tp8-replica)) |
| `--tool-call-parser deepseek_v41 --enable-auto-tool-choice` | Turn the model's DSML tool-call blocks into OpenAI `tool_calls` |
| `--reasoning-parser deepseek_v41` | Return the thinking in `reasoning_content`, separate from the answer |
| `--host 0.0.0.0 --port …` | Listen on every interface inside the container. Docker exposes the port only on the node's loopback |

vLLM's defaults apply to everything else: prefix caching on, a GPU memory
utilization of 0.9, and the default scheduler token budget. The experiments in
the PoC plan change these one at a time.

To change a flag, remove the replica and start it again with the flag added.
Docker won't reuse a container name while a container with that name exists,
even a stopped one.

```bash
# Remove replica A, then start it again with a higher GPU memory utilization.
# Any arguments after the port go to "vllm serve".
docker rm -f vllm-a
~/serve.sh vllm-a 0,1,2,3 8000 --gpu-memory-utilization 0.92
```

Both replicas read the same 511 GB from disk, so the first start is slow. Once
the files are in the page cache, the second replica and later restarts are
faster. A replica is ready when its log shows `Application startup complete`.

Record these from each replica's startup log:

- The KV cache memory available and the number of tokens the KV cache can hold.
- **Which MoE kernel was picked for the MXFP4 experts.** B200 has FP4 tensor
  cores, so vLLM v0.33.0 can run the experts with FP4 math. It may instead pick
  a kernel that computes in FP8 or BF16 (for example W4A16: 4-bit weights, BF16
  compute). The kernel sets the compute ceiling for the expert layers (the FP4,
  FP8 or BF16 rate), which changes the analytical model.
- The Engram offload message and how much host memory it pinned.
- How long the start took, split into weight loading and CUDA graph capture.

### 11. Test it from the Mac

With `ssh b200` or `ssh -N b200` open:

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
# vLLM/torch/CUDA versions, GPU settings, the GPU topology, and the CPU sockets
# and NUMA nodes.
mkdir -p ~/runs && {
  date -u +%FT%TZ
  echo "$IMG  model $MODEL rev $REV"
  docker exec vllm-a python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit,clocks.max.sm,clocks.max.mem --format=csv,noheader
  nvidia-smi topo -m
  lscpu | grep -E 'Model name|Socket|NUMA'
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

- Run the baseline and the quality smoke tests: Week 1 of the [PoC plan](poc.md).
- Start the analytical model: see "Analytical model and profiles" in the
  [PoC plan](poc.md#analytical-model-and-profiles).
