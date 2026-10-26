# Getting started

How to reach the project's B300 VM and serve Qwen3.8-Flash-Next (FP8) on it
with SGLang. On 2026-10-26 the SSH login worked, the step 5 checks were
recorded, and step 6 was needed. Steps 7–12 have not been run yet. Every
command block has a comment above each command saying what it does.

This guide uses **SGLang**, but [ADR 0001](adr/0001-qwen3.8-flash-next-on-1xb300-with-vllm.md)
still names vLLM as the engine. The ADR needs an amendment before results are
compared.

| | |
|---|---|
| VM | `computeinstance-u02t2sm9f69wknpzt2` (Nebius AI Cloud) |
| SSH user | `jimmyshah-sshkey` |
| SSH key (on the Mac) | `~/.ssh/id_ed25519_js-b300-1gpu-usn1` |
| GPU | 1× NVIDIA B300 SXM6 PC, driver 580.173.02 |
| Engine | SGLang v0.5.22, image `lmsysorg/sglang@sha256:e3a4f060…a4f1ae` (CUDA 13.0) |
| Model | `Qwen/Qwen3.8-Flash-Next-FP8`, revision `236dfdf285828023ca3bcd3f37366c58a3469b13` (185.5 GB) |

## On the Mac

### 1. Create a Nebius CLI profile for AI Cloud

The default profile, `tokenfactory-solutions-architects`, signs in through
Token Factory and gets `PermissionDenied` on this VM. Add a profile that signs
in to AI Cloud instead. Run this in your own terminal, because it opens a
browser to log in.

```bash
# Create a profile that signs in through AI Cloud (auth.nebius.com) and
# defaults to the project that owns the VM. It becomes the active profile.
nebius profile create --profile ai-cloud \
  --endpoint api.nebius.cloud \
  --federation-endpoint auth.nebius.com \
  --parent-id <project-id>

# List profiles; the active one is marked [default].
nebius profile list
```

To go back to Token Factory, add `--profile tokenfactory-solutions-architects`
to a command.

### 2. Find the VM's public IP

```bash
# Read the VM's details as JSON and print the address of its first network
# interface, with the "/32" suffix removed.
nebius compute instance get --id computeinstance-u02t2sm9f69wknpzt2 --format json \
  | jq -r '.status.network_interfaces[0].public_ip_address.address | split("/")[0]'
```

If the VM is stopped and started again, the IP can change unless a public IP
allocation is attached.

### 3. Add an SSH shortcut

Add this to `~/.ssh/config`, replacing `<public-ip>`:

```
Host b300
  HostName <public-ip>
  User jimmyshah-sshkey
  IdentityFile ~/.ssh/id_ed25519_js-b300-1gpu-usn1
  IdentitiesOnly yes
  LocalForward 30000 localhost:30000
```

- `IdentitiesOnly yes` offers only this key, so the server doesn't reject the
  login after trying every key in the agent.
- `LocalForward` makes the VM's port 30000 (the SGLang server) reachable at
  `localhost:30000` on the Mac while the SSH session is open. The server is
  never exposed on the public IP.

### 4. Connect

```bash
# Open a shell on the VM. This also starts the port-30000 tunnel.
ssh b300

# Open only the tunnel, with no shell (-N). Use this in a spare terminal.
ssh -N b300
```

## On the VM

### 5. Check the GPU and Docker

```bash
# Show the GPU model, the memory the driver exposes, and the driver version.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Confirm Docker is installed.
docker --version
```

Recorded on 2026-10-26:

```
NVIDIA B300 SXM6 PC, 275040 MiB, 580.173.02
Docker version 29.9.2, build a82e528
```

- 275,040 MiB is 288.4 GB, the full advertised HBM. That answers ADR 0001's
  open question 1. NVIDIA's "2.1 TB per HGX node" figure appears to be
  8 × 268.6 GiB, so it is a units difference rather than missing memory. The
  CUDA context takes a little, so check SGLang's startup log (step 10) for the
  memory it can actually allocate.
- Driver 580 supports the CUDA 13.0 SGLang image.

### 6. Let your user run Docker

The first `docker run` failed with `permission denied while trying to connect
to the docker API at unix:///var/run/docker.sock`. Add your user to the
`docker` group:

```bash
# Add the current user to the docker group, so docker works without sudo.
sudo usermod -aG docker $USER

# Apply the new group in this shell only. Alternatively, log out and SSH back in.
newgrp docker
```

Membership in the `docker` group is effectively root on the VM. That is fine
on a single-user VM.

### 7. Check that containers can see the GPU

```bash
# Start a throwaway Ubuntu container with GPU access and list the GPUs it
# sees. --rm deletes the container afterwards.
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L
```

You should see `GPU 0: NVIDIA B300 SXM6 PC (UUID: …)`.

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
# Host RAM in GB. Offload runs need ≥ 51 GB free for the n-gram table, plus headroom.
free -g | head -2

# CPU cores. The scheduler and the load generator share them.
nproc

# Free space per filesystem. Look for ~230 GB free for the checkpoint and the image.
df -h

# Disks and where they are mounted, to choose where the checkpoint goes.
lsblk -o NAME,SIZE,MOUNTPOINT

# Power limit and maximum clocks. They set the compute ceiling in the
# analytical model, so record them with every run.
nvidia-smi --query-gpu=power.limit,clocks.max.sm,clocks.max.mem --format=csv
```

The SGLang image is 15.6 GB compressed and is stored under `/var/lib/docker`,
so the boot disk needs room for it as well.

### 9. Pull the pinned SGLang image and download the model

```bash
# The pinned v0.5.22 image (CUDA 13.0). Pinning by digest means a later
# "latest" can't change the engine in the middle of a phase.
export IMG=lmsysorg/sglang@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae

# Exact model revision from ADR 0001 (full hash; a short hash may not resolve).
export REV=236dfdf285828023ca3bcd3f37366c58a3469b13

# Hugging Face cache on the large disk chosen in step 8.
export HF_HOME=/path/on/big/disk/hf

# Download the image (one time, ~15.6 GB).
docker pull $IMG

# Print the SGLang version inside the image, to confirm it is 0.5.22.
docker run --rm $IMG python3 -c "import sglang; print(sglang.__version__)"

# Download the FP8 checkpoint into HF_HOME, using the hf CLI that ships in the
# image. 185.5 GB, one time.
docker run --rm -v $HF_HOME:/root/.cache/huggingface $IMG \
  hf download Qwen/Qwen3.8-Flash-Next-FP8 --revision $REV
```

Put the `export` lines in `~/.bashrc` so later sessions have them.

### 10. Start the server

```bash
# Start SGLang in the background (-d) as a container named "sglang":
#   --gpus all          give the container the B300
#   --ipc=host          share host memory for inter-process communication
#   -p 127.0.0.1:...    publish port 30000 on the VM's loopback only
#   HF_HUB_OFFLINE=1    use the downloaded files and never re-download
#   -v ...              mount the Hugging Face cache into the container
docker run -d --name sglang --gpus all --ipc=host \
  -p 127.0.0.1:30000:30000 -e HF_HUB_OFFLINE=1 \
  -v $HF_HOME:/root/.cache/huggingface $IMG \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.8-Flash-Next-FP8 --revision $REV --tp 1 \
    --linear-attn-prefill-backend flashinfer --linear-attn-decode-backend flashinfer \
    --mamba-ssm-dtype bfloat16 --reasoning-parser auto \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --host 0.0.0.0 --port 30000

# Follow the startup log. Ctrl-C stops following; the server keeps running.
docker logs -f sglang
```

What the SGLang flags do:

| Flag | Purpose |
|---|---|
| `--model-path`, `--revision` | Which checkpoint and which exact commit to load |
| `--tp 1` | One GPU, no tensor parallelism |
| `--linear-attn-*-backend flashinfer` | FlashInfer kernels for the Gated DeltaNet (linear attention) layers |
| `--mamba-ssm-dtype bfloat16` | Store the recurrent state in BF16. That is half the ~115 MB per sequence that ADR 0001 assumed in FP32 |
| `--reasoning-parser auto` | Return the model's thinking in `reasoning_content`, separate from the answer |
| `--speculative-algorithm NEXTN …` | MTP speculative decoding: draft 3 tokens per step and verify 4 |
| `--host 0.0.0.0 --port 30000` | Listen on every interface inside the container. Docker exposes the port only on the VM's loopback |

Startup loads 185.5 GB of weights and captures CUDA graphs, so it takes
several minutes. It is ready when the log says `The server is fired up and
ready to roll!`.

From the log, record the free memory before the weights load, the size of the
KV cache pool, and the maximum number of tokens it can hold.

Most of these flags come from a published single-B300 run of the NVFP4 build
([sglang#41937](https://github.com/sgl-project/sglang/issues/41937)), not the
FP8 build. If the server rejects a flag, compare against the B300 + FP8 cell
in the [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next).
FP8 KV cache flags (`--kv-cache-dtype fp8_e4m3 --qsa-indexer-dtype fp8_e4m3`)
are left out until it is confirmed that this image accepts them.

### 11. Test it from the Mac

With `ssh b300` or `ssh -N b300` open:

```bash
# Health check. Returns HTTP 200 once the server is ready.
curl -i localhost:30000/health

# List the model being served.
curl -s localhost:30000/v1/models

# Send one chat request. The model always reasons, so leave room in max_tokens.
curl -s localhost:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3.8-Flash-Next-FP8",
  "messages": [{"role": "user", "content": "What is 15 percent of 240?"}],
  "max_tokens": 1024
}'
```

The answer should be 36, with the reasoning in `reasoning_content`.

### 12. Record the environment, then stop

```bash
# Append the run environment to a log, as ADR 0001 requires: UTC time, image
# digest, SGLang/torch/CUDA versions, and GPU settings.
mkdir -p ~/runs && {
  date -u +%FT%TZ
  echo "$IMG  model rev $REV"
  docker exec sglang python3 -c "import sglang, torch; print('sglang', sglang.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
  nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit,clocks.max.sm,clocks.max.mem --format=csv,noheader
} | tee -a ~/runs/env.txt

# Stop and remove the server container. The image and the checkpoint stay on disk.
docker stop sglang && docker rm sglang
```

On the Mac, stop the VM when the session ends so the GPU isn't billed while
idle:

```bash
# Stop the VM. Bring it back with "nebius compute instance start --id …".
nebius compute instance stop --id computeinstance-u02t2sm9f69wknpzt2
```

## Next

- Amend ADR 0001 for SGLang, and record the 288.4 GB result (step 5).
- Confirm the FP8 KV cache and QSA indexer flags, then add them to step 10.
- Phase 1: start the dense calibration on Qwen3.8-27B.
