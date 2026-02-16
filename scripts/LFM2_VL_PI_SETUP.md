# LFM2-VL-450M on Raspberry Pi (for `bruno_cam_rover.py`)

## 1) One-time setup on Pi
```bash
cd /Users/kaushiksivakumar/workspace/ai-bruno
chmod +x scripts/setup_lfm2_vl_pi.sh
./scripts/setup_lfm2_vl_pi.sh
```

## 2) Start local llama.cpp VLM server
```bash
cd /Users/kaushiksivakumar/workspace/ai-bruno
chmod +x scripts/run_lfm2_vl_server.sh
./scripts/run_lfm2_vl_server.sh
```

Defaults:
- Hugging Face repo: `LiquidAI/LFM2-VL-450M-GGUF`
- Quant file alias: `Q4_0`
- Port: `8080`

Override example:
```bash
LFM2_VL_HF_FILE=Q5_K_M LFM2_VL_PORT=8090 ./scripts/run_lfm2_vl_server.sh
```

## 3) Run Bruno with local-first VLM
```bash
python3 bruno_cam_rover.py \
  --mode external \
  --show \
  --vlm-provider auto \
  --vlm-local-base http://127.0.0.1:8080/v1 \
  --vlm-remote-base http://<MAC_IP>:1234/v1
```

If you want local only:
```bash
python3 bruno_cam_rover.py \
  --mode external \
  --show \
  --vlm-provider local \
  --vlm-local-base http://127.0.0.1:8080/v1
```
