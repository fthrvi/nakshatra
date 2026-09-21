#!/usr/bin/env bash
# nks-q3a.sh start|stop - blackwell's stage (layers 0-15) of the qwen3-30b-q3 chain, registered with Sthambha as
# blackwell-q3a at 10.42.0.7:5562 (Windows portproxy -> this WSL VM). Summoned/reaped over ssh by nakshatra-deep-bw.service.
W=$HOME/.nakshatra-worker
case "${1:-}" in
  start)
    systemctl --user is-active --quiet nks-q3-blackwell && exit 0
    systemctl --user reset-failed nks-q3-blackwell 2>/dev/null
    systemd-run --user --unit=nks-q3-blackwell --collect \
      --setenv=PYTHONPATH=$W/nakshatra-scripts \
      --setenv=LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/targets/x86_64-linux/lib:$HOME/llama-nks-build/bin \
      $W/venv/bin/python -u $W/nakshatra-scripts/worker.py --port 5562 \
      --sub-gguf $HOME/.nakshatra/slices/qwen3-30b-q3-L0-15.gguf --mode first --layer-start 0 --layer-end 15 \
      --model-id qwen3-30b-q3 --daemon-bin $HOME/llama-nks-build/bin/llama-nakshatra-worker --n-ctx 4096 --n-gpu-layers 99 \
      --pillar-url http://10.42.0.3:7777 --node-id blackwell-q3a --public-address 10.42.0.7:5562 \
      --gpu-vendor NVIDIA --gpu-model "RTX 5070" --gpu-vram-gb 12 --gpu-backend cuda --vram-offered-gb 8 \
      --ownership-class dedicated --idle-grace-s 600 >/dev/null 2>&1 ;;
  stop)
    systemctl --user stop nks-q3-blackwell 2>/dev/null
    pkill -f llama-nks-build/bin/llama-nakshatra-worker 2>/dev/null
    true ;;
  *) echo "usage: $0 start|stop" >&2; exit 2 ;;
esac
