#!/usr/bin/env bash
# nks-q3b.sh start|stop - ijru's stage (layers 15-48) of the qwen3-30b-q3 chain, registered with Sthambha as ijru-q3b at
# 10.51.0.14:5572. Summoned/reaped over ssh by nakshatra-deep-bw.service (scale-to-zero: the GPU is only held while in use).
case "${1:-}" in
  start)
    systemctl --user is-active --quiet nks-q3-ijru && exit 0
    systemctl --user reset-failed nks-q3-ijru 2>/dev/null
    systemd-run --user --unit=nks-q3-ijru --collect --setenv=HOME=/home/prithviraj \
      /usr/bin/python3 /home/prithviraj/nakshatra/scripts/worker.py --port 5572 \
      --sub-gguf /home/prithviraj/.nakshatra/slices/qwen3-30b-q3-L15-48.gguf --mode last --layer-start 15 --layer-end 48 \
      --model-id qwen3-30b-q3 --daemon-bin /home/prithviraj/llama.cpp/build-cuda-static/bin/llama-nakshatra-worker \
      --n-ctx 4096 --n-gpu-layers 99 \
      --pillar-url http://10.42.0.3:7777 --node-id ijru-q3b --public-address 10.51.0.14:5572 \
      --gpu-vendor NVIDIA --gpu-model RTX-3060 --gpu-vram-gb 12 --gpu-backend cuda --vram-offered-gb 10 \
      --ownership-class dedicated --idle-grace-s 600 >/dev/null 2>&1 ;;
  stop)
    systemctl --user stop nks-q3-ijru 2>/dev/null
    pkill -f llama.cpp/build-cuda-static/bin/llama-nakshatra-worker 2>/dev/null
    true ;;
  *) echo "usage: $0 start|stop" >&2; exit 2 ;;
esac
