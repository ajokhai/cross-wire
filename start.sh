#!/bin/zsh
# Starts everything except the sim: the local model server, a Cross Talk channel and the
# pilot in it. Prints the channel address and the two commands to run next.
#   ./start.sh
cd "$(dirname "$0")"
MODEL=~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf

if ! curl -s --max-time 1 http://127.0.0.1:8080/health | grep -q ok; then
  echo "starting the model server..."
  llama-server -m "$MODEL" --port 8080 -c 4096 --log-disable > /tmp/llama-server.log 2>&1 &
  for i in {1..30}; do curl -s --max-time 1 http://127.0.0.1:8080/health | grep -q ok && break; sleep 1; done
fi

CHANNEL=$(crosstalk new robot-lab --topic "Instruct robots in plain English; the ESP32 brain decides" 2>&1 | grep -oE "xt_[A-Za-z0-9_-]{22}" | head -1)
echo "$CHANNEL" > .channel
crosstalk llm "$CHANNEL" --openai http://127.0.0.1:8080/v1 --model qwen2.5-1.5b --name pilot --reply mention \
  --history 20 --max-tokens 100 --system "$(cat pilot_prompt.txt)" > /tmp/pilot.log 2>&1 &

echo
echo "channel: $CHANNEL   (pilot log: /tmp/pilot.log)"
echo
echo "1. robot + 3D view:   .venv/bin/mjpython sim/mj_sim.py --channel $CHANNEL"
echo "2. talk to it:        crosstalk up $CHANNEL"
echo "   e.g.  @husky turn around, investigate that blue square, then go back to where you started"
