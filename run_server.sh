#!/usr/bin/bash

# Start the controller
python3 -m fastchat.serve.controller &
sleep 20

# Zhongke chatglm model
python3 -m fastchat.serve.model_worker \
	--model-name 'chatglm-6b-zhongke' \
	--model-path /home/zhihengw/model/chatglm-6b-zhongke-ft
	--port 21002 \
	--worker-address http://localhost:21002 &
sleep 1

# native Chatglm model
python3 -m fastchat.serve.model_worker \
	--model-name 'chatglm-6b' \
	--model-path /home/zhihengw/model/chatglm-6b \
	--port 21003 \
	--worker-address http://localhost:21003 &
sleep 1

# BELLE
python3 -m fastchat.serve.model_worker \
	--model-name 'belle-13b-zhongke' \
	--model-path /home/zhihengw/model/BELLE-LLaMA-EXT-13B-zhongke-lora \
	--port 21004 \
	--worker-address http://localhost:21004 &
sleep 120


# API server
python3 -m fastchat.serve.openai_api_server --host 192.168.0.20 --port 9308
