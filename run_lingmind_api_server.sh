#!/usr/bin/bash

# Start the controller
echo "Starting controller..."
nohup python3 -m fastchat.serve.controller >nohup.controller 2>&1 &
sleep 5 

# Zhongke chatglm model
echo "Starting worker LLM01-6B-gov..."
nohup CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker \
	--model-name 'llm01-6b-gov' \
	--model-path /home/zhihengw/model/chatglm-6b-zhongke-ft
	--port 21002 \
	--worker-address http://localhost:21002 >nohup.llm01-6b-gov 2>&1 &
echo "$!" > llm01-6b-gov.pid
sleep 1

# native Chatglm model
echo "Starting worker LLM01-6B..."
nohup CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker \
	--model-name 'llm01-6b' \
	--model-path /home/zhihengw/model/chatglm-6b \
	--port 21003 \
	--worker-address http://localhost:21003 >nohup.llm01-6b 2>&1 &
echo "$!" > llm01-6b.pid
sleep 1

# BELLE
echo "Starting worker LLM02-13B-gov..."
nohup CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker \
	--model-name 'llm02-13b-gov' \
	--model-path /home/zhihengw/model/BELLE-LLaMA-EXT-13B-zhongke-lora \
	--port 21004 \
	--worker-address http://localhost:21004 >nohup.llm02-13b-gov 2>&1 &
echo "$!" > llm02-12-gov.pid
sleep 1


# API server
nohup python3 -m fastchat.serve.openai_api_server --host 192.168.0.20 --port 9308 >nohup.openai_api_server 2>&1 &
echo "$!" > openai_api.pid
sleep 1

# LingMind server
nohup python3 -m lingmind.lingmind_api_server --port=9307 --host=192.168.0.20 --use-auto-agent >nohup.lingmind_api_server 2>&1 &
echo "$!" > lingmind_api.pid
