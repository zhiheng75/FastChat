#!/usr/bin/bash

local_ip_address="192.168.0.20"

controller_port=22001
# Start the controller
echo "Starting controller..."
nohup python3 -m fastchat.serve.controller --port ${controller_port} >nohup.controller 2>&1 &
echo "$!" > controller.pid
sleep 5

# Zhongke chatglm model
worker_name0="llm01-6b-gov"
worker_port0=22002
echo "Starting worker ${worker_name0}..."
CUDA_VISIBLE_DEVICES=6 nohup python3 -m fastchat.serve.model_worker \
	--model-name "${worker_name0}" \
	--model-path /home/zhihengw/model/chatglm-6b-zhongke-ft \
	--port ${worker_port0} \
	--controller-address http://localhost:${controller_port} \
	--worker-address http://localhost:${worker_port0} >${worker_name0}.nohup 2>&1 &
echo "$!" > ${worker_name0}.pid
sleep 1

# native Chatglm model
worker_name1="llm01-6b"
worker_port1=22003
echo "Starting worker ${worker_name1}..."
CUDA_VISIBLE_DEVICES=6 nohup python3 -m fastchat.serve.model_worker \
	--model-name "${worker_name1}" \
	--model-path /home/zhihengw/model/chatglm-6b \
	--port $worker_port1 \
	--controller-address http://localhost:${controller_port} \
	--worker-address http://localhost:{worker_port1} >${worker_name1}.nohup 2>&1 &
echo "$!" > ${worker_name1}.pid
sleep 1

# BELLE
worker_name2="llm02-13b-gov"
worker_port2=22004
echo "Starting worker ${worker_name2}..."
CUDA_VISIBLE_DEVICES=7 nohup python3 -m fastchat.serve.model_worker \
	--model-name "${worker_name2}" \
	--model-path /home/zhihengw/model/BELLE-LLaMA-EXT-13B-zhongke-lora \
	--port ${worker_port2} \
	--controller-address http://localhost:${controller_port} \
	--worker-address http://localhost:${worker_port2} >${worker_name2}.nohup 2>&1 &
echo "$!" > ${worker_name2}.pid
sleep 1

# API server
openai_api_server_port=9318
nohup python3 -m fastchat.serve.openai_api_server --host ${local_ip_address} \
	                                          --controller-address http://localhost:${controller_port} \
                                                  --port ${openai_api_server_port} >nohup.openai_api_server 2>&1 &
echo "$!" > openai_api.pid
sleep 1

# LingMind server
lingmind_api_server_port=9317
nohup python3 -m lingmind.lingmind_api_server --port ${lingmind_api_server_port} \
                                              --host ${local_ip_address} \
	                                      --controller-address http://localhost:${controller_port} \
                                              --use-auto-agent >nohup.lingmind_api_server 2>&1 &
echo "$!" > lingmind_api.pid
