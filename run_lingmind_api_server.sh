#!/usr/bin/bash

LOG_DIR="logs"
mkdir -p ${LOG_DIR}

# Read input parameters.
# mode = [dev|staging],
# command=[start|stop|restart]
if [ $# -le 1 ]; then
    echo "No enough arguments provided. Usage: $0 [dev|staging] [start|stop|restart]"
    exit 1
fi

mode=$1
if [ "$mode" != "dev" ] && [ "$mode" != "staging" ]; then
    echo "Invalid mode. Usage: $0 [dev|staging] [start|stop|restart]"
    exit 1
fi

command=$2
if [ "$command" != "start" ] && [ "$command" != "stop" ] && [ "$command" != "restart" ]; then
    echo "Invalid command. Usage: $0 [dev|staging] [start|stop|restart]"
    exit 1
fi

# As we are running on the same machine, we need to specify different ports for different modes.
case "$mode" in
    "dev")
        echo "Running in dev mode..."
        GPU0=6
        GPU1=7
        controller_port=22001
        worker_port0=22002
        worker_port1=22003
        worker_port2=22004
        openai_api_server_port=9318
        lingmind_api_server_port=9317
        ;;
    "staging")
        echo "Running in staging mode..."
        GPU0=0
        GPU1=1
        controller_port=21001
        worker_port0=21002
        worker_port1=21003
        worker_port2=21004
        openai_api_server_port=9308
        lingmind_api_server_port=9307
        ;;
esac


function stop_server {
  # Read PIDs from all the .pid files in the logging directory and kill the processes
  for pid_file in ${LOG_DIR}/*.pid; do
    if [ -f "$pid_file" ]; then
      pid=$(cat "$pid_file")
      echo "Killing process $pid (from $pid_file)..."
      kill -9 "$pid"
      rm "$pid_file"
    fi
  done
}

# define a function to start all processes
function start_server {
  local_ip_address="192.168.0.20"

  controller_address="http://localhost:${controller_port}"
  # Start the controller
  echo "Starting controller..."
  nohup python3 -m fastchat.serve.controller --port ${controller_port} >${LOG_DIR}/controller.nohup 2>&1 &
  echo "$!" > ${LOG_DIR}/controller.pid
  sleep 5

  # Gov chatglm model
  worker_name0="llm01-6b-gov"
  echo "Starting worker ${worker_name0}..."
  CUDA_VISIBLE_DEVICES=$GPU0 nohup python3 -m fastchat.serve.model_worker \
	  --model-name "${worker_name0}" \
	  --model-path /home/zhihengw/model/chatglm-6b-zhongke-ft \
	  --port ${worker_port0} \
	  --controller-address ${controller_address} \
	  --worker-address http://localhost:${worker_port0} >${LOG_DIR}/${worker_name0}.nohup 2>&1 &
  echo "$!" > ${LOG_DIR}/${worker_name0}.pid
  sleep 1

  # native Chatglm model
  worker_name1="llm01-6b"
  echo "Starting worker ${worker_name1}..."
  CUDA_VISIBLE_DEVICES=$GPU0 nohup python3 -m fastchat.serve.model_worker \
	  --model-name "${worker_name1}" \
	  --model-path /home/zhihengw/model/chatglm-6b \
	  --port $worker_port1 \
	  --controller-address ${controller_address} \
	  --worker-address http://localhost:${worker_port1} >${LOG_DIR}/${worker_name1}.nohup 2>&1 &
  echo "$!" > ${LOG_DIR}/${worker_name1}.pid
  sleep 1

  # BELLE
  worker_name2="llm02-13b-gov"
  echo "Starting worker ${worker_name2}..."
  CUDA_VISIBLE_DEVICES=$GPU1 nohup python3 -m fastchat.serve.model_worker \
	  --model-name "${worker_name2}" \
	  --model-path /home/zhihengw/model/BELLE-LLaMA-EXT-13B-zhongke-lora \
	  --port ${worker_port2} \
	  --controller-address ${controller_address} \
	  --worker-address http://localhost:${worker_port2} >${LOG_DIR}/${worker_name2}.nohup 2>&1 &
  echo "$!" > ${LOG_DIR}/${worker_name2}.pid
  sleep 1

  # API server
  nohup python3 -m fastchat.serve.openai_api_server --host ${local_ip_address} \
        	                                    --controller-address ${controller_address} \
                                                    --port ${openai_api_server_port} >${LOG_DIR}/nohup.openai_api_server 2>&1 &
  echo "$!" > ${LOG_DIR}/openai_api.pid
  sleep 1

  # LingMind server
  nohup python3 -m lingmind.lingmind_api_server --port ${lingmind_api_server_port} \
                                                --host ${local_ip_address} \
        	                                --controller-address ${controller_address} \
                                                --use-auto-agent >${LOG_DIR}/nohup.lingmind_api_server 2>&1 &
  echo "$!" > ${LOG_DIR}/lingmind_api.pid
}


case "$command" in
    "start")
        echo "Starting all processes..."
        start_server
        ;;
    "stop")
        echo "Stopping all processes..."
        stop_server
        ;;
    "restart")
        echo "Restarting all processes..."
        stop_server
        sleep 5
        start_server
        ;;
esac
