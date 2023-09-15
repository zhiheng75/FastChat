#curl http://gpu.qrgraph.com:9308/v1/chat/completions \
#    "messages": [{"role": "user", "content": "你是一个回答用户提问的政务助手。用分行的列表来回答以下这个问题：有哪些情形不能办理新车上牌手续？"}],
#curl http://localhost:9999/v1/chat/policy_completions \
curl http://10.60.100.180:9318/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 1024,
    "temperature": 0.1,
    "top_p": 0.1,
    "n": 1
  }'
