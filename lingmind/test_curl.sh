#curl http://gpu.qrgraph.com:9308/v1/chat/completions \
curl http://localhost:9999/v1/chat/policy_completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatglm-6b-zhongke",
    "messages": [{"role": "user", "content": "有哪些情形不能办理新车上牌手续？"}],
    "max_tokens": 1024,
    "temperature": 0.1,
    "top_p": 0.1,
    "n": 1
  }'