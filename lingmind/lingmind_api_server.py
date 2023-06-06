"""A server that provides LingMind related RESTful APIs.

Usage:
python3 -m lingmind.api_server
"""

import asyncio

import argparse
import asyncio
import json
import openai

from typing import Generator, Optional, Union, Dict, List, Any

import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastchat.constants import WORKER_API_TIMEOUT, WORKER_API_EMBEDDING_BATCH_SIZE, ErrorCode
from fastchat.serve.openai_api_server import create_chat_completion
from pydantic import BaseSettings, BaseModel
import uvicorn
from fastapi.exceptions import RequestValidationError
from fastchat.protocol.openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DeltaMessage,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
    TokenCheckRequest,
    TokenCheckResponse,
    UsageInfo,
)
from fastchat.utils import build_logger


logger = build_logger(__name__, __name__+'.log')


#class AppSettings(BaseSettings):
#    # The address of the model controller.
#    llm_api_base: str = "http://gpu.qrgraph.com:9308/v1"


#app_settings = AppSettings()
app = fastapi.FastAPI()
headers = {"User-Agent": "LingMind API Server"}

#openai.api_base = app_settings.llm_api_base


def create_error_response(code: int, message: str) -> JSONResponse:
    logger.error(f'Status = {code}: {message}')
    return JSONResponse(
        ErrorResponse(message=message, code=code).dict(), status_code=400
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return create_error_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))


class RewriteRequest(BaseModel):
    model: str
    messages: str
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.1
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


@app.post("/demo/chat/completions")
async def chat_policy_completions(request: ChatCompletionRequest):
    """ 政务问答生成。主要包含两个步骤：
        - 通过LLM API 的 /chat/completion 接口生成答案
        - 通过LLM API 的 /completion 接口对答案进行重写
    """
    print(request)
    request_stream = request.stream
    request_n = request.n
    request.stream = False
    request.n = 1

    response = await create_chat_completion(request)

    print(response.choices[0])
    response_text = response.choices[0].message.content
    print('整理前：' + response_text)
    """
    completion = openai.ChatCompletion.create(model=request.model,
                                              messages=request.messages,
                                              max_tokens=request.max_tokens,
                                              temperature=request.temperature,
                                              top_p=request.top_p,
                                              n=request.n,
                                              stop=request.stop,
                                              presence_penalty=request.presence_penalty,
                                              frequency_penalty=request.frequency_penalty,
                                              stream=False)   # disable streaming as we will further process
    """

    p = (f"你的任务是整理以下文本的格式然后输出，输出内容必须为中文。尽可能用列表作为输出格式。只需输出改写后的正文，不能包含回答提示信息。"
          f"譬如输入为'你好, 第一点是一， 第二点是二。', 输出为'你好！\n1.一。\n2.二。'。 待整理的文本内容为：{response_text}")
    messages = [{"role": "user", "content": p}]
    format_request = ChatCompletionRequest(model="chatglm-6b",
                                           messages=messages,
                                           max_tokens=1024,
                                           temperature=0,
                                           top_p=0.1,
                                           n=1,
                                           stream=request_stream)
    format_response = await create_chat_completion(format_request)
    if not request_stream:
        print('整理后：' + format_response.choices[0].message.content)
    return format_response


def rewrite_text(text):
    """ 通过LLM 的 chatcompletion 对输入的文本进行重写"""
    p = (f"你的任务是整理以下文本的格式然后输出，输出内容必须为中文。尽可能用列表作为输出格式。只需输出改写后的正文，不能包含回答提示信息。譬如输入为'你好, 第一点是一， 第二点是二。', 输出为'你好！\n1.一。\n2.二。'。 待整理的文本内容为：{text}")
    messages = [{"role": "user", "content": p}]
    completion = openai.ChatCompletion.create(model="chatglm-6b",
                                              messages=messages,
                                              max_tokens=1024,
                                              temperature=0,
                                              top_p=0.1,
                                              n=1,
                                              stream=False)
    return completion.choices[0]['message']['content']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LingMind RESTful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=9306, help="port number")
    parser.add_argument(
        "--llm-api-base", type=str, default="http://gpu.qrgraph.com:9308/v1"
    )
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )
    #app_settings.llm_api_base = args.llm_api_base

    openai.api_key = "EMPTY"  # Not support yet
    openai.api_base = args.llm_api_base

    logger.info(f"args: {args}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
