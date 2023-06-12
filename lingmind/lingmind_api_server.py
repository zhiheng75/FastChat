"""A server that provides LingMind related RESTful APIs.

Usage:
python3 -m lingmind.api_server
"""
import argparse
import json
import requests
import logging
import copy
from typing import Union

import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastchat.constants import ErrorCode
from fastapi.exceptions import RequestValidationError
from fastchat.serve.openai_api_server import (
    create_chat_completion,
    create_completion,
    create_error_response,
)
from fastchat.protocol.openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
)
from fastchat.utils import build_logger
from pydantic import BaseSettings, BaseModel


logger = build_logger('lingmind_api_server', 'lingmind_api_server.log')
logger.setLevel(logging.DEBUG)
app = fastapi.FastAPI()
headers = {"User-Agent": "LingMind API Server"}

_use_auto_agent = False


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return create_error_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))


def get_last_question(request: ChatCompletionRequest) -> str:
    """
        Obtain the latest user question from the request.
    """
    user_question = None
    if isinstance(request.messages, str):
        user_question = request.messages
    else:
        # [{"role": "user", "content": "some content"}]
        user_question = request.messages[-1]['content']
    return user_question


def inject_identity_prompt(request: ChatCompletionRequest) -> ChatCompletionRequest:
    """
        Insert identity prompt data into a chat request. This is a temporary fix before we build this data into
        the model via fine-tuning.
    """
    print(request.messages)
    user_question = get_last_question(request)
    new_request = copy.deepcopy(request)
    #identity_prompts = [{"role": "user", "content": "你叫“小灵”，是一个由灵迈智能创建的AI智能助手，为用户回答任何关于政策、法规、服务、资源等方面的问题。"},
    #                    {"role": "assistant", "content": "好的。"}]
    identity_prompts = f'你叫“小灵”，是一个由灵迈智能创建的AI智能助手，为用户回答任何关于政策、法规、服务、资源等方面的问题。现在回答用户提出的这个问题：{user_question}'
    #if isinstance(request.messages, str):
    #    request.messages = identity_prompts.append({"role": "user", "content": request.messages})
    #else:
    #    # list
    #    identity_prompts.extend(request.messages)
    #    request.messages = identity_prompts
    new_request.messages = identity_prompts
    print(new_request.messages)
    return new_request


async def search_knowledge(question: str) -> Union[str, None]:
    """
       Search ES for the given question and return the answer if it hits with high confidence.
       Currently, the confidence threshold is set to 20.
    """
    es_api_url = 'http://gpu.qrgraph.com:9306/search'
    es_query_threshold = 20
    payload = dict(question=question)
    logger.debug(f'Question for ES: {question}')
    try:
        r = requests.post(es_api_url, data=payload)
        print(r.text)
        results = json.loads(r.text)
        if not results:
            return None
        if 'score' not in results or 'answer' not in results:
            return None
        logger.debug(f"ES score: {results['score']}")
        if results['score'] < es_query_threshold:
            return None
        return results['answer']
    except Exception as e:
        logger.error('Failure querying ES. ' + e)
        return None


def prepare_request(request: ChatCompletionRequest) -> ChatCompletionRequest:
    new_request = copy.deepcopy(request)
    # 为了避开Fastchat内置的聊天历史机制，只保留最后一个用户问题。
    new_request.messages = get_last_question(request)
    return new_request


@app.post("/demo/chat/completions")
async def demo_chat_completions(request: ChatCompletionRequest):
    """ 政务问答生成。主要包含以下步骤：
        - 判断问题是否与政务相关
            - 政务相关问题通过政务LLM生成答案，然后通过ChatGLM对答案进行重写
            - 非政务问题交由通用模型生成答案
        TODO(zhihengw):
          - 政务问题是否需要对历史进行裁剪
          - 只支持 n=1
    """
    GOV_QA_MODEL = 'llm01-6b-gov'  # chatglm-6b-zhongke
    CLASSIFICATION_MODEL = 'llm02-13b-gov'  # belle-13b-zhongke
    QA_MODEL = 'llm01-6b'  # chatglm-6b

    print(request)

    # original settings
    request_stream = request.stream
    request_n = request.n

    request.stream = False
    request.n = 1

    # 判断是否政务问题
    global _use_auto_agent
    if not _use_auto_agent:
        # 不使用模型自动选择
        new_request = prepare_request(request)
        return await create_chat_completion(new_request)
    else:
        # 模型自动选择
        user_question = get_last_question(request)
        logger.info(f'用户提问: {user_question}')
        # 查询ES看是否命中知识库
        response_text = await search_knowledge(user_question)

        if response_text:
            logger.info('用户提问由知识库回答')
        else:
            # 问题不在知识库，判断是否政务相关问题
            classification_question = ("你的任务是判断一个问题或者陈述是否与政府部门的业务内容，譬如工商、行政、车辆政策等问题都属于政府业务。"
                                       "相反，日常问候语、礼貌用语、天气和自然现象等内容则与政府业务无关。"
                                       f"现在用户提出的问题是:“{user_question}”，你需要判定这个问题是否属于政府业务内容, "
                                       "你的回答只能从”是“/”否“里面选择一个最可能的作为你的答案，不需要后续分析描述。"
                                       "譬如，问题:“你好”,回答:“否”; 问题：”如何申请驾照“，回答:“是”。")
            classification_request = ChatCompletionRequest(model=CLASSIFICATION_MODEL,
                                                           messages=classification_question,
                                                           max_tokens=1024,
                                                           temperature=0.1,
                                                           top_p=0.1,
                                                           n=1,
                                                           stream=False)
            classification_response = await create_chat_completion(classification_request)
            if not isinstance(classification_response, ChatCompletionResponse):
                # error
                return classification_response
            classification_response_text = classification_response.choices[0].message.content

            if classification_response_text.strip().startswith('是'):
                logger.info(f'用户提问属于政务问题。通过LLM处理, 选用模型为: {request.model}')
                original_messages = request.messages
                request.messages = user_question
                chat_response = await create_chat_completion(request)
                request.messages = original_messages
                # print(chat_response.choices[0])
                response_text = chat_response.choices[0].message.content
            else:
                logger.info(f'用户提问不属于政务问题。选用模型为: {QA_MODEL}')
                # General questions. Leave it to chatglm.
                request.model = QA_MODEL
                request = inject_identity_prompt(request)
                # remember to restore the original request
                request.stream = request_stream
                print(request.__dict__)
                return await create_chat_completion(request)

        # 整理答案
        print('整理前：' + response_text)

        # 整理输出格式
        #p = (f"你的任务是整理以下文本的格式然后输出，输出内容必须为中文。如果内容包含多个步骤，用列表作为输出格式。只需输出改写后的正文，不能包含回答提示信息。"
        #     f"譬如输入为'你好, 第一点是一， 第二点是二。', 输出为'你好！\n1.一。\n2.二。'。 待整理的文本内容为：{chat_response_text}")
        p = (f"你的任务是对以下文本的格式进行润饰，除必要的格式文本外不能添加额外的内容。只需输出润饰后的正文，不能包含回答提示信息。"
             f"含有多个要点要点的部分需要以完整的列表展示，譬如输入为“你好。这是第一点。这是第二点。“, 输出为“你好！\n 1.第一点。\n 2.第二点。“。"
             f"待整理的文本内容为：{response_text}")
        format_request = ChatCompletionRequest(model=QA_MODEL,
                                               messages=p,
                                               max_tokens=1024,
                                               temperature=0,
                                               top_p=0.1,
                                               n=1,
                                               stream=request_stream)
        format_response = await create_chat_completion(format_request)
        if not request_stream and isinstance(format_response, ChatCompletionResponse):
            print('整理后：' + format_response.choices[0].message.content)
        return format_response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LingMind RESTful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=9306, help="port number")
    parser.add_argument(
        "--use-auto-agent", action="store_true", help="use auto agent to select model."
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
    _use_auto_agent = args.use_auto_agent

    logger.info(f"args: {args}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
