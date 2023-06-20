"""A server that provides LingMind related RESTful APIs.

Usage:
python3 -m lingmind.api_server
"""
import argparse
import json
import requests
import logging
from typing import Union

import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastchat.constants import ErrorCode
from fastapi.exceptions import RequestValidationError
from fastchat.serve.openai_api_server import (
    create_chat_completion,
    create_error_response,
)
from fastchat.protocol.openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from fastchat.utils import build_logger
from pydantic import BaseSettings, BaseModel


logger = build_logger('lingmind_api_server', 'lingmind_api_server.log')
logger.setLevel(logging.DEBUG)
app = fastapi.FastAPI()
headers = {"User-Agent": "LingMind API Server"}


class AppSettings(BaseSettings):
    # The address of the model controller.
    controller_address: str = "http://localhost:21001"


app_settings = AppSettings()
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
    identity_prompts = [{"role": "user", "content": "你叫“小灵”，是一个由灵迈智能创建的AI智能助手，为用户回答任何关于政策、法规、服务、资源等方面的问题。"},
                        {"role": "assistant", "content": "好的。"}]
    if isinstance(request.messages, str):
        request.messages = identity_prompts.append({"role": "user", "content": request.messages})
    else:
        # list
        identity_prompts.extend(request.messages)
        request.messages = identity_prompts
    return request


async def search_knowledge(question: str) -> Union[str, None]:
    """
       Search ES for the given question and return the answer if it hits with high confidence.
       Currently, the confidence threshold is set to 25.
    """
    es_api_url = 'http://gpu.qrgraph.com:9306/search'
    es_query_threshold = 25
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
        logger.error('Failure querying ES: ' + json.dumps(e.__dict__))
        return None


async def summarize_chat_question(request: ChatCompletionRequest) -> str:
    """
        Summarize the user question history into a single question.
    """
    _ctemplate = """鉴于以下对话和后续问题，将后续问题改写为独立问题。
    
    对话历史:
    {chat_history}
    后续问题: {question}
    独立问题:"""
    follow_up_question = get_last_question(request)
    if not follow_up_question:
        return None
    summarization_prompt = _ctemplate.format(chat_history='\n'.join([f"{m['role']}: {m['content']}" for m in request.messages[:-1]]),
                                             question=follow_up_question)
    summarization_request = ChatCompletionRequest(model="belle-13b-zhongke",
                                                  messages=summarization_prompt,
                                                  max_tokens=1024,
                                                  temperature=0,
                                                  top_p=0.1,
                                                  n=1,
                                                  stream=False)
    summarization_response = await create_chat_completion(summarization_request)
    if not isinstance(summarization_response, ChatCompletionResponse):
        # error
        return summarization_response
    stand_alone_question = summarization_response.choices[0].message.content
    return stand_alone_question

    # _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    #
    # Chat History:
    # {chat_history}
    # Follow Up Input: {question}
    # Standalone question:"""
    # CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


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
    CLASSIFICATION_MODEL = 'llm02-13b-gov'  # belle-13b-zhongke
    QA_MODEL = 'llm01-6b'  # chatglm-6b

    logger.debug('用户请求:' + json.dumps(request.__dict__))

    # original settings
    request_stream = request.stream
    request_n = request.n

    request.stream = False
    request.n = 1

    # 判断是否政务问题
    global _use_auto_agent
    if not _use_auto_agent:
        # 不使用模型自动选择
        return await create_chat_completion(request)
    else:
        # 模型自动选择
        # condense the user question into a stand-alone question.
        user_question = await summarize_chat_question(request)
        # user_question = get_last_question(request)
        logger.info(f'用户提问: {user_question}')
        # 查询ES看是否命中知识库
        response_text = await search_knowledge(user_question)

        if response_text:
            logger.info('用户提问由知识库回答')
        else:
            logger.info('用户提问由LLM回答')
            # 问题不在知识库，判断是否政务相关问题
            classification_question = ("与工商、行政、车辆政策等有关的内容问题属于政府业务范围。"
                                       "相反，日常问候语、礼貌用语、天气和自然现象等内容则与政府业务范围无关。"
                                       "例如，当问题是:“你好”,你应该回答:“否”; 当问题是：”如何申请驾照？“，你应该回答:“是”。"
                                       f"根据以上规则，判断以下内容是否与政务业务相关：“{user_question}”。你的回答只能是“是”或者“否”,答案是")
            classification_request = ChatCompletionRequest(model=CLASSIFICATION_MODEL,
                                                           messages=classification_question,
                                                           max_tokens=1024,
                                                           temperature=0,
                                                           top_p=0.1,
                                                           n=1,
                                                           stream=False)
            classification_response = await create_chat_completion(classification_request)
            if not isinstance(classification_response, ChatCompletionResponse):
                # error
                return classification_response
            classification_response_text = classification_response.choices[0].message.content

            logger.debug(f'判断返回: {classification_response_text}')
            if classification_response_text.strip().strip("“").startswith('是'):
                logger.info(f'用户提问属于政务问题。通过LLM处理, 选用模型为: {request.model}')
                chat_response = await create_chat_completion(request)
                if not isinstance(chat_response, ChatCompletionResponse):
                    # error
                    return chat_response
                response_text = chat_response.choices[0].message.content
            else:
                logger.info(f'用户提问不属于政务问题。选用模型为: {QA_MODEL}')
                # General questions. Leave it to chatglm.
                request.model = QA_MODEL
                request = inject_identity_prompt(request)
                # remember to restore the original request
                request.stream = request_stream
                logger.debug(request.__dict__)
                return await create_chat_completion(request)

        # 整理答案
        print('整理前：' + response_text)

        # 整理输出格式
        p = (f"你的任务是对以下文本的格式进行润饰，除必要的格式文本外不能添加额外的内容。只需输出润饰后的正文，不能包含回答提示信息。"
             f"含有多个要点要点的部分需要以完整的列表展示，"
             f"\n\n待整理的文本为：“你好。这是第一点。这是第二点。“, 润饰后的文本为：“你好！\n 1.第一点。\n 2.第二点。“"
             f"\n\n待整理的文本内容为：“{response_text}”，润饰后的文本为：")
        format_request = ChatCompletionRequest(model=QA_MODEL,
                                               messages=[{"role": "user", "content": p}],
                                               max_tokens=1024,
                                               temperature=0,
                                               top_p=0.1,
                                               n=1,
                                               stream=request_stream)
        format_response = await create_chat_completion(format_request)
        if not request_stream and isinstance(format_response, ChatCompletionResponse):
            stripped_string = format_response.choices[0].message.content.strip().strip('“').strip('”')
            format_response.choices[0].message.content = stripped_string
            # print('整理后：' + format_response.choices[0].message.content.strip().strip('“').strip('”'))
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
        "--controller-address", type=str, default="http://localhost:21001"
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
    app_settings.controller_address = args.controller_address

    logger.info(f"args: {args}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
