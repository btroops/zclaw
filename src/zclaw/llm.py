"""OpenAI-compatible vLLM chat model for LangChain."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional

import requests
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

logger = logging.getLogger(__name__)

# Avoid proxy interference with LAN vLLM endpoints
for _key in (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
):
    os.environ.pop(_key, None)


class VLLMChatModel(BaseChatModel):
    """Chat model that talks to a vLLM OpenAI-compatible `/v1/chat/completions` server."""

    base_url: str = "http://127.0.0.1:8000/v1"
    model_name: str = "Qwen/Qwen2.5-7B"
    api_key: str = "token-abc123"
    temperature: float = 0.4
    max_tokens: int = 1000
    repetition_penalty: float = 1.1
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    request_timeout: int = 120

    model_config = {"extra": "allow"}

    def _normalize_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        normalized: List[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, (HumanMessageChunk, SystemMessageChunk)):
                normalized.append(
                    HumanMessage(content=msg.content)
                    if isinstance(msg, HumanMessageChunk)
                    else SystemMessage(content=msg.content)
                )
            elif isinstance(msg, ChatMessage):
                if msg.role == "system":
                    normalized.append(SystemMessage(content=msg.content))
                elif msg.role == "user":
                    normalized.append(HumanMessage(content=msg.content))
            elif isinstance(msg, (SystemMessage, HumanMessage, AIMessage)):
                normalized.append(msg)
        return normalized

    def _convert_to_api_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        api_messages: List[Dict[str, str]] = []
        for m in messages:
            role_map = {"system": "system", "human": "user", "ai": "assistant"}
            api_role = role_map.get(m.type, "user")
            api_messages.append({"role": api_role, "content": m.content})
        return api_messages

    def _prepare_payload(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        normalized_msgs = self._normalize_messages(messages)
        logger.debug(
            "Normalized messages: %s",
            [(type(m).__name__, m.content) for m in normalized_msgs],
        )
        api_messages = self._convert_to_api_messages(normalized_msgs)
        logger.debug("API messages: %s", api_messages)

        return {
            "model": self.model_name,
            "messages": api_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": stop or self.stop or ["<|im_end|>"],
            "repetition_penalty": self.repetition_penalty,
            "stream": stream,
        }

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        payload = self._prepare_payload(messages, stop, stream=True)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            logger.info("POST %s/chat/completions (stream)", self.base_url)
            response = requests.post(
                url=f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
                timeout=self.request_timeout,
            )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        choices = chunk_data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            finish_reason = choices[0].get("finish_reason")

                            if content is not None:
                                message_chunk = AIMessageChunk(content=content)
                                yield ChatGenerationChunk(message=message_chunk)

                            if finish_reason is not None:
                                logger.debug("finish_reason: %s", finish_reason)
                                break
                    except json.JSONDecodeError as e:
                        logger.warning("JSON decode failed: %s — %s", data_str, e)
                        continue
        except Exception as e:
            raise RuntimeError(f"vLLM stream failed: {e}") from e

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        full_content = ""
        for chunk in self._stream(messages, stop, run_manager, **kwargs):
            piece = chunk.message.content
            if piece:
                full_content += piece

        ai_message = AIMessage(content=full_content)
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "vllm-chat-model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model_name": self.model_name,
            "repetition_penalty": self.repetition_penalty,
        }
