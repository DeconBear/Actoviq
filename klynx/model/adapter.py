"""
 - 

 KlynxAgent  .invoke(messages) .
 LiteLLM , (OpenAI, Anthropic, Google, DeepSeek ).
 httpx.Client .
"""

import base64
import json
import logging
import mimetypes
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
os.environ.setdefault("LITELLM_LOG", "WARNING")
import litellm
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

#  SSL 
litellm.suppress_debug_info = True
for _litellm_logger_name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
    logging.getLogger(_litellm_logger_name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _coerce_usage_int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except Exception:
        return 0


def normalize_usage_payload(payload: Any) -> dict:
    """Normalize provider usage payloads into prompt/completion/total token counts."""
    if payload is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    if isinstance(payload, dict):
        for key in ("usage", "usage_metadata", "token_usage"):
            nested = payload.get(key)
            if nested is not None and nested is not payload:
                normalized = normalize_usage_payload(nested)
                if normalized["total_tokens"] > 0:
                    return normalized

        response_metadata = payload.get("response_metadata")
        if response_metadata is not None and response_metadata is not payload:
            normalized = normalize_usage_payload(response_metadata)
            if normalized["total_tokens"] > 0:
                return normalized

        prompt_tokens = _coerce_usage_int(
            payload.get("prompt_tokens", payload.get("input_tokens", 0))
        )
        completion_tokens = _coerce_usage_int(
            payload.get("completion_tokens", payload.get("output_tokens", 0))
        )
        total_tokens = _coerce_usage_int(payload.get("total_tokens", 0))
    else:
        for attr_name in ("usage", "usage_metadata", "token_usage"):
            nested = getattr(payload, attr_name, None)
            if nested is not None and nested is not payload:
                normalized = normalize_usage_payload(nested)
                if normalized["total_tokens"] > 0:
                    return normalized

        response_metadata = getattr(payload, "response_metadata", None)
        if response_metadata is not None and response_metadata is not payload:
            normalized = normalize_usage_payload(response_metadata)
            if normalized["total_tokens"] > 0:
                return normalized

        prompt_tokens = _coerce_usage_int(
            getattr(payload, "prompt_tokens", getattr(payload, "input_tokens", 0))
        )
        completion_tokens = _coerce_usage_int(
            getattr(payload, "completion_tokens", getattr(payload, "output_tokens", 0))
        )
        total_tokens = _coerce_usage_int(getattr(payload, "total_tokens", 0))

    if total_tokens <= 0 and (prompt_tokens > 0 or completion_tokens > 0):
        total_tokens = prompt_tokens + completion_tokens
    if prompt_tokens <= 0 and total_tokens > 0 and completion_tokens > 0:
        prompt_tokens = max(total_tokens - completion_tokens, 0)
    if completion_tokens <= 0 and total_tokens > 0 and prompt_tokens > 0:
        completion_tokens = max(total_tokens - prompt_tokens, 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


class _InferenceBackend:
    """."""

    def completion(self, call_kwargs: dict):
        raise NotImplementedError

    def stream_completion(self, call_kwargs: dict):
        raise NotImplementedError


class _LiteLLMBackend(_InferenceBackend):
    """:LiteLLM completion."""

    def completion(self, call_kwargs: dict):
        return litellm.completion(**call_kwargs)

    def stream_completion(self, call_kwargs: dict):
        return litellm.completion(**call_kwargs)


class _ResponsesCompatBackend(_InferenceBackend):
    """
    Responses API .

    , LiteLLM completion.
     Responses API .
    """

    def completion(self, call_kwargs: dict):
        return litellm.completion(**call_kwargs)

    def stream_completion(self, call_kwargs: dict):
        return litellm.completion(**call_kwargs)


class LiteLLMResponse:
    """
    
     graph.py  response.content / response.reasoning_content / response.usage 
     Tool Calling( Function Calling) tool_calls
    """
    def __init__(self, content: str, reasoning_content: str = ""):
        self.content = content
        self.reasoning_content = reasoning_content
        self.additional_kwargs = {}
        self.response_metadata = {}
        self.usage = None  # 
        self.tool_calls = []  #  Tool Calling 


class LiteLLMChat:
    """
    
    
     LiteLLM ,
     .invoke(messages)  .stream(messages) .
     Tool Calling( Function Calling,tools ).
    """

    _DATA_URL_PATTERN = re.compile(
        r"^data:(?P<mime>[\w.+-]+/[\w.+-]+)?;base64,(?P<data>[A-Za-z0-9+/=\s]+)$",
        re.IGNORECASE,
    )
    _URI_SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://")
    _RUNTIME_ONLY_KWARGS = {
        "initial_prompt",
        "thread_id",
        "session_id",
        "checkpointer",
        "store",
        "session_manager",
    }
    _STRICT_EMPTY_MESSAGE_PROVIDERS = frozenset(
        {"xiaomi_mimo", "openrouter", "minimax", "moonshot"}
    )
    _TOOL_HISTORY_COMPAT_PROVIDERS = frozenset({"openrouter", "xiaomi_mimo"})
    _COMPAT_NOOP_TOOL_NAME = "__klynx_noop_compat__"
    _UNSET = object()

    def __init__(self, model: str, api_key: str = None, api_base: str = None,
                 temperature: float = 0.1, top_p: Optional[float] = None,
                 backend: str = "litellm", **kwargs):
        """
        Args:
            model:  ( "deepseek/deepseek-reasoner", "gpt-4o")
                    litellm provider/model 
            api_key: API Key
            api_base:  API  ()
            temperature: 
            **kwargs: 
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.top_p = self._parse_optional_float_option(
            kwargs.pop("top_p", os.getenv("KLYNX_MODEL_TOP_P", top_p)),
            default=top_p,
            minimum=0.0,
            maximum=1.0,
        )
        self.max_context_tokens = 128000
        self.backend_name = str(backend or "litellm").strip().lower()
        self.backend = self._create_backend(self.backend_name)
        self.model_capabilities = self._normalize_model_capabilities(
            kwargs.pop("model_capabilities", None)
        )
        self.supports_native_tool_calling = self._resolve_capability_bool(
            key="supports_native_tool_calling",
            explicit_value=kwargs.pop("supports_native_tool_calling", self._UNSET),
            default=True,
        )
        self.supports_temperature = self._resolve_capability_bool(
            key="supports_temperature",
            explicit_value=kwargs.pop("supports_temperature", self._UNSET),
            default=self._heuristic_supports_sampling_controls(),
        )
        self.supports_top_p = self._resolve_capability_bool(
            key="supports_top_p",
            explicit_value=kwargs.pop("supports_top_p", self._UNSET),
            default=self._heuristic_supports_sampling_controls(),
        )
        self.supports_multimodal = self._resolve_capability_bool(
            key="supports_multimodal",
            explicit_value=kwargs.pop("supports_multimodal", self._UNSET),
            default=True,
        )
        self.supports_parallel_tool_calls = self._resolve_capability_bool(
            key="supports_parallel_tool_calls",
            explicit_value=kwargs.pop("supports_parallel_tool_calls", self._UNSET),
            default=False,
        )
        self.supports_streaming_tool_calls = self._resolve_capability_bool(
            key="supports_streaming_tool_calls",
            explicit_value=kwargs.pop("supports_streaming_tool_calls", self._UNSET),
            default=True,
        )
        self.supports_usage_in_stream = self._resolve_capability_bool(
            key="supports_usage_in_stream",
            explicit_value=kwargs.pop("supports_usage_in_stream", self._UNSET),
            default=False,
        )
        self.allowed_openai_params = self._normalize_openai_params(
            kwargs.pop(
                "allowed_openai_params",
                self.model_capabilities.get("allowed_openai_params"),
            )
        )
        self.stream_tool_recovery_enabled = self._parse_bool_option(
            kwargs.pop(
                "stream_tool_recovery_enabled",
                os.getenv("KLYNX_STREAM_TOOL_RECOVERY_ENABLED", "1"),
            ),
            default=True,
        )
        self.provider_name = self._infer_provider_name(self.model)
        self._active_tool_name_map: Dict[str, str] = {}
        self._last_prepare_meta: Dict[str, Any] = {}
        self._tool_name_repair_count = 0
        self._noop_tool_injected_count = 0
        self._stream_tool_recovery_triggered_count = 0
        self.log_call_params = self._parse_bool_option(
            kwargs.pop(
                "log_call_params",
                os.getenv("KLYNX_MODEL_LOG_CALL_PARAMS", "0"),
            ),
            default=False,
        )

        # ()
        self.retry_enabled = self._parse_bool_option(
            kwargs.pop("retry_enabled", os.getenv("KLYNX_MODEL_RETRY_ENABLED", "1")),
            default=True,
        )
        self.retry_max_attempts = self._parse_int_option(
            kwargs.pop("retry_max_attempts", os.getenv("KLYNX_MODEL_RETRY_MAX_ATTEMPTS", "3")),
            default=3,
            minimum=1,
        )
        self.retry_initial_delay = self._parse_float_option(
            kwargs.pop("retry_initial_delay", os.getenv("KLYNX_MODEL_RETRY_INITIAL_DELAY", "1.0")),
            default=1.0,
            minimum=0.0,
        )
        self.retry_backoff_factor = self._parse_float_option(
            kwargs.pop("retry_backoff_factor", os.getenv("KLYNX_MODEL_RETRY_BACKOFF_FACTOR", "2.0")),
            default=2.0,
            minimum=1.0,
        )
        self.retry_max_delay = self._parse_float_option(
            kwargs.pop("retry_max_delay", os.getenv("KLYNX_MODEL_RETRY_MAX_DELAY", "8.0")),
            default=8.0,
            minimum=0.0,
        )
        self.retry_jitter = self._parse_float_option(
            kwargs.pop("retry_jitter", os.getenv("KLYNX_MODEL_RETRY_JITTER", "0.2")),
            default=0.2,
            minimum=0.0,
        )
        fallback_models_raw = kwargs.pop("fallback_models", os.getenv("KLYNX_FALLBACK_MODELS", ""))
        self.fallback_models = self._parse_fallback_models(fallback_models_raw)

        # optional multimodal/base64 preprocessing
        self.is_multimodal = self._parse_bool_option(
            kwargs.pop("is_multimodal", os.getenv("KLYNX_MODEL_IS_MULTIMODAL", "0")),
            default=False,
        )
        if self.is_multimodal and not self.supports_multimodal:
            logger.warning(
                "Multimodal preprocessing is enabled while model_capabilities.supports_multimodal=false (model=%s).",
                self.model,
            )
        self.multimodal_auto_base64 = self._parse_bool_option(
            kwargs.pop(
                "multimodal_auto_base64",
                os.getenv("KLYNX_MODEL_MULTIMODAL_AUTO_BASE64", "1"),
            ),
            default=True,
        )
        self.multimodal_extract_output = self._parse_bool_option(
            kwargs.pop(
                "multimodal_extract_output",
                os.getenv("KLYNX_MODEL_MULTIMODAL_EXTRACT_OUTPUT", "1"),
            ),
            default=True,
        )
        self.multimodal_include_base64_output = self._parse_bool_option(
            kwargs.pop(
                "multimodal_include_base64_output",
                os.getenv("KLYNX_MODEL_MULTIMODAL_INCLUDE_BASE64_OUTPUT", "1"),
            ),
            default=True,
        )
        self.multimodal_output_dir = str(
            kwargs.pop(
                "multimodal_output_dir",
                os.getenv("KLYNX_MODEL_MULTIMODAL_OUTPUT_DIR", ""),
            )
            or ""
        ).strip()

        self.extra_kwargs = kwargs

        # (, SSL )
        self.http_client = httpx.Client(trust_env=False)
    
    def invoke(self, messages, tools: Optional[list] = None):
        """
        , LiteLLMResponse
        
        Args:
            messages: LangChain BaseMessage   OpenAI 
            tools: ,OpenAI  tools JSON Schema ( Tool Calling)
            
        Returns:
            LiteLLMResponse 
        """
        openai_messages = self._sanitize_jsonable(self._convert_messages(messages))
        return self._invoke_once_from_openai_messages(
            openai_messages=openai_messages,
            tools=tools,
        )
    
    def stream(self, messages, tools: Optional[list] = None):
        """
        
        
        Args:
            messages: LangChain 
            tools: ,OpenAI  tools JSON Schema ( Tool Calling)
            
        Yields:
            chunks with 'content', 'reasoning_content', 'usage', and 'tool_calls'
        """
        openai_messages = self._sanitize_jsonable(self._convert_messages(messages))

        # Providers with unstable stream tool chunks can force a non-stream path.
        if tools and not self.supports_streaming_tool_calls:
            fallback = self._invoke_once_from_openai_messages(
                openai_messages=openai_messages,
                tools=tools,
            )
            for payload in self._response_to_stream_chunks(
                fallback,
                stream_recovery_triggered=False,
                stream_mode="disabled",
            ):
                yield payload
            return

        call_kwargs = self._build_call_kwargs(
            openai_messages=openai_messages,
            tools=tools,
            stream=True,
        )

        response = self._stream_with_retry(call_kwargs)
        tool_call_accumulators: Dict[int, Dict[str, Any]] = {}
        stream_saw_content = False
        stream_saw_reasoning = False
        stream_saw_tool_deltas = False

        for chunk in response:
            usage = normalize_usage_payload(getattr(chunk, "usage", None) or chunk)
            if usage["total_tokens"] > 0:
                yield {"usage": usage}

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            raw_content = getattr(delta, "content", "")
            content = self._coerce_content_to_text(raw_content)
            reasoning_content = getattr(delta, "reasoning_content", "") or ""
            tool_deltas = getattr(delta, "tool_calls", None) or []

            if content:
                stream_saw_content = True
            if reasoning_content:
                stream_saw_reasoning = True
            if tool_deltas:
                stream_saw_tool_deltas = True
                for tc_delta in tool_deltas:
                    self._merge_stream_tool_call_delta(tool_call_accumulators, tc_delta)

            if content or reasoning_content or tool_deltas:
                payload = {
                    "content": content,
                    "reasoning_content": reasoning_content,
                }
                if raw_content != content:
                    payload["raw_content"] = raw_content
                yield payload

        assembled_tool_calls = self._finalize_stream_tool_calls(tool_call_accumulators)
        if assembled_tool_calls:
            yield {"tool_calls": assembled_tool_calls}
            return

        # One-shot non-stream recovery when stream yielded no actionable assistant output.
        should_recover = (
            bool(self.stream_tool_recovery_enabled)
            and not stream_saw_content
            and not stream_saw_reasoning
            and not stream_saw_tool_deltas
        )
        if should_recover:
            recovered = self._invoke_once_from_openai_messages(
                openai_messages=openai_messages,
                tools=tools,
            )
            self._stream_tool_recovery_triggered_count += 1
            for payload in self._response_to_stream_chunks(
                recovered,
                stream_recovery_triggered=True,
                stream_mode="recovered_non_stream",
            ):
                yield payload
    
    def _parse_native_tool_calls(self, raw_tool_calls) -> list:
        """
         LiteLLM  tool_calls 
        
        Returns:
            [{"tool": "read_file", "params": {"path": "..."}}, ...]
        """
        results = []
        for tc in raw_tool_calls:
            func = getattr(tc, "function", None)
            if func is None:
                continue
            name = self._normalize_incoming_tool_name(getattr(func, "name", ""))
            raw_arguments = getattr(func, "arguments", {})
            args = self._parse_tool_arguments(raw_arguments)
            results.append({
                "tool": name,
                "params": args
            })
        return results

    def _create_backend(self, backend: str) -> _InferenceBackend:
        if backend == "responses":
            return _ResponsesCompatBackend()
        return _LiteLLMBackend()

    def _invoke_once_from_openai_messages(
        self,
        *,
        openai_messages: List[Dict[str, Any]],
        tools: Optional[list],
    ) -> LiteLLMResponse:
        call_kwargs = self._build_call_kwargs(
            openai_messages=openai_messages,
            tools=tools,
            stream=False,
        )
        response = self._completion_with_retry(call_kwargs)
        message = response.choices[0].message
        raw_content = getattr(message, "content", "")
        content = self._coerce_content_to_text(raw_content)
        reasoning_content = getattr(message, "reasoning_content", "") or ""

        resp = LiteLLMResponse(content=content, reasoning_content=reasoning_content)
        if raw_content != content:
            resp.additional_kwargs["raw_content"] = raw_content

        if self.is_multimodal:
            extracted_outputs = self._collect_multimodal_outputs(raw_content)
            if extracted_outputs:
                resp.additional_kwargs["multimodal_outputs"] = extracted_outputs

        raw_tool_calls = getattr(message, "tool_calls", None) or []
        if raw_tool_calls:
            resp.tool_calls = self._parse_native_tool_calls(raw_tool_calls)

        usage = normalize_usage_payload(getattr(response, "usage", None) or response)
        if usage["total_tokens"] > 0:
            resp.usage = usage
        return resp

    def _response_to_stream_chunks(
        self,
        response: LiteLLMResponse,
        *,
        stream_recovery_triggered: bool,
        stream_mode: str,
    ):
        if response.usage:
            yield {"usage": dict(response.usage)}
        if response.content or response.reasoning_content:
            payload = {
                "content": str(response.content or ""),
                "reasoning_content": str(response.reasoning_content or ""),
                "stream_mode": stream_mode,
                "stream_tool_recovery_triggered": bool(stream_recovery_triggered),
            }
            if "raw_content" in (response.additional_kwargs or {}):
                payload["raw_content"] = response.additional_kwargs.get("raw_content")
            yield payload
        if response.tool_calls:
            yield {
                "tool_calls": list(response.tool_calls),
                "stream_mode": stream_mode,
                "stream_tool_recovery_triggered": bool(stream_recovery_triggered),
            }

    @staticmethod
    def _infer_provider_name(model_name: str) -> str:
        route = str(model_name or "").strip().lower()
        if "/" in route:
            return route.split("/", 1)[0]
        return route

    def _provider_requires_non_empty_messages(self) -> bool:
        return self.provider_name in self._STRICT_EMPTY_MESSAGE_PROVIDERS

    @staticmethod
    def _is_empty_message_content(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, list):
            return len(value) == 0
        if isinstance(value, dict):
            return len(value) == 0
        return False

    @staticmethod
    def _has_tool_role_messages(messages: List[Dict[str, Any]]) -> bool:
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "") or "").strip().lower()
            if role == "tool":
                return True
        return False

    def _compat_noop_tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self._COMPAT_NOOP_TOOL_NAME,
                "description": "Compatibility noop for strict proxy payload validation. Do not call.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        }

    @staticmethod
    def _extract_tool_name_map(tools: Optional[list]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for item in tools or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("type", "") or "").strip() != "function":
                continue
            function_block = item.get("function", {}) or {}
            if not isinstance(function_block, dict):
                continue
            name = str(function_block.get("name", "") or "").strip()
            if not name:
                continue
            mapping[name.lower()] = name
        return mapping

    def _normalize_incoming_tool_name(self, value: Any) -> str:
        raw_name = str(value or "").strip()
        if not raw_name:
            return raw_name
        canonical = self._active_tool_name_map.get(raw_name.lower(), raw_name)
        if canonical != raw_name:
            self._tool_name_repair_count += 1
        return canonical

    @staticmethod
    def _parse_tool_arguments(raw_arguments: Any) -> Dict[str, Any]:
        if raw_arguments is None:
            return {}
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if not isinstance(raw_arguments, str):
            try:
                return dict(raw_arguments)  # type: ignore[arg-type]
            except Exception:
                return {"raw_arguments": raw_arguments}
        text = raw_arguments.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except json.JSONDecodeError:
            return {"raw_arguments": raw_arguments}

    @staticmethod
    def _get_obj_value(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    def _merge_stream_tool_call_delta(
        self,
        tool_call_accumulators: Dict[int, Dict[str, Any]],
        tc_delta: Any,
    ) -> None:
        raw_index = self._get_obj_value(tc_delta, "index", 0)
        try:
            index = int(raw_index)
        except Exception:
            index = 0

        acc = tool_call_accumulators.setdefault(
            index,
            {"id": "", "function": {"name": "", "arguments": ""}, "malformed": False},
        )
        tc_id = self._get_obj_value(tc_delta, "id", "")
        if tc_id:
            acc["id"] = str(tc_id)

        function_block = self._get_obj_value(tc_delta, "function", None)
        if function_block is None:
            return
        name_delta = self._get_obj_value(function_block, "name", "")
        args_delta = self._get_obj_value(function_block, "arguments", "")
        if name_delta:
            acc["function"]["name"] += str(name_delta)
        if args_delta:
            acc["function"]["arguments"] += str(args_delta)

    def _finalize_stream_tool_calls(
        self,
        tool_call_accumulators: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        assembled: List[Dict[str, Any]] = []
        for idx in sorted(tool_call_accumulators.keys()):
            acc = tool_call_accumulators[idx]
            function_block = dict(acc.get("function", {}) or {})
            if not function_block:
                continue
            name = str(function_block.get("name", "") or "").strip()
            raw_args = function_block.get("arguments", "")
            if not name and not raw_args:
                continue
            assembled.append(
                {
                    "tool": self._normalize_incoming_tool_name(name),
                    "params": self._parse_tool_arguments(raw_args),
                }
            )
        return assembled

    def _prepare_messages_and_tools_for_invoke(
        self,
        *,
        openai_messages: List[Dict[str, Any]],
        tools: Optional[list],
    ) -> Tuple[List[Dict[str, Any]], Optional[list], Dict[str, Any]]:
        normalized_messages: List[Dict[str, Any]] = []
        dropped_empty_messages = 0

        for message in openai_messages:
            if not isinstance(message, dict):
                continue
            normalized = dict(message)
            role = str(normalized.get("role", "user") or "user").strip().lower()
            if not role:
                role = "user"
            normalized["role"] = role
            if self._provider_requires_non_empty_messages() and self._is_empty_message_content(
                normalized.get("content")
            ):
                dropped_empty_messages += 1
                continue
            normalized_messages.append(normalized)

        if not normalized_messages and openai_messages:
            normalized_messages = [dict(openai_messages[-1])]

        normalized_tools: Optional[list] = (
            self._sanitize_jsonable(tools) if isinstance(tools, list) else tools
        )

        noop_tool_injected = False
        if (
            self.provider_name in self._TOOL_HISTORY_COMPAT_PROVIDERS
            and self._has_tool_role_messages(normalized_messages)
            and not normalized_tools
        ):
            normalized_tools = [self._compat_noop_tool_schema()]
            noop_tool_injected = True
            self._noop_tool_injected_count += 1

        self._active_tool_name_map = self._extract_tool_name_map(normalized_tools)
        prepare_meta = {
            "provider": self.provider_name,
            "dropped_empty_messages": dropped_empty_messages,
            "noop_tool_injected": noop_tool_injected,
        }
        return normalized_messages, normalized_tools, prepare_meta

    def _heuristic_supports_sampling_controls(self) -> bool:
        model_name = str(self.model or "").lower()
        return not any(marker in model_name for marker in ("reasoner", "o1", "kimi-k2.5"))

    def _supports_sampling_controls(self) -> bool:
        return bool(self.supports_temperature or self.supports_top_p)

    def _can_send_openai_param(self, param_name: str) -> bool:
        normalized = str(param_name or "").strip()
        if not normalized:
            return False
        if not self.allowed_openai_params:
            return True
        return normalized in self.allowed_openai_params

    def _build_call_kwargs(
        self,
        *,
        openai_messages: List[Dict[str, Any]],
        tools: Optional[list] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        prepared_messages, prepared_tools, prepare_meta = self._prepare_messages_and_tools_for_invoke(
            openai_messages=openai_messages,
            tools=tools,
        )
        self._last_prepare_meta = dict(prepare_meta)

        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": prepared_messages,
            "api_key": self.api_key,
        }
        if stream:
            call_kwargs["stream"] = True
            if self.supports_usage_in_stream and self._can_send_openai_param("stream_options"):
                call_kwargs["stream_options"] = {"include_usage": True}

        if self.api_base:
            call_kwargs["api_base"] = self.api_base

        if self.supports_temperature:
            call_kwargs["temperature"] = self.temperature
        if self.supports_top_p and self.top_p is not None:
            call_kwargs["top_p"] = self.top_p

        if prepared_tools and self._can_send_native_tools():
            if self._can_send_openai_param("tools"):
                call_kwargs["tools"] = prepared_tools
            if self._can_send_openai_param("tool_choice"):
                call_kwargs["tool_choice"] = "auto"
            if self.supports_parallel_tool_calls and self._can_send_openai_param("parallel_tool_calls"):
                call_kwargs["parallel_tool_calls"] = True

            allowed_params = list(self.allowed_openai_params or [])
            if self._is_xiaomi_mimo_model():
                allowed_params.extend(["tools", "tool_choice"])
            self._append_allowed_openai_params(call_kwargs, allowed_params)

        call_kwargs.update(self.extra_kwargs)
        self._drop_runtime_only_call_kwargs(call_kwargs)

        # Keep capability matrix authoritative even when users pass overrides via kwargs.
        if not self.supports_temperature:
            call_kwargs.pop("temperature", None)
        if not self.supports_top_p:
            call_kwargs.pop("top_p", None)
        if not self.supports_parallel_tool_calls:
            call_kwargs.pop("parallel_tool_calls", None)
        if not self.supports_usage_in_stream:
            call_kwargs.pop("stream_options", None)

        call_kwargs = self._sanitize_jsonable(call_kwargs)
        self._log_call_summary(call_kwargs, stream=stream)
        return call_kwargs

    def _can_send_native_tools(self) -> bool:
        explicit_support = getattr(self, "supports_native_tool_calling", None)
        if explicit_support is not None and not bool(explicit_support):
            return False
        return True

    def _is_xiaomi_mimo_model(self) -> bool:
        model_name = str(self.model or "").strip().lower()
        if model_name.startswith("xiaomi_mimo/"):
            return True
        if model_name.startswith("mimo-v2-"):
            return True
        return False

    def _append_allowed_openai_params(self, call_kwargs: Dict[str, Any], params: list) -> None:
        if not isinstance(params, (list, tuple, set)):
            return
        existing = call_kwargs.get("allowed_openai_params")
        merged: list = []
        if isinstance(existing, (list, tuple)):
            merged.extend([str(item).strip() for item in existing if str(item or "").strip()])
        elif isinstance(existing, str) and existing.strip():
            merged.append(existing.strip())
        for item in params:
            normalized = str(item or "").strip()
            if normalized and normalized not in merged:
                merged.append(normalized)
        if merged:
            call_kwargs["allowed_openai_params"] = merged

    def _drop_runtime_only_call_kwargs(self, call_kwargs: Dict[str, Any]) -> None:
        for key in self._RUNTIME_ONLY_KWARGS:
            call_kwargs.pop(key, None)

    def _log_call_summary(self, call_kwargs: Dict[str, Any], *, stream: bool) -> None:
        summary = {
            "model": str(call_kwargs.get("model", "")),
            "stream": bool(stream),
            "has_tools": "tools" in call_kwargs,
            "tool_count": len(call_kwargs.get("tools", []) or []) if isinstance(call_kwargs.get("tools"), list) else 0,
            "parallel_tool_calls": bool(call_kwargs.get("parallel_tool_calls", False)),
            "supports_parallel_tool_calls": bool(self.supports_parallel_tool_calls),
            "supports_streaming_tool_calls": bool(self.supports_streaming_tool_calls),
            "supports_usage_in_stream": bool(self.supports_usage_in_stream),
            "tool_name_repair_count": int(self._tool_name_repair_count),
            "noop_tool_injected_count": int(self._noop_tool_injected_count),
            "stream_tool_recovery_triggered_count": int(self._stream_tool_recovery_triggered_count),
            "prepare_meta": dict(self._last_prepare_meta or {}),
            "temperature": call_kwargs.get("temperature", None),
            "top_p": call_kwargs.get("top_p", None),
            "api_base": call_kwargs.get("api_base", self.api_base),
            "keys": sorted(list(call_kwargs.keys())),
        }
        logger.debug("Model call summary: %s", summary)
        if self.log_call_params:
            try:
                print(f"[Model Call] {json.dumps(summary, ensure_ascii=False)}")
            except Exception:
                print(f"[Model Call] {summary}")

    def _log_model_call_failure(
        self,
        exc: Exception,
        call_kwargs: Dict[str, Any],
        *,
        stream: bool,
    ) -> None:
        summary = {
            "model": str(call_kwargs.get("model", "")),
            "stream": bool(stream),
            "has_tools": "tools" in call_kwargs,
            "tool_choice": call_kwargs.get("tool_choice"),
            "temperature": call_kwargs.get("temperature", None),
            "top_p": call_kwargs.get("top_p", None),
            "api_base": call_kwargs.get("api_base", self.api_base),
            "keys": sorted(list(call_kwargs.keys())),
        }
        logger.error(
            "Model call failed (%s): %s | summary=%s",
            type(exc).__name__,
            exc,
            summary,
        )
        if self.log_call_params:
            try:
                print(
                    "[Model Call Error] "
                    + json.dumps(
                        {
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "summary": summary,
                        },
                        ensure_ascii=False,
                    )
                )
            except Exception:
                print(f"[Model Call Error] {type(exc).__name__}: {exc} | {summary}")

    @staticmethod
    def _error_contains_param_rejection(error_text: str, param_name: str) -> bool:
        """Detect provider-side request validation errors for a specific OpenAI-style param."""
        text = str(error_text or "").lower()
        if not text:
            return False
        normalized = str(param_name or "").strip().lower()
        if not normalized:
            return False

        aliases = {normalized, normalized.replace("_", " ")}
        if normalized.endswith("s"):
            aliases.add(normalized[:-1])
            aliases.add(normalized[:-1].replace("_", " "))
        if not any(alias and alias in text for alias in aliases):
            return False

        rejection_markers = (
            "unsupported",
            "not support",
            "unexpected",
            "unknown",
            "unrecognized",
            "invalid",
            "not allowed",
            "forbidden",
            "extra_forbidden",
            "extra fields not permitted",
            "additional properties are not allowed",
        )
        return any(marker in text for marker in rejection_markers)

    def _apply_call_kwargs_compat_fallback(
        self,
        exc: Exception,
        call_kwargs: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Apply one-shot compatibility downgrades when providers reject certain request params.

        Returns:
            (changed, removed_keys)
        """
        if not isinstance(call_kwargs, dict):
            return False, []

        error_text = f"{type(exc).__name__}: {exc}"
        removed: List[str] = []

        if "parallel_tool_calls" in call_kwargs and self._error_contains_param_rejection(
            error_text, "parallel_tool_calls"
        ):
            call_kwargs.pop("parallel_tool_calls", None)
            removed.append("parallel_tool_calls")
            # Keep subsequent calls aligned with runtime-observed provider behavior.
            self.supports_parallel_tool_calls = False
            self.model_capabilities["supports_parallel_tool_calls"] = False

        if "tool_choice" in call_kwargs and self._error_contains_param_rejection(
            error_text, "tool_choice"
        ):
            call_kwargs.pop("tool_choice", None)
            removed.append("tool_choice")

        return bool(removed), removed

    @staticmethod
    def _normalize_model_capabilities(value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        normalized: Dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key or "").strip()
            if not text_key:
                continue
            normalized[text_key] = item
        return normalized

    def _resolve_capability_bool(self, *, key: str, explicit_value: Any, default: bool) -> bool:
        if explicit_value is not self._UNSET:
            return self._parse_bool_option(explicit_value, default=default)
        if key in self.model_capabilities:
            return self._parse_bool_option(self.model_capabilities.get(key), default=default)
        return default

    @staticmethod
    def _normalize_openai_params(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            parts = [item.strip() for item in value.split(",")]
            return [item for item in parts if item]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item or "").strip()]
        return []
    
    def _parse_single_tool_call(self, acc: dict) -> dict:
        """
         tool_call 
        """
        name = self._normalize_incoming_tool_name(acc["function"]["name"])
        raw_args = acc["function"]["arguments"]
        args = self._parse_tool_arguments(raw_args)
        return {
            "tool": name,
            "params": args
        }
    
    def _convert_messages(self, messages):
        """ LangChain  OpenAI """
        openai_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                openai_messages.append(dict(msg))
            elif isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})
            else:
                role = getattr(msg, 'type', 'user')
                openai_messages.append({"role": role, "content": str(msg.content)})
        if self.is_multimodal:
            return [self._normalize_multimodal_message(message) for message in openai_messages]
        return openai_messages

    def _normalize_multimodal_message(self, message: dict) -> dict:
        if not isinstance(message, dict):
            return message

        normalized = dict(message)
        helper_images = normalized.pop("images", None)
        if helper_images is None:
            helper_images = normalized.pop("image_paths", None)

        content = normalized.get("content")
        if isinstance(content, list):
            normalized["content"] = [self._normalize_multimodal_part(part) for part in content]
        elif isinstance(content, dict):
            normalized["content"] = [self._normalize_multimodal_part(content)]
        elif isinstance(content, str):
            if helper_images:
                parts = []
                if content:
                    parts.append({"type": "text", "text": content})
                parts.extend(self._build_multimodal_image_parts(helper_images))
                normalized["content"] = parts
        elif content is None and helper_images:
            normalized["content"] = self._build_multimodal_image_parts(helper_images)
        return normalized

    def _build_multimodal_image_parts(self, images: Any) -> list:
        if images is None:
            return []
        if isinstance(images, (list, tuple)):
            values = images
        else:
            values = [images]

        parts = []
        for item in values:
            if isinstance(item, str):
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._maybe_encode_path_to_data_url(item)},
                    }
                )
                continue
            if isinstance(item, dict):
                parts.append(self._normalize_multimodal_part(item))
                continue
            parts.append(item)
        return parts

    def _normalize_multimodal_part(self, part: Any) -> Any:
        if isinstance(part, str):
            return {"type": "text", "text": part}
        if not isinstance(part, dict):
            return part

        item = dict(part)
        part_type = str(item.get("type") or "").strip().lower()

        if part_type in {"text", "input_text"}:
            text_value = item.get("text", item.get("content", ""))
            return {"type": "text", "text": str(text_value)}

        if part_type in {"image_url", "input_image", "image", "image_path", "local_image"}:
            return self._normalize_image_payload(item)

        if any(key in item for key in ("image_url", "image_path", "image_base64", "file_path")):
            return self._normalize_image_payload(item)

        return item

    def _normalize_image_payload(self, payload: dict) -> dict:
        image_url = payload.get("image_url")
        if image_url is None and "url" in payload:
            image_url = payload.get("url")
        if image_url is None:
            image_url = payload.get("image_path", payload.get("file_path"))

        if isinstance(image_url, str):
            return {
                "type": "image_url",
                "image_url": {"url": self._maybe_encode_path_to_data_url(image_url)},
            }

        if isinstance(image_url, dict):
            normalized_image_url = dict(image_url)
            if isinstance(normalized_image_url.get("url"), str):
                normalized_image_url["url"] = self._maybe_encode_path_to_data_url(
                    normalized_image_url["url"]
                )
            elif isinstance(normalized_image_url.get("uri"), str):
                normalized_image_url["url"] = self._maybe_encode_path_to_data_url(
                    normalized_image_url["uri"]
                )
                normalized_image_url.pop("uri", None)
            elif isinstance(normalized_image_url.get("b64_json"), str):
                mime = str(normalized_image_url.get("mime_type") or "image/png")
                normalized_image_url["url"] = (
                    f"data:{mime};base64,{normalized_image_url['b64_json']}"
                )
            return {"type": "image_url", "image_url": normalized_image_url}

        image_base64 = payload.get("image_base64")
        if isinstance(image_base64, str):
            mime = str(payload.get("mime_type") or "image/png")
            return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_base64}"}}

        return payload

    def _maybe_encode_path_to_data_url(self, url_or_path: str) -> str:
        value = str(url_or_path or "").strip()
        if not value:
            return value
        if not self.multimodal_auto_base64:
            return value
        if value.lower().startswith("data:"):
            return value

        if self._URI_SCHEME_PATTERN.match(value) and not value.lower().startswith("file://"):
            return value

        local_path = value[7:] if value.lower().startswith("file://") else value
        path_obj = Path(local_path).expanduser()
        if not path_obj.is_file():
            return value

        mime_type = mimetypes.guess_type(str(path_obj))[0] or "application/octet-stream"
        file_bytes = path_obj.read_bytes()
        encoded = base64.b64encode(file_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _coerce_content_to_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    if item:
                        text_parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                part_type = str(item.get("type") or "").strip().lower()
                if part_type in {"text", "input_text", "output_text"}:
                    text = item.get("text", item.get("content", ""))
                    if text:
                        text_parts.append(str(text))
            return "\n".join(text_parts)
        return str(content)

    def _collect_multimodal_outputs(self, raw_content: Any) -> list:
        if not self.multimodal_extract_output:
            return []

        data_urls = []
        self._walk_for_data_urls(raw_content, data_urls)
        if not data_urls:
            return []

        outputs = []
        for idx, (mime_type, base64_payload) in enumerate(data_urls, start=1):
            try:
                decoded = base64.b64decode(base64_payload, validate=True)
            except Exception:
                continue

            item = {
                "mime_type": mime_type,
                "size_bytes": len(decoded),
            }
            if self.multimodal_include_base64_output:
                item["base64"] = base64_payload

            saved_path = self._write_multimodal_output(decoded, mime_type, index=idx)
            if saved_path:
                item["path"] = saved_path
            outputs.append(item)
        return outputs

    def _walk_for_data_urls(self, value: Any, sink: list) -> None:
        if isinstance(value, str):
            parsed = self._parse_data_url(value)
            if parsed is not None:
                sink.append(parsed)
            return
        if isinstance(value, list):
            for item in value:
                self._walk_for_data_urls(item, sink)
            return
        if isinstance(value, dict):
            for item in value.values():
                self._walk_for_data_urls(item, sink)

    @classmethod
    def _parse_data_url(cls, maybe_data_url: str):
        text = str(maybe_data_url or "").strip()
        if not text:
            return None
        match = cls._DATA_URL_PATTERN.match(text)
        if not match:
            return None
        mime_type = match.group("mime") or "application/octet-stream"
        payload = re.sub(r"\s+", "", match.group("data") or "")
        if not payload:
            return None
        return mime_type, payload

    def _write_multimodal_output(self, decoded: bytes, mime_type: str, index: int) -> Optional[str]:
        if not self.multimodal_output_dir:
            return None

        output_dir = Path(self.multimodal_output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        guessed_ext = mimetypes.guess_extension(mime_type) or ".bin"
        safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(self.model or "model"))
        filename = f"{safe_model}_{int(time.time() * 1000)}_{index}{guessed_ext}"
        output_path = output_dir / filename
        output_path.write_bytes(decoded)
        return str(output_path)

    def _sanitize_jsonable(self, value: Any) -> Any:
        """ JSON  Unicode ,."""
        if isinstance(value, str):
            #  read_tui  lone surrogate(\ud800-\udfff)
            # /JSON  UTF-8 , U+FFFD.
            return value.encode("utf-8", errors="replace").decode("utf-8")
        if isinstance(value, list):
            return [self._sanitize_jsonable(item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize_jsonable(item) for item in value]
        if isinstance(value, dict):
            return {
                self._sanitize_jsonable(key): self._sanitize_jsonable(item)
                for key, item in value.items()
            }
        return value

    def _completion_with_retry(self, call_kwargs: dict):
        model_candidates = self._get_model_candidates()
        for model_idx, model_name in enumerate(model_candidates):
            model_kwargs = dict(call_kwargs)
            model_kwargs["model"] = model_name
            for attempt in range(1, self.retry_max_attempts + 1):
                try:
                    return self.backend.completion(model_kwargs)
                except Exception as exc:
                    compat_changed, removed_keys = self._apply_call_kwargs_compat_fallback(exc, model_kwargs)
                    if compat_changed:
                        logger.warning(
                            "Model completion compatibility downgrade applied (model=%s, removed=%s), retrying: %s",
                            model_name,
                            ",".join(removed_keys),
                            exc,
                        )
                        continue

                    if (not self.retry_enabled) or (not self._is_retriable_error(exc)):
                        self._log_model_call_failure(exc, model_kwargs, stream=False)
                        raise

                    has_next_attempt = attempt < self.retry_max_attempts
                    has_next_model = model_idx < len(model_candidates) - 1
                    if not has_next_attempt and not has_next_model:
                        raise

                    if has_next_attempt:
                        delay = self._retry_delay_seconds(attempt)
                        logger.warning(
                            "Model completion failed, retrying in %.2fs (model=%s, attempt=%s/%s): %s",
                            delay,
                            model_name,
                            attempt,
                            self.retry_max_attempts,
                            exc,
                        )
                        time.sleep(delay)
                    else:
                        logger.warning(
                            "Model completion failed after %s attempts, switching to fallback model: %s -> %s; error=%s",
                            self.retry_max_attempts,
                            model_name,
                            model_candidates[model_idx + 1],
                            exc,
                        )
                    continue

        raise RuntimeError("Model completion failed unexpectedly without captured exception")

    def _stream_with_retry(self, call_kwargs: dict):
        model_candidates = self._get_model_candidates()
        for model_idx, model_name in enumerate(model_candidates):
            model_kwargs = dict(call_kwargs)
            model_kwargs["model"] = model_name
            for attempt in range(1, self.retry_max_attempts + 1):
                emitted_any = False
                try:
                    response = self.backend.stream_completion(model_kwargs)
                    for chunk in response:
                        emitted_any = True
                        yield chunk
                    return
                except Exception as exc:
                    # ,
                    if emitted_any:
                        self._log_model_call_failure(exc, model_kwargs, stream=True)
                        raise

                    compat_changed, removed_keys = self._apply_call_kwargs_compat_fallback(exc, model_kwargs)
                    if compat_changed:
                        logger.warning(
                            "Model stream compatibility downgrade applied (model=%s, removed=%s), retrying: %s",
                            model_name,
                            ",".join(removed_keys),
                            exc,
                        )
                        continue

                    if (not self.retry_enabled) or (not self._is_retriable_error(exc)):
                        self._log_model_call_failure(exc, model_kwargs, stream=True)
                        raise

                    has_next_attempt = attempt < self.retry_max_attempts
                    has_next_model = model_idx < len(model_candidates) - 1
                    if not has_next_attempt and not has_next_model:
                        raise

                    if has_next_attempt:
                        delay = self._retry_delay_seconds(attempt)
                        logger.warning(
                            "Model stream failed, retrying in %.2fs (model=%s, attempt=%s/%s): %s",
                            delay,
                            model_name,
                            attempt,
                            self.retry_max_attempts,
                            exc,
                        )
                        time.sleep(delay)
                    else:
                        logger.warning(
                            "Model stream failed after %s attempts, switching to fallback model: %s -> %s; error=%s",
                            self.retry_max_attempts,
                            model_name,
                            model_candidates[model_idx + 1],
                            exc,
                        )
                    continue

        raise RuntimeError("Model stream failed unexpectedly without captured exception")

    def _get_model_candidates(self) -> list:
        names = [str(self.model).strip()]
        for item in self.fallback_models:
            candidate = str(item or "").strip()
            if candidate and candidate not in names:
                names.append(candidate)
        return names

    def _retry_delay_seconds(self, attempt: int) -> float:
        # attempt  1 ,
        base_delay = self.retry_initial_delay * (self.retry_backoff_factor ** max(0, attempt - 1))
        jitter = random.uniform(0.0, self.retry_jitter) if self.retry_jitter > 0 else 0.0
        return min(self.retry_max_delay, base_delay + jitter)

    def _is_retriable_error(self, exc: Exception) -> bool:
        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError, TimeoutError, ConnectionError)):
            return True

        text = f"{type(exc).__name__}: {exc}".lower()
        non_retriable_markers = (
            "invalid api key",
            "incorrect api key",
            "authentication",
            "unauthorized",
            "forbidden",
            "permission",
            "invalid_request",
            "badrequest",
            "context_length_exceeded",
            "max context length",
            "unsupported",
            "not found",
            "does not exist",
        )
        if any(marker in text for marker in non_retriable_markers):
            return False

        retriable_markers = (
            "connection error",
            "connection aborted",
            "connection reset",
            "remote protocol error",
            "temporarily unavailable",
            "service unavailable",
            "timeout",
            "timed out",
            "eof occurred in violation of protocol",
            "ssl",
            "429",
            "502",
            "503",
            "504",
            "rate limit",
            "try again",
        )
        return any(marker in text for marker in retriable_markers)

    @staticmethod
    def _parse_bool_option(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return default

    @staticmethod
    def _parse_int_option(value: Any, default: int, minimum: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            return default
        return max(minimum, parsed)

    @staticmethod
    def _parse_float_option(value: Any, default: float, minimum: float) -> float:
        try:
            parsed = float(value)
        except Exception:
            return default
        return max(minimum, parsed)

    @staticmethod
    def _parse_optional_float_option(
        value: Any,
        default: Optional[float] = None,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> Optional[float]:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        try:
            parsed = float(text)
        except Exception:
            return default
        if minimum is not None:
            parsed = max(minimum, parsed)
        if maximum is not None:
            parsed = min(maximum, parsed)
        return parsed

    @staticmethod
    def _parse_fallback_models(value: Any) -> list:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item or "").strip()]
        text = str(value).strip()
        if not text:
            return []
        return [segment.strip() for segment in text.split(",") if segment.strip()]

    def __repr__(self):
        return f"LiteLLMChat(model={self.model!r}, max_context_tokens={self.max_context_tokens})"


# ,, LiteLLMChat
DeepSeekReasonerChat = LiteLLMChat
DeepSeekReasonerResponse = LiteLLMResponse
