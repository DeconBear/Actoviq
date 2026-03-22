"""
RequestOrchestrator 运行时。

本运行时与默认 ReAct 图循环解耦，采用“双脑协作”模式：
1. 小脑（fast model）负责短回路决策、工具调用与快速响应。
2. 大脑（thinking model）负责规划、子任务推进与最终收敛。
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from ...graph import GraphKlynxAgent
from ...state import AgentState
from ...subgraphs import stream_ask


def _load_prompt_asset(*relative_parts: str, fallback: str = "") -> str:
    """读取提示词资源文件，读取失败时返回回退文本。"""
    prompt_path = Path(__file__).resolve().parents[2].joinpath(*relative_parts)
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except Exception:
        return fallback.strip()


class RequestOrchestratorAgent(GraphKlynxAgent):
    """请求编排代理：大脑规划，小脑执行。"""

    REQUEST_THINK_TOOL_NAME = "request_think"
    REQUEST_THINK_TOOL_SCHEMA = {
        "type": "function",
        "function": {
            "name": REQUEST_THINK_TOOL_NAME,
            "description": (
                "Send a request to Think Brain for a complex task "
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "给 Think Brain 的主要请求信息。",
                    },
                    "reason": {
                        "type": "string",
                        "description": "升级原因（可选）。",
                    },
                    "focus": {
                        "type": "string",
                        "description": "希望 Think Brain 重点关注的方向（可选）。",
                    },
                    "information": {
                        "type": "string",
                        "description": "兼容字段：你希望 Think Brain 知道的有利于完成任务的信息。",
                    },
                    "capabilities": {
                        "type": "string",
                        "description": "兼容字段：能力描述，告知 Think Brain 可执行的操作范围。",
                    },
                    "task": {
                        "type": "string",
                        "description": "任务目标",
                    },
                    "include_recent_call_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "可选：填写需要带给 Think Brain 的历史工具调用 id 列表。"
                            "系统会按这些 id 提取对应工具结果并注入给 Think Brain 进行决策。"
                        ),
                    },
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        },
    }

    REQUEST_ACT_TOOL_NAME = "request_act"
    REQUEST_ACT_TOOL_SCHEMA = {
        "type": "function",
        "function": {
            "name": REQUEST_ACT_TOOL_NAME,
            "description": (
                "Only control tool for Think Brain. Use this tool every round to send instructions "
                "to Act Brain and runtime. Do not respond with plain JSON."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "act": {
                        "type": "object",
                        "description": (
                            "结构化执行动作。根据 Act Brain 的能力描述，用 JSON 说明要做什么以及参数。"
                            "例如读取文件、修改文件、查看目录等操作。"
                        ),
                    },
                    "task_complete": {
                        "type": "string",
                        "enum": ["是", "否"],
                        "description": "当前任务是否完成。",
                    },
                    "task_goal_update": {
                        "type": "string",
                        "description": "任务目标更新内容（可选）。",
                    },
                    "task_list": {
                        "type": "array",
                        "description": "将当前任务拆解后的细分任务列表（可选）。",
                        "items": {
                            "type": "object",
                            "description": "单个细分任务的信息对象。",
                        },
                    },
                    "message": {
                        "type": "string",
                        "description": (
                            "给 Act Brain 的消息，例如任务无法完成、需要询问用户、"
                            "或要求 Act Brain 提供更多信息。"
                        ),
                    },
                },
                "required": ["act", "task_complete", "message"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(
        self,
        *args,
        append_system_prompt: str = "",
        enable_subagent: bool = False,
        fast_model: Any = None,
        thinking_model: Any = None,
        small_self_loop_max: int = 3,
        context_warn_ratio: float = 0.80,
        context_hard_ratio: float = 0.92,
        **kwargs,
    ):
        """
        初始化 RequestOrchestrator 运行时参数。

        - fast_model/thinking_model: 小脑与大脑模型，可分别指定。
        - small_self_loop_max: 小脑最大自循环轮数。
        - context_warn_ratio/context_hard_ratio: 上下文压力阈值（告警/强压缩）。
        """
        # 启动时注入的系统提示词追加片段（全局有效）。
        self._init_system_prompt_append = (append_system_prompt or "").strip()
        # 预留运行时动态追加提示词位置。
        self._runtime_system_prompt_append = ""
        # 是否允许大脑请求并触发子代理（当前仅占位流程）。
        self.enable_subagent = bool(enable_subagent)
        self.fast_model = fast_model
        self.thinking_model = thinking_model
        # 小脑自循环轮数至少为 1。
        self.small_self_loop_max = max(1, int(small_self_loop_max or 1))
        self.context_warn_ratio = float(context_warn_ratio or 0.80)
        self.context_hard_ratio = float(context_hard_ratio or 0.92)
        # 线程上下文存储锁与上下文字典。
        self._request_lock = threading.Lock()
        self._request_threads: Dict[str, Dict[str, Any]] = {}
        super().__init__(*args, **kwargs)
        # 若未显式传入双模型，则退化复用基类 model。
        self.fast_model = self.fast_model or self.model
        self.thinking_model = self.thinking_model or self.model

    def _build_graph(self) -> StateGraph:
        # 该运行时由 invoke 驱动而非图循环驱动，这里仅保持一个最小可编译图。
        workflow = StateGraph(AgentState)
        workflow.add_node("noop", lambda state: {})
        workflow.set_entry_point("noop")
        workflow.add_edge("noop", END)
        return workflow

    def ask(self, message: str, system_prompt: str = None, thread_id: str = "default"):
        """问答直通接口：复用 ask 子图实现单轮/流式问答。"""
        return stream_ask(self, message, system_prompt=system_prompt, thread_id=thread_id)

    def _extract_response_text(self, response: Any) -> str:
        """
        从模型响应对象中提取文本。

        兼容：
        - 纯字符串；
        - LangChain 的多段 content 列表（dict 或文本片段）。
        """
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    # 常见片段格式：{"text": "..."}
                    text = str(item.get("text", "") or "").strip()
                    if text:
                        chunks.append(text)
                else:
                    text = str(item or "").strip()
                    if text:
                        chunks.append(text)
            return "\n".join(chunks).strip()
        return str(content or "").strip()

    def _parse_json_object(self, raw_text: str) -> Dict[str, Any]:
        """
        从模型文本中尽力提取 JSON 对象。

        解析策略：
        1) 直接 json.loads；
        2) 解析 ```json ... ``` 代码块；
        3) 截取首尾大括号片段重试。
        """
        text = str(raw_text or "").strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, re.IGNORECASE)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            snippet = text[start : end + 1]
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return {}

    def _invoke_model_dict(
        self,
        *,
        model: Any,
        system_prompt: str,
        payload: Dict[str, Any],
        fallback: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        统一模型调用入口：输入 payload，输出“字典结果”。

        若模型不可用、调用失败或解析失败，则返回 fallback。
        """
        if model is None or not hasattr(model, "invoke"):
            return dict(fallback)
        # 统一将 payload 转成可读 JSON 文本传给模型。
        user_text = json.dumps(payload, ensure_ascii=False, indent=2)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_text)]
        try:
            response = model.invoke(messages)
        except Exception:
            return dict(fallback)
        parsed = self._parse_json_object(self._extract_response_text(response))
        if not parsed:
            return dict(fallback)
        # 在 fallback 基础上做“覆盖式合并”，保证关键字段始终存在。
        merged = dict(fallback)
        merged.update(parsed)
        return merged

    def _build_small_brain_native_tools(self) -> List[Dict[str, Any]]:
        """
        构建小脑可见的 Native Tool Calling schema 列表。

        包含：
        - 当前激活工具 schema（来自运行时 _json_schemas）
        - request_think（小脑向大脑发起升级请求的内部工具）
        """
        schemas: List[Dict[str, Any]] = []
        existing_names: set[str] = set()
        for row in list(getattr(self, "_json_schemas", []) or []):
            if not isinstance(row, dict):
                continue
            if str(row.get("type", "") or "").strip() != "function":
                continue
            fn = row.get("function", {}) or {}
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name", "") or "").strip()
            if not name:
                continue
            existing_names.add(name)
            schemas.append(dict(row))
        if self.REQUEST_THINK_TOOL_NAME not in existing_names:
            schemas.append(dict(self.REQUEST_THINK_TOOL_SCHEMA))
        return schemas

    def _normalize_request_think_payload(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """规范化 request_think 参数，避免模型传入异常结构。"""
        raw = dict(params or {})
        message = str(raw.get("message", "") or "").strip()
        if not message:
            message = str(raw.get("information", "") or "").strip()
        reason = str(raw.get("reason", "") or "").strip()
        focus = str(raw.get("focus", "") or "").strip()
        if not focus:
            focus = str(raw.get("capabilities", "") or "").strip()
        task = str(raw.get("task", "") or "").strip()
        include_recent_call_ids_raw = raw.get("include_recent_call_ids", [])
        include_recent_call_ids: List[str] = []
        if isinstance(include_recent_call_ids_raw, list):
            for item in include_recent_call_ids_raw:
                text = str(item or "").strip()
                if text and text not in include_recent_call_ids:
                    include_recent_call_ids.append(text)
        return {
            "message": message,
            "reason": reason,
            "focus": focus,
            "task": task,
            "include_recent_call_ids": include_recent_call_ids,
        }

    def _invoke_small_brain_with_tools(
        self,
        *,
        payload: Dict[str, Any],
        system_prompt: str,
    ) -> Dict[str, Any]:
        """
        使用 Native Tool Calling 调用小脑。

        返回统一结构：
        {
          "direct_answer": str,
          "tool_plan": list[tool call],
          "request_think": dict|None,
          "decision": str,
          "reason_summary": str
        }
        """
        fallback = {
            "direct_answer": "",
            "tool_plan": [],
            "request_think": {
                "message": "Need Think Brain to continue due insufficient confidence in short loop.",
                "reason": "small_brain_fallback",
                "focus": "",
                "include_recent_call_ids": [],
            },
            "decision": "escalate",
            "reason_summary": "fallback",
        }
        model = self.fast_model
        if model is None or not hasattr(model, "invoke"):
            return dict(fallback)

        tools = self._build_small_brain_native_tools()
        user_text = json.dumps(payload, ensure_ascii=False, indent=2)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_text)]

        try:
            response = model.invoke(messages, tools=tools)
        except TypeError:
            # 兼容不支持 tools 参数的模型实现，退化到普通调用。
            try:
                response = model.invoke(messages)
            except Exception:
                return dict(fallback)
        except Exception:
            return dict(fallback)

        content_text = self._extract_response_text(response).strip()
        raw_tool_calls = list(getattr(response, "tool_calls", []) or [])
        tool_plan: List[Dict[str, Any]] = []
        request_think: Optional[Dict[str, Any]] = None

        for row in raw_tool_calls:
            if not isinstance(row, dict):
                continue
            raw_tool_name = str(row.get("tool", "") or "").strip()
            params = row.get("params", {})
            if not isinstance(params, dict):
                params = {}

            if raw_tool_name == self.REQUEST_THINK_TOOL_NAME:
                request_think = self._normalize_request_think_payload(params)
                continue

            tool_name = self._normalize_tool_name(raw_tool_name)
            if not tool_name:
                continue
            tool_plan.append(
                {
                    "call_id": f"call_{uuid.uuid4().hex[:8]}",
                    "tool": tool_name,
                    "args": dict(params),
                    "parallel_group": "",
                }
            )

        if request_think:
            return {
                "direct_answer": "",
                "tool_plan": [],
                "request_think": request_think,
                "decision": "escalate_via_request_tool",
                "reason_summary": str(request_think.get("reason", "") or ""),
            }
        if tool_plan:
            return {
                "direct_answer": "",
                "tool_plan": tool_plan,
                "request_think": None,
                "decision": "tool_calling",
                "reason_summary": "",
            }
        if content_text:
            return {
                "direct_answer": content_text,
                "tool_plan": [],
                "request_think": None,
                "decision": "direct_answer",
                "reason_summary": "",
            }
        return dict(fallback)

    def _invoke_small_brain_with_tools_stream(
        self,
        *,
        payload: Dict[str, Any],
        system_prompt: str,
    ):
        """
        小脑流式调用入口。

        行为：
        - 优先走 model.stream，并在直答场景实时产出 token 事件；
        - 若模型/适配器不支持 stream，自动回退到 _invoke_small_brain_with_tools；
        - 生成器 return 值为与 _invoke_small_brain_with_tools 一致的 reply dict。
        """
        fallback = {
            "direct_answer": "",
            "tool_plan": [],
            "request_think": {
                "message": "Need Think Brain to continue due insufficient confidence in short loop.",
                "reason": "small_brain_fallback",
                "focus": "",
                "include_recent_call_ids": [],
            },
            "decision": "escalate",
            "reason_summary": "fallback",
        }
        model = self.fast_model
        if model is None:
            return dict(fallback)
        if not hasattr(model, "stream"):
            return self._invoke_small_brain_with_tools(payload=payload, system_prompt=system_prompt)

        tools = self._build_small_brain_native_tools()
        user_text = json.dumps(payload, ensure_ascii=False, indent=2)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_text)]

        try:
            stream = model.stream(messages, tools=tools)
        except TypeError:
            return self._invoke_small_brain_with_tools(payload=payload, system_prompt=system_prompt)
        except Exception:
            return self._invoke_small_brain_with_tools(payload=payload, system_prompt=system_prompt)

        content_parts: List[str] = []
        raw_tool_calls: List[Dict[str, Any]] = []
        streamed_direct_answer = False
        saw_tool_calls = False

        try:
            for chunk in stream:
                if isinstance(chunk, dict):
                    chunk_tool_calls = chunk.get("tool_calls", [])
                    content_delta = str(chunk.get("content", "") or "")
                else:
                    chunk_tool_calls = list(getattr(chunk, "tool_calls", []) or [])
                    content_delta = str(getattr(chunk, "content", "") or "")
                    if not content_delta:
                        content_delta = self._extract_response_text(chunk)
                if isinstance(chunk_tool_calls, list) and chunk_tool_calls:
                    raw_tool_calls = [dict(row) for row in chunk_tool_calls if isinstance(row, dict)]
                    saw_tool_calls = True
                if content_delta:
                    content_parts.append(content_delta)
                    # 仅在未观测到工具调用时向外发 token，避免工具模式下输出噪声文本。
                    if not saw_tool_calls:
                        streamed_direct_answer = True
                        yield {"type": "token", "content": content_delta, "brain": "ACT"}
        except Exception:
            return self._invoke_small_brain_with_tools(payload=payload, system_prompt=system_prompt)

        content_text = "".join(content_parts).strip()
        tool_plan: List[Dict[str, Any]] = []
        request_think: Optional[Dict[str, Any]] = None

        for row in raw_tool_calls:
            raw_tool_name = str(row.get("tool", "") or "").strip()
            params = row.get("params", {})
            if not isinstance(params, dict):
                params = {}

            if raw_tool_name == self.REQUEST_THINK_TOOL_NAME:
                request_think = self._normalize_request_think_payload(params)
                continue

            tool_name = self._normalize_tool_name(raw_tool_name)
            if not tool_name:
                continue
            tool_plan.append(
                {
                    "call_id": f"call_{uuid.uuid4().hex[:8]}",
                    "tool": tool_name,
                    "args": dict(params),
                    "parallel_group": "",
                }
            )

        if request_think:
            return {
                "direct_answer": "",
                "tool_plan": [],
                "request_think": request_think,
                "decision": "escalate_via_request_tool",
                "reason_summary": str(request_think.get("reason", "") or ""),
            }
        if tool_plan:
            return {
                "direct_answer": "",
                "tool_plan": tool_plan,
                "request_think": None,
                "decision": "tool_calling",
                "reason_summary": "",
            }
        if content_text:
            return {
                "direct_answer": content_text,
                "tool_plan": [],
                "request_think": None,
                "decision": "direct_answer",
                "reason_summary": "",
                "_direct_answer_streamed": bool(streamed_direct_answer),
            }
        return dict(fallback)

    def _build_big_brain_native_tools(self) -> List[Dict[str, Any]]:
        """构建大脑可见工具：仅暴露 request_act 单一控制工具。"""
        return [dict(self.REQUEST_ACT_TOOL_SCHEMA)]

    def _normalize_request_act_result(
        self,
        *,
        params: Dict[str, Any],
        request_type: str,
        fallback: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        """
        规范化 request_act 参数并做有效性校验。

        request_act 仅接受以下 5 个字段：
        - act
        - task_complete
        - task_goal_update（可选）
        - task_list（可选）
        - message

        返回 (normalized, error_message)。error_message 为空表示通过校验。
        """
        merged = dict(fallback or {})
        if not isinstance(params, dict):
            return merged, "request_act params must be an object."
        allowed_keys = {"act", "task_complete", "task_goal_update", "task_list", "message"}
        unknown_keys = [key for key in params.keys() if key not in allowed_keys]
        if unknown_keys:
            return merged, f"request_act contains unsupported fields: {', '.join(sorted(unknown_keys))}."

        if "act" not in params:
            return merged, "request_act requires act."
        act = params.get("act", {})
        if not isinstance(act, dict):
            return merged, "request_act.act must be an object."

        if "task_complete" not in params:
            return merged, "request_act requires task_complete."
        task_complete_raw = params.get("task_complete")
        task_complete: Optional[bool] = None
        if isinstance(task_complete_raw, bool):
            task_complete = bool(task_complete_raw)
        else:
            text = str(task_complete_raw or "").strip().lower()
            if text in {"是", "yes", "true", "1", "done", "completed"}:
                task_complete = True
            elif text in {"否", "no", "false", "0", "continue", "ongoing"}:
                task_complete = False
        if task_complete is None:
            return merged, "request_act.task_complete must be '是' or '否'."

        message = str(params.get("message", "") or "").strip()
        if not message:
            return merged, "request_act.message must be non-empty."

        task_goal_update = str(params.get("task_goal_update", "") or "").strip()
        task_list_raw = params.get("task_list", [])
        if task_list_raw is None:
            task_list_raw = []
        if not isinstance(task_list_raw, list):
            return merged, "request_act.task_list must be an array when provided."
        task_list = [dict(row) for row in task_list_raw if isinstance(row, dict)]

        merged["act"] = dict(act)
        merged["task_complete"] = task_complete
        merged["task_goal_update"] = task_goal_update
        merged["task_list"] = task_list
        merged["message"] = message

        # 兼容运行时内部结构：从 act 中提取可执行 tool_plan，并将 task_list 映射为 subtasks。
        action_rows: Any = []
        for key in ("operations", "tool_plan", "actions", "steps"):
            value = act.get(key)
            if isinstance(value, list):
                action_rows = value
                break
        if not action_rows:
            single_tool = str(act.get("tool", "") or "").strip()
            if single_tool:
                single_args = act.get("args", act.get("parameters", {}))
                if not isinstance(single_args, dict):
                    single_args = {}
                action_rows = [{"tool": single_tool, "args": single_args}]
        merged["tool_plan"] = self._normalize_tool_plan(action_rows)
        if task_list:
            merged["subtasks"] = list(task_list)

        mode = str(act.get("mode", "") or "").strip().lower()
        final_answer = str(merged.get("final_answer", "") or "").strip()
        ask_user_prompt = str(merged.get("ask_user_prompt", "") or "").strip()
        stop_reason = str(merged.get("stop_reason", "") or "").strip()

        # 允许通过 act.mode 或 message 前缀声明控制语义，避免再引入额外字段。
        if mode in {"ask_user", "ask-user", "askuser"}:
            ask_user_prompt = message
        elif mode in {"stop", "blocked", "abort", "cannot_complete", "fail"}:
            stop_reason = message
        elif mode in {"final", "finalize", "final_answer", "done", "complete"}:
            final_answer = message

        lowered_message = message.lower()
        if lowered_message.startswith("ask_user:"):
            ask_user_prompt = message.split(":", 1)[1].strip()
        elif lowered_message.startswith("final:"):
            final_answer = message.split(":", 1)[1].strip()
        elif lowered_message.startswith("stop:"):
            stop_reason = message.split(":", 1)[1].strip()

        req = str(request_type or "").strip().lower()
        if req == "initial_planning":
            if task_complete:
                final_answer = final_answer or message
            if not (task_list or ask_user_prompt or final_answer or stop_reason):
                return merged, "initial_planning requires task_list, or a message describing ask/stop/final action."
        elif req == "subtask_request":
            if task_complete:
                final_answer = final_answer or message
                merged["subtask_status"] = "done"
            elif merged["tool_plan"]:
                merged["subtask_status"] = "running"
            elif ask_user_prompt or stop_reason:
                merged["subtask_status"] = "blocked"
            else:
                merged["subtask_status"] = "done"
            if not (merged["tool_plan"] or ask_user_prompt or final_answer or stop_reason):
                return merged, "subtask_request requires tool actions or a message describing ask/stop/final action."
        elif req == "final_synthesis":
            if task_complete:
                final_answer = final_answer or message
            else:
                stop_reason = stop_reason or message
            if not (final_answer or stop_reason):
                return merged, "final_synthesis requires final answer or stop reason in message."

        merged["ask_user_prompt"] = ask_user_prompt or None
        merged["final_answer"] = final_answer or None
        merged["stop_reason"] = stop_reason or None

        reason_parts: List[str] = []
        if task_goal_update:
            reason_parts.append(f"task_goal_update={task_goal_update}")
        reason_parts.append(message)
        merged["reason_summary"] = " | ".join([part for part in reason_parts if part]).strip()
        return merged, ""

    def _invoke_big_brain_with_request_act(
        self,
        *,
        payload: Dict[str, Any],
        system_prompt: str,
        fallback: Dict[str, Any],
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        用 request_act 工具驱动大脑决策。

        若工具调用缺失/参数不合规，会把失败信息回注给大脑并重试。
        """
        model = self.thinking_model
        if model is None or not hasattr(model, "invoke"):
            return dict(fallback)

        tools = self._build_big_brain_native_tools()
        request_type = str((payload or {}).get("request_type", "") or "").strip().lower()
        last_error = ""

        for attempt in range(1, max(1, int(max_retries or 1)) + 1):
            round_payload = dict(payload or {})
            if last_error:
                round_payload["request_act_error_feedback"] = {
                    "error": last_error,
                    "retry_index": attempt - 1,
                    "instruction": "You must call request_act tool in this round with valid arguments.",
                }

            user_text = json.dumps(round_payload, ensure_ascii=False, indent=2)
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_text)]

            try:
                response = model.invoke(messages, tools=tools)
            except TypeError:
                try:
                    response = model.invoke(messages)
                except Exception as exc:
                    last_error = f"invoke_failed: {exc}"
                    continue
            except Exception as exc:
                last_error = f"invoke_failed: {exc}"
                continue

            raw_tool_calls = list(getattr(response, "tool_calls", []) or [])
            request_act_call = None
            for row in raw_tool_calls:
                if not isinstance(row, dict):
                    continue
                tool_name = str(row.get("tool", "") or "").strip()
                if tool_name == self.REQUEST_ACT_TOOL_NAME:
                    request_act_call = row
                    break

            if request_act_call is None:
                last_error = "missing_request_act_tool_call"
                continue

            params = request_act_call.get("params", {})
            normalized, err = self._normalize_request_act_result(
                params=params if isinstance(params, dict) else {},
                request_type=request_type,
                fallback=fallback,
            )
            if err:
                last_error = err
                continue
            return normalized

        failed = dict(fallback or {})
        reason_prefix = str(failed.get("reason_summary", "") or "").strip()
        failure_note = f"request_act_protocol_failed: {last_error or 'unknown_error'}"
        failed["reason_summary"] = f"{reason_prefix}; {failure_note}".strip("; ").strip()
        if "stop_reason" in failed and not failed.get("stop_reason"):
            failed["stop_reason"] = failure_note
        return failed

    def _inject_tool_results_by_ids(
        self,
        *,
        thread_state: Dict[str, Any],
        call_ids: List[str],
        max_items: int = 8,
        per_item_chars: int = 4000,
        max_total_chars: int = 20000,
    ) -> str:
        """
        根据 call_id 列表提取工具结果并拼接为字符串，供 Think Brain 读取。

        说明：
        - 优先使用完整 output；为空时退回 summary。
        - 为防止上下文失控，做条目数/单条长度/总长度截断。
        """
        ids: List[str] = []
        for item in list(call_ids or []):
            text = str(item or "").strip()
            if text and text not in ids:
                ids.append(text)
        if not ids:
            return ""

        tool_calls = dict((thread_state or {}).get("tool_calls", {}) or {})
        chunks: List[str] = []
        used_chars = 0
        count = 0

        for call_id in ids:
            if count >= max(1, int(max_items or 1)):
                break
            row = dict(tool_calls.get(call_id, {}) or {})
            if not row:
                chunk = f"[call_id={call_id}] <missing_tool_result>"
            else:
                tool_name = str(row.get("tool", "") or "").strip()
                args = row.get("args", {})
                try:
                    args_text = json.dumps(args if isinstance(args, dict) else {}, ensure_ascii=False)
                except Exception:
                    args_text = "{}"
                output_text = str(row.get("output", "") or "").strip()
                if not output_text:
                    output_text = str(row.get("summary", "") or "").strip()
                if len(output_text) > max(1, int(per_item_chars or 1)):
                    keep = max(1, int(per_item_chars or 1))
                    output_text = output_text[:keep] + "\n...<truncated>"
                chunk = (
                    f"[call_id={call_id}] tool={tool_name}\n"
                    f"args={args_text}\n"
                    f"result=\n{output_text}"
                )

            chunk_size = len(chunk)
            if used_chars + chunk_size > max(1, int(max_total_chars or 1)):
                remain = max(0, max(1, int(max_total_chars or 1)) - used_chars)
                if remain > 0:
                    chunks.append(chunk[:remain] + "\n...<total_truncated>")
                break
            chunks.append(chunk)
            used_chars += chunk_size
            count += 1

        return "\n\n".join(chunks).strip()

    def _active_tool_names(self) -> List[str]:
        """返回当前激活工具名列表（去重、排序）。"""
        names = []
        for name in (getattr(self, "tools", {}) or {}).keys():
            text = str(name or "").strip()
            if text:
                names.append(text)
        return sorted(set(names))

    def _build_allowed_tools_guidance(self) -> str:
        """构造注入给模型的“允许工具白名单”提示文本。"""
        names = self._active_tool_names()
        if not names:
            return "allowed_tools: []"
        lines = [
            "Allowed tool names (use exact strings when calling tools):",
            ", ".join(names),
        ]
        return "\n".join(lines)

    def _normalize_tool_name(self, tool_name: str) -> str:
        """将模型输出的工具名标准化到运行时可执行名称。"""
        raw = str(tool_name or "").strip()
        if not raw:
            return ""
        lowered = raw.lower()
        active = set(self._active_tool_names())
        if raw in active:
            return raw
        if lowered in active:
            return lowered
        # 大小写不敏感匹配，避免模型输出大小写偏差导致失配。
        for item in active:
            if item.lower() == lowered:
                return item
        return raw

    def _normalize_tool_plan(self, value: Any) -> List[Dict[str, Any]]:
        """
        规范化模型返回的 tool_plan。

        目标：
        - 过滤非法条目；
        - 标准化工具名；
        - 保证 args 为 dict；
        - 兜底生成 call_id。
        """
        if not isinstance(value, list):
            return []
        plans: List[Dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            tool_name = self._normalize_tool_name(str(item.get("tool", "") or "").strip())
            if not tool_name:
                continue
            args = item.get("args", {})
            if not isinstance(args, dict):
                args = {}
            plans.append(
                {
                    "call_id": str(item.get("call_id", "") or "").strip() or f"call_{uuid.uuid4().hex[:8]}",
                    "tool": tool_name,
                    "args": args,
                    "parallel_group": str(item.get("parallel_group", "") or "").strip(),
                }
            )
        return plans

    def _normalize_task_board(self, value: Any) -> List[Dict[str, Any]]:
        """
        规范化大脑产出的任务板。

        若模型未给出有效 subtasks，则回退到单一默认子任务，保证流程可推进。
        """
        rows: List[Dict[str, Any]] = []
        if isinstance(value, list):
            for index, item in enumerate(value, 1):
                if not isinstance(item, dict):
                    continue
                subtask_id = str(item.get("subtask_id", "") or "").strip() or f"s{index}"
                objective = str(item.get("objective", "") or "").strip() or f"Subtask {index}"
                deps = item.get("dependencies", [])
                if not isinstance(deps, list):
                    deps = []
                acceptance = item.get("acceptance_criteria", [])
                if not isinstance(acceptance, list):
                    acceptance = []
                rows.append(
                    {
                        "subtask_id": subtask_id,
                        "objective": objective,
                        "dependencies": deps,
                        "required_evidence": item.get("required_evidence", []),
                        "acceptance_criteria": acceptance,
                        "status": str(item.get("status", "todo") or "todo").strip().lower(),
                        "tries": 0,
                    }
                )
        if not rows:
            rows.append(
                {
                    "subtask_id": "s1",
                    "objective": "Resolve the task with available tools and produce final answer",
                    "dependencies": [],
                    "required_evidence": [],
                    "acceptance_criteria": ["Provide a concrete and correct final answer"],
                    "status": "todo",
                    "tries": 0,
                }
            )
        return rows

    def _build_thread_state(self, *, task: str, thread_id: str) -> Dict[str, Any]:
        """构建线程级状态容器。"""
        return {
            # 线程与任务基础信息。
            "thread_id": thread_id,
            "task": task,
            "created_at": int(time.time()),
            # 大脑规划后的任务板。
            "task_board": [],
            # 工具调用记录：按 call_id 建索引，并维护调用顺序。
            "tool_calls": {},
            "tool_call_order": [],
            "recent_tool_call_ids": [],
            # 上下文压缩摘要（全局/小脑/大脑）。
            "checkpoint_summary": "",
            "small_compact_summary": "",
            "big_compact_summary": "",
            # 小脑通过 request_think 提交给大脑的最近一次请求。
            "small_brain_handoff": None,
            # 预留：循环打击计数，可用于后续收敛策略。
            "loop_strike": 0,
        }

    def _estimate_context_ratio(self, state: Dict[str, Any]) -> float:
        """
        粗估当前上下文压力比例。

        通过摘要文本与最近工具摘要长度估算“已使用字符 / 预算字符”。
        """
        budget_chars = max(1, int(getattr(self, "max_context_tokens", 128000) or 128000) * 4)
        used = len(str(state.get("checkpoint_summary", "") or ""))
        used += len(str(state.get("small_compact_summary", "") or ""))
        used += len(str(state.get("big_compact_summary", "") or ""))
        for call_id in list(state.get("recent_tool_call_ids", []) or [])[-12:]:
            row = dict(state.get("tool_calls", {}).get(call_id, {}) or {})
            used += len(str(row.get("summary", "") or ""))
        return min(1.5, float(used) / float(budget_chars))

    def _compact_state(self, state: Dict[str, Any], *, scope: str, reason: str) -> str:
        """
        生成并写入紧凑摘要。

        摘要只保留最关键的上下文元信息，避免上下文持续膨胀。
        """
        compact_lines = [
            f"scope={scope}",
            f"reason={reason}",
            f"task={state.get('task', '')}",
            f"open_subtasks={len([s for s in state.get('task_board', []) if s.get('status') != 'done'])}",
            f"recent_calls={','.join(list(state.get('recent_tool_call_ids', []) or [])[-6:])}",
        ]
        summary = "\n".join(compact_lines).strip()
        if scope == "big":
            state["big_compact_summary"] = summary
        else:
            state["small_compact_summary"] = summary
        state["checkpoint_summary"] = summary
        return summary

    def _next_subtask(self, task_board: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """从任务板中选取下一个可推进子任务，并标记为 running。"""
        for row in task_board:
            status = str(row.get("status", "todo") or "todo").strip().lower()
            if status in {"todo", "running"}:
                row["status"] = "running"
                return row
        return None

    def _run_tool_plan(
        self,
        *,
        tool_plan: List[Dict[str, Any]],
        thread_state: Dict[str, Any],
        loaded_skill_names: List[str],
        skill_context: str,
        thread_id: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], str]:
        """
        按顺序执行工具计划，并沉淀事件与调用记录。

        返回：
        - events: 面向外部流式输出的 tool_exec/tool_result 事件；
        - records: 结构化工具调用记录（含参数与摘要）；
        - loaded_skill_names / skill_context: 执行后更新的技能上下文。
        """
        # 工具执行过程中产生的工件（例如技能加载副作用）统一复用一个容器。
        tool_artifacts: List[Dict[str, Any]] = []
        events: List[Dict[str, Any]] = []
        records: List[Dict[str, Any]] = []
        total = len(tool_plan)
        for index, row in enumerate(tool_plan, 1):
            call_id = str(row.get("call_id", "") or f"call_{uuid.uuid4().hex[:8]}").strip()
            tool_name = str(row.get("tool", "") or "").strip()
            params = dict(row.get("args", {}) or {})
            # 先发出“工具开始执行”事件，便于前端/日志实时感知进度。
            events.append({"type": "tool_exec", "content": f"[Tool {index}/{total}] {tool_name}"})
            try:
                output, loaded_skill_names, skill_context = self._execute_single_tool_action(
                    tool_name=tool_name,
                    params=params,
                    loaded_skill_names=loaded_skill_names,
                    skill_context=skill_context,
                    tool_artifacts=tool_artifacts,
                    thread_id=thread_id,
                )
                output_text = str(output or "")
            except Exception as exc:
                # 工具异常不抛出到外层循环，转成可观测错误文本继续推进。
                output_text = f"<error>{exc}</error>"
            # 工具执行结束事件，附带 call_id 便于关联。
            events.append({"type": "tool_result", "content": output_text, "tool_name": tool_name, "call_id": call_id})
            record = {
                "call_id": call_id,
                "tool": tool_name,
                "args": params,
                "output": output_text,
                # summary 用于上下文估算与后续模型暴露，避免传递完整长输出。
                "summary": output_text[:1200],
                "created_at": int(time.time()),
            }
            records.append(record)
            # 同步更新线程态中的工具调用索引与最近调用窗口。
            thread_state["tool_calls"][call_id] = dict(record)
            thread_state["tool_call_order"].append(call_id)
            thread_state["recent_tool_call_ids"] = list(thread_state["tool_call_order"][-12:])
        return events, records, loaded_skill_names, skill_context

    def _stream_answer_events(self, answer: str, chunk_size: int = 64) -> Iterable[Dict[str, Any]]:
        """将最终答案切片为 token 事件流，便于前端渐进式渲染。"""
        text = str(answer or "")
        for i in range(0, len(text), chunk_size):
            yield {"type": "token", "content": text[i : i + chunk_size]}

    def get_context(self, thread_id: str = "default", checkpoint_id: str = "") -> dict:
        """读取指定线程上下文快照。"""
        # 兼容旧接口参数，当前未使用 checkpoint_id。
        _ = checkpoint_id
        normalized = self._normalize_thread_id(thread_id)
        with self._request_lock:
            state = dict(self._request_threads.get(normalized, {}) or {})
        return state

    def compact_context(self, thread_id: str = "default") -> tuple:
        """对指定线程执行一次手动压缩，并返回压缩提示与摘要。"""
        normalized = self._normalize_thread_id(thread_id)
        with self._request_lock:
            state = self._request_threads.get(normalized)
            if not state:
                return ("No context to compact.", "")
            summary = self._compact_state(state, scope="big", reason="manual_compact")
            self._request_threads[normalized] = dict(state)
        return ("Context compacted.", summary)

    def invoke(
        self,
        task: str,
        thread_id: str = "default",
        thinking_context: bool = False,
        system_prompt_append: str = "",
    ):
        """
        主执行入口（流式生成事件）。

        核心流程：
        1) 小脑最多自循环 N 轮，优先直接回答或直接调工具；
        2) 需要深度推理时升级到大脑，先做全局规划；
        3) 大脑按子任务逐步请求动作（可带工具计划）并收敛；
        4) 产出最终答案或在超限时退出。
        """
        # 当前实现未启用 thinking_context，保留参数兼容调用方。
        _ = thinking_context
        normalized_thread_id = self._normalize_thread_id(thread_id)
        # 合并初始化追加提示与本次调用追加提示。
        runtime_append = "\n\n".join(
            [part for part in [self._init_system_prompt_append, str(system_prompt_append or "").strip()] if part]
        ).strip()
        act_brain_system_prompt = _load_prompt_asset(
            "prompts",
            "request_act_base.md",
            fallback=(
                "# 角色定位\n\n"
                "你是 Klynx RequestOrchestrator 的 Act Brain。"
            ),
        )
        think_brain_system_prompt = _load_prompt_asset(
            "prompts",
            "request_think_base.md",
            fallback=(
                "# 角色定位\n\n"
                "你是 Klynx RequestOrchestrator 的 Think Brain。"
            ),
        )
        with self._request_lock:
            # 每次 invoke 基于传入 task 重建线程状态。
            state = self._build_thread_state(task=str(task or ""), thread_id=normalized_thread_id)
            self._request_threads[normalized_thread_id] = dict(state)

        # 双脑任一未配置都无法进入编排流程。
        if self.fast_model is None or self.thinking_model is None:
            message = "RequestOrchestratorAgent requires model configuration for both small and big brain."
            yield {"type": "error", "content": message}
            yield {"type": "done", "content": "stopped", "answer": message, "task_completed": False, "iteration_count": 0, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
            return

        # 技能上下文在整轮任务中累积并传递给后续工具调用。
        loaded_skill_names: List[str] = []
        skill_context = ""
        # 统一迭代计数：覆盖小脑与大脑阶段。
        iteration_count = 0

        # 阶段 1：小脑预循环（默认 <= 3 轮）
        for loop_index in range(1, self.small_self_loop_max + 1):
            iteration_count += 1
            # 上下文压力过高时，先进行轻量压缩再继续。
            ratio = self._estimate_context_ratio(state)
            if ratio >= self.context_warn_ratio:
                summary = self._compact_state(state, scope="small", reason=f"context_pressure_preloop_{loop_index}")
                yield {"type": "info", "content": f"[Compact][small] {summary}"}

            # 小脑输入：任务本身 + 最近调用窗口 + 白名单工具。
            # 约定：
            # - 首次 request_think 之前，系统自动把已有工具执行结果注入给小脑；
            # - 首次 request_think 之后，小脑如需结果应再次调用工具自行获取（手动模式）。
            has_requested_think = bool(state.get("small_brain_handoff"))
            recent_call_ids = list(state.get("recent_tool_call_ids", []) or [])
            auto_injected_tool_results = ""
            if not has_requested_think and recent_call_ids:
                auto_injected_tool_results = self._inject_tool_results_by_ids(
                    thread_state=state,
                    call_ids=recent_call_ids,
                    max_items=max(1, len(recent_call_ids)),
                    per_item_chars=6000,
                    max_total_chars=80000,
                )
            allowed_tools = self._active_tool_names()
            small_payload = {
                "request_type": "small_loop",
                "task": task,
                "loop_index": loop_index,
                "max_loop": self.small_self_loop_max,
                "recent_tool_call_ids": recent_call_ids,
                "recent_tool_results": auto_injected_tool_results,
                "tool_result_injection_mode": (
                    "auto_before_first_request_think"
                    if not has_requested_think
                    else "manual_tool_call_after_first_request_think"
                ),
                "checkpoint_summary": state.get("checkpoint_summary", ""),
                "allowed_tools": allowed_tools,
            }
            small_reply = yield from self._invoke_small_brain_with_tools_stream(
                payload=small_payload,
                system_prompt=(
                    f"{act_brain_system_prompt}\n\n"
                    f"{self._build_allowed_tools_guidance()}\n"
                    f"{runtime_append}"
                ).strip(),
            )

            # 读取小脑决策结果。
            direct_answer = str(small_reply.get("direct_answer", "") or "").strip()
            direct_answer_streamed = bool(small_reply.get("_direct_answer_streamed", False))
            tool_plan = self._normalize_tool_plan(small_reply.get("tool_plan", []))
            request_think = small_reply.get("request_think")
            need_think = bool(request_think) or (not direct_answer and not tool_plan)

            # 小脑可直接回答时，立即流式输出并结束。
            if direct_answer:
                if not direct_answer_streamed:
                    for event in self._stream_answer_events(direct_answer):
                        yield event
                yield {
                    "type": "done",
                    "content": "",
                    "answer": direct_answer,
                    "iteration_count": iteration_count,
                    "task_completed": True,
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
                return

            # 小脑给出工具计划且判定无需大脑时，直接执行并继续下一轮小脑判断。
            if tool_plan and not need_think:
                events, records, loaded_skill_names, skill_context = self._run_tool_plan(
                    tool_plan=tool_plan,
                    thread_state=state,
                    loaded_skill_names=loaded_skill_names,
                    skill_context=skill_context,
                    thread_id=normalized_thread_id,
                )
                for event in events:
                    yield event
                if records:
                    state["recent_tool_call_ids"] = [row["call_id"] for row in records][-12:]
                continue
            # 需要大脑时，跳出小脑循环。
            if need_think:
                if isinstance(request_think, dict):
                    state["small_brain_handoff"] = dict(request_think)
                break

        # 阶段 2：升级到大脑，先做全局规划。
        yield {"type": "info", "content": "[RequestAgent] Escalate to big brain."}
        handoff_request = dict(state.get("small_brain_handoff", {}) or {})
        include_recent_call_ids = list(handoff_request.get("include_recent_call_ids", []) or [])
        handoff_tool_results = self._inject_tool_results_by_ids(
            thread_state=state,
            call_ids=include_recent_call_ids,
        )
        recent_tool_results = self._inject_tool_results_by_ids(
            thread_state=state,
            call_ids=list(state.get("recent_tool_call_ids", []) or [])[-8:],
        )
        plan_payload = {
            "request_type": "initial_planning",
            "task": task,
            "done_criteria": ["deliver correct final answer with evidence"],
            "recent_tool_call_ids": list(state.get("recent_tool_call_ids", []) or []),
            "checkpoint_summary": state.get("checkpoint_summary", ""),
            "handoff_request": handoff_request,
            "handoff_tool_results": handoff_tool_results,
            "recent_tool_results": recent_tool_results,
        }
        plan_fallback = {"decision": "plan", "subtasks": [], "final_answer": None, "ask_user_prompt": None, "stop_reason": None, "reason_summary": ""}
        plan_reply = self._invoke_big_brain_with_request_act(
            system_prompt=f"{think_brain_system_prompt}\n\n{runtime_append}".strip(),
            payload=plan_payload,
            fallback=plan_fallback,
        )
        # 若大脑在规划阶段即可直接收敛，直接返回最终答案。
        final_answer = str(plan_reply.get("final_answer", "") or "").strip()
        if final_answer:
            for event in self._stream_answer_events(final_answer):
                yield event
            yield {"type": "done", "content": "", "answer": final_answer, "iteration_count": iteration_count, "task_completed": True, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
            return

        # 初始化任务板并写入一次大脑侧压缩摘要。
        state["task_board"] = self._normalize_task_board(plan_reply.get("subtasks", []))
        compact_summary = self._compact_state(state, scope="big", reason="after_big_plan")
        yield {"type": "info", "content": f"[Compact][big] {compact_summary}"}

        # 阶段 3：按任务板推进子任务，直到完成或达到迭代上限。
        max_steps = self.max_iterations if isinstance(self.max_iterations, int) and self.max_iterations > 0 else 60
        while iteration_count < max_steps:
            # 大脑阶段同样执行上下文压力管理。
            ratio = self._estimate_context_ratio(state)
            if ratio >= self.context_hard_ratio:
                summary = self._compact_state(state, scope="big", reason="context_pressure_hard")
                yield {"type": "warning", "content": f"[Compact][hard] {summary}"}
            elif ratio >= self.context_warn_ratio:
                summary = self._compact_state(state, scope="big", reason="context_pressure_warn")
                yield {"type": "info", "content": f"[Compact][warn] {summary}"}

            # 选取下一个可推进子任务；若为空，进入最终收敛生成。
            subtask = self._next_subtask(list(state.get("task_board", []) or []))
            if not subtask:
                final_payload = {
                    "request_type": "final_synthesis",
                    "task": task,
                    "task_board": state.get("task_board", []),
                    "checkpoint_summary": state.get("checkpoint_summary", ""),
                    "recent_tool_call_ids": list(state.get("recent_tool_call_ids", []) or []),
                    "recent_tool_results": self._inject_tool_results_by_ids(
                        thread_state=state,
                        call_ids=list(state.get("recent_tool_call_ids", []) or [])[-8:],
                    ),
                }
                final_fallback = {"final_answer": "Task flow completed.", "reason_summary": ""}
                final_reply = self._invoke_big_brain_with_request_act(
                    system_prompt=f"{think_brain_system_prompt}\n\n{runtime_append}".strip(),
                    payload=final_payload,
                    fallback=final_fallback,
                )
                final_answer = str(final_reply.get("final_answer", "") or "").strip() or "Task flow completed."
                for event in self._stream_answer_events(final_answer):
                    yield event
                yield {"type": "done", "content": "", "answer": final_answer, "iteration_count": iteration_count, "task_completed": True, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
                return

            iteration_count += 1
            # 仅向大脑暴露最近调用窗口，避免历史噪声过多。
            expose_call_ids = list(state.get("recent_tool_call_ids", []) or [])[-8:]
            expose_tool_results = self._inject_tool_results_by_ids(
                thread_state=state,
                call_ids=expose_call_ids,
            )
            # 由“小脑视角”构造子任务请求，再交给大脑给出下一步动作。
            subtask_request = {
                "task_id": normalized_thread_id,
                "request_type": "subtask_request",
                "subtask_id": subtask.get("subtask_id", ""),
                "subtask_goal": subtask.get("objective", ""),
                "required_context": [state.get("checkpoint_summary", "")],
                "expose_call_ids": expose_call_ids,
                "expose_tool_results": expose_tool_results,
                "subtask_progress": {"status": subtask.get("status", "running"), "tries": int(subtask.get("tries", 0) or 0)},
                "question_for_think": "What should the next action be?",
            }
            step_fallback = {
                "response_type": "plan_or_step",
                "decision": "continue",
                "tool_plan": [],
                "subtask_status": "done",
                "ask_user_prompt": None,
                "final_answer": None,
                "stop_reason": None,
                "subagent_plan": {"use_subagent": False, "reason": ""},
                "reason_summary": "",
            }
            allowed_tools = self._active_tool_names()
            step_payload = dict(subtask_request)
            step_payload["allowed_tools"] = allowed_tools
            step_reply = self._invoke_big_brain_with_request_act(
                system_prompt=(
                    f"{think_brain_system_prompt}\n\n"
                    f"{self._build_allowed_tools_guidance()}\n"
                    f"{runtime_append}"
                ).strip(),
                payload=step_payload,
                fallback=step_fallback,
            )

            # 读取大脑步骤决策。
            ask_user_prompt = str(step_reply.get("ask_user_prompt", "") or "").strip()
            stop_reason = str(step_reply.get("stop_reason", "") or "").strip()
            final_answer = str(step_reply.get("final_answer", "") or "").strip()
            tool_plan = self._normalize_tool_plan(step_reply.get("tool_plan", []))
            subagent_plan = dict(step_reply.get("subagent_plan", {}) or {})

            # 大脑要求向用户追问时，挂起等待用户输入。
            if ask_user_prompt:
                yield {"type": "answer", "content": ask_user_prompt}
                yield {"type": "done", "content": "wait_user", "answer": ask_user_prompt, "iteration_count": iteration_count, "task_completed": False, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
                return
            # 大脑主动停止（例如不可达条件）时退出。
            if stop_reason:
                yield {"type": "warning", "content": stop_reason}
                yield {"type": "done", "content": "stopped", "answer": stop_reason, "iteration_count": iteration_count, "task_completed": False, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
                return
            # 大脑直接给出最终答案时结束。
            if final_answer:
                for event in self._stream_answer_events(final_answer):
                    yield event
                yield {"type": "done", "content": "", "answer": final_answer, "iteration_count": iteration_count, "task_completed": True, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
                return

            # 子代理请求：当前仅占位逻辑，真实子代理编排待扩展。
            if bool(subagent_plan.get("use_subagent", False)):
                if self.enable_subagent:
                    subagent_summary = "subagent completed (summary placeholder)."
                    yield {"type": "summary", "content": f"[Subagent] {subagent_summary}"}
                else:
                    yield {"type": "warning", "content": "[Subagent] requested by big brain but disabled by config."}

            # 执行大脑给出的工具计划。
            if tool_plan:
                events, records, loaded_skill_names, skill_context = self._run_tool_plan(
                    tool_plan=tool_plan,
                    thread_state=state,
                    loaded_skill_names=loaded_skill_names,
                    skill_context=skill_context,
                    thread_id=normalized_thread_id,
                )
                for event in events:
                    yield event
                if records:
                    state["recent_tool_call_ids"] = [row["call_id"] for row in records][-12:]

            # 根据本轮回复更新子任务状态与重试计数。
            subtask_status = str(step_reply.get("subtask_status", "") or "").strip().lower()
            subtask["tries"] = int(subtask.get("tries", 0) or 0) + 1
            if subtask_status in {"done", "completed"}:
                subtask["status"] = "done"
            elif subtask_status in {"blocked", "failed"}:
                subtask["status"] = "blocked"
            else:
                if not tool_plan:
                    subtask["status"] = "done"
                else:
                    subtask["status"] = "running"

            # 输出当前任务板进度摘要。
            done_count = sum(1 for row in state.get("task_board", []) if str(row.get("status", "")).lower() == "done")
            total_count = len(state.get("task_board", []))
            yield {"type": "summary", "content": f"[Subtask] done={done_count}/{total_count}"}

        # 超过最大迭代仍未收敛，返回超时结束信号。
        timeout_answer = "RequestOrchestratorAgent reached max iterations before completion."
        yield {"type": "warning", "content": timeout_answer}
        yield {"type": "done", "content": "timeout", "answer": timeout_answer, "iteration_count": iteration_count, "task_completed": False, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
