"""
Model registry and setup helper.

This module maps model aliases to LiteLLM provider/model routes and exposes
`setup(...)` to build a configured `LiteLLMChat` instance.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .adapter import LiteLLMChat


MODEL_REGISTRY = {
    # ---------------- DeepSeek ----------------
    "deepseek-reasoner": {
        "model": "deepseek/deepseek-reasoner",
        "env_key": "DEEPSEEK_API_KEY",
        "max_context_tokens": 128000,
        "supports_native_tool_calling": True,
        "description": "DeepSeek Reasoner",
    },
    "deepseek-chat": {
        "model": "deepseek/deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "max_context_tokens": 128000,
        "supports_native_tool_calling": True,
        "description": "DeepSeek Chat",
    },
    # ---------------- OpenAI ----------------
    "gpt-5.3": {
        "model": "openai/gpt-5.3",
        "env_key": "OPENAI_API_KEY",
        "max_context_tokens": 400000,
        "supports_native_tool_calling": True,
        "description": "GPT-5.3",
    },
    "gpt-5.2": {
        "model": "openai/gpt-5.2",
        "env_key": "OPENAI_API_KEY",
        "max_context_tokens": 400000,
        "supports_native_tool_calling": True,
        "description": "GPT-5.2",
    },
    "o3-mini": {
        "model": "openai/o3-mini",
        "env_key": "OPENAI_API_KEY",
        "max_context_tokens": 200000,
        "supports_native_tool_calling": True,
        "description": "OpenAI o3-mini",
    },
    "gpt-4o": {
        "model": "openai/gpt-4o",
        "env_key": "OPENAI_API_KEY",
        "max_context_tokens": 128000,
        "supports_native_tool_calling": True,
        "description": "GPT-4o",
    },
    "o1-preview": {
        "model": "openai/o1-preview",
        "env_key": "OPENAI_API_KEY",
        "max_context_tokens": 128000,
        "supports_native_tool_calling": True,
        "description": "OpenAI o1-preview",
    },
    # ---------------- Anthropic ----------------
    "claude-4.6-sonnet": {
        "model": "anthropic/claude-sonnet-4-6",
        "env_key": "ANTHROPIC_API_KEY",
        "max_context_tokens": 200000,
        "supports_native_tool_calling": True,
        "description": "Claude 4.6 Sonnet",
    },
    "claude-4.5-opus": {
        "model": "anthropic/claude-opus-4-5-20251101",
        "env_key": "ANTHROPIC_API_KEY",
        "max_context_tokens": 200000,
        "supports_native_tool_calling": True,
        "description": "Claude 4.5 Opus",
    },
    "claude-3.5-sonnet": {
        "model": "anthropic/claude-3-5-sonnet-20241022",
        "env_key": "ANTHROPIC_API_KEY",
        "max_context_tokens": 200000,
        "supports_native_tool_calling": True,
        "description": "Claude 3.5 Sonnet",
    },
    # ---------------- Gemini ----------------
    "gemini-3.1-pro": {
        "model": "gemini/gemini-3.1-pro",
        "env_key": "GEMINI_API_KEY",
        "max_context_tokens": 1048576,
        "supports_native_tool_calling": True,
        "description": "Gemini 3.1 Pro",
    },
    "gemini-2.5-flash": {
        "model": "gemini/gemini-2.5-flash",
        "env_key": "GEMINI_API_KEY",
        "max_context_tokens": 1048576,
        "supports_native_tool_calling": True,
        "description": "Gemini 2.5 Flash",
    },
    "gemini-1.5-pro": {
        "model": "gemini/gemini-1.5-pro",
        "env_key": "GEMINI_API_KEY",
        "max_context_tokens": 2097152,
        "supports_native_tool_calling": True,
        "description": "Gemini 1.5 Pro",
    },
    # ---------------- GLM ----------------
    "glm-5": {
        "model": "openai/glm-5",
        "env_key": "GLM_API_KEY",
        "max_context_tokens": 200000,
        "supports_native_tool_calling": True,
        "description": "GLM-5",
        "default_kwargs": {"api_base": "https://open.bigmodel.cn/api/paas/v4/"},
    },
    "glm-4-plus": {
        "model": "openai/glm-4-plus",
        "env_key": "GLM_API_KEY",
        "max_context_tokens": 128000,
        "supports_native_tool_calling": True,
        "description": "GLM-4 Plus",
        "default_kwargs": {"api_base": "https://open.bigmodel.cn/api/paas/v4/"},
    },
    "glm-4-long": {
        "model": "openai/glm-4-long",
        "env_key": "GLM_API_KEY",
        "max_context_tokens": 1000000,
        "supports_native_tool_calling": True,
        "description": "GLM-4 Long",
        "default_kwargs": {"api_base": "https://open.bigmodel.cn/api/paas/v4/"},
    },
    "glm-4-flash": {
        "model": "openai/glm-4-flash",
        "env_key": "GLM_API_KEY",
        "max_context_tokens": 128000,
        "supports_native_tool_calling": True,
        "description": "GLM-4 Flash",
        "default_kwargs": {"api_base": "https://open.bigmodel.cn/api/paas/v4/"},
    },
    # ---------------- Moonshot ----------------
    "kimi-k2.5": {
        "model": "openai/kimi-k2.5",
        "env_key": ["KIMI_API_KEY", "MOONSHOT_API_KEY"],
        "max_context_tokens": 256000,
        "supports_native_tool_calling": True,
        "description": "Kimi k2.5",
        "default_kwargs": {
            "api_base": "https://api.moonshot.cn/v1",
            "extra_body": {"enable_thinking": True},
        },
        "capabilities": {
            "supports_native_tool_calling": True,
            "supports_temperature": False,
            "supports_top_p": False,
            "supports_parallel_tool_calls": True,
            "supports_streaming_tool_calls": True,
            "supports_usage_in_stream": True,
        },
    },
    "kimi": {
        "model": "openai/moonshot-v1-auto",
        "env_key": ["KIMI_API_KEY", "MOONSHOT_API_KEY"],
        "max_context_tokens": 128000,
        "supports_native_tool_calling": True,
        "description": "Kimi",
        "default_kwargs": {"api_base": "https://api.moonshot.cn/v1"},
    },
    # ---------------- MiniMax ----------------
    "minimax-m2.5": {
        "model": "minimax/MiniMax-M2.5",
        "env_key": "MINIMAX_API_KEY",
        "max_context_tokens": 204800,
        "supports_native_tool_calling": True,
        "description": "MiniMax M2.5",
        "default_kwargs": {
            "api_base": os.environ.get("MINIMAX_API_BASE", "https://api.minimaxi.com/v1"),
        },
        "capabilities": {
            "supports_native_tool_calling": True,
            "supports_temperature": True,
            "supports_top_p": True,
            "supports_parallel_tool_calls": True,
            "supports_streaming_tool_calls": True,
            "supports_usage_in_stream": True,
        },
    },
    "minimax-text-01": {
        "model": "minimax/MiniMax-M2.1",
        "env_key": "MINIMAX_API_KEY",
        "max_context_tokens": 200000,
        "supports_native_tool_calling": True,
        "description": "MiniMax M2.1",
        "default_kwargs": {
            "api_base": os.environ.get("MINIMAX_API_BASE", "https://api.minimaxi.com/v1"),
        },
        "capabilities": {
            "supports_native_tool_calling": True,
            "supports_temperature": True,
            "supports_top_p": True,
            "supports_parallel_tool_calls": True,
            "supports_streaming_tool_calls": True,
            "supports_usage_in_stream": True,
        },
    },
    # ---------------- Qwen ----------------
    "qwen-max-latest": {
        "model": "dashscope/qwen-max-latest",
        "env_key": "DASHSCOPE_API_KEY",
        "max_context_tokens": 131072,
        "supports_native_tool_calling": True,
        "description": "Qwen Max Latest",
    },
    "qwen-plus-latest": {
        "model": "dashscope/qwen-plus-latest",
        "env_key": "DASHSCOPE_API_KEY",
        "max_context_tokens": 1000000,
        "supports_native_tool_calling": True,
        "description": "Qwen Plus Latest",
    },
    # ---------------- Xiaomi MiMo ----------------
    "mimo-v2-pro": {
        "model": "xiaomi_mimo/mimo-v2-pro",
        "env_key": ["XIAOMI_MIMO_API_KEY", "MIMO_API_KEY"],
        "max_context_tokens": 1048576,
        "supports_native_tool_calling": True,
        "description": "Xiaomi MiMo v2 Pro",
        "default_kwargs": {
            "is_multimodal": False,
            "temperature": 0.3,
            "top_p": 0.95,
        },
        "capabilities": {
            "supports_native_tool_calling": True,
            "supports_temperature": True,
            "supports_top_p": True,
            "supports_multimodal": False,
            "supports_parallel_tool_calls": False,
            "supports_streaming_tool_calls": True,
            "supports_usage_in_stream": False,
            "allowed_openai_params": ["tools", "tool_choice"],
        },
    },
    "mimo-v2-omni": {
        "model": "xiaomi_mimo/mimo-v2-omni",
        "env_key": ["XIAOMI_MIMO_API_KEY", "MIMO_API_KEY"],
        "max_context_tokens": 262144,
        "supports_native_tool_calling": True,
        "description": "Xiaomi MiMo v2 Omni",
        "default_kwargs": {"is_multimodal": True},
        "capabilities": {
            "supports_native_tool_calling": True,
            "supports_temperature": True,
            "supports_top_p": True,
            "supports_multimodal": True,
            "supports_parallel_tool_calls": False,
            "supports_streaming_tool_calls": True,
            "supports_usage_in_stream": False,
            "allowed_openai_params": ["tools", "tool_choice"],
        },
    },
    "mimo-v2-flash": {
        "model": "xiaomi_mimo/mimo-v2-flash",
        "env_key": ["XIAOMI_MIMO_API_KEY", "MIMO_API_KEY"],
        "max_context_tokens": 262144,
        "supports_native_tool_calling": True,
        "description": "Xiaomi MiMo v2 Flash",
        "default_kwargs": {"is_multimodal": False},
        "capabilities": {
            "supports_native_tool_calling": True,
            "supports_temperature": True,
            "supports_top_p": True,
            "supports_multimodal": False,
            "supports_parallel_tool_calls": False,
            "supports_streaming_tool_calls": True,
            "supports_usage_in_stream": False,
            "allowed_openai_params": ["tools", "tool_choice"],
        },
    },
    # ---------------- OpenRouter ----------------
    "openrouter-auto": {
        "model": "openrouter/auto",
        "env_key": "OPENROUTER_API_KEY",
        "max_context_tokens": 262144,
        "supports_native_tool_calling": True,
        "description": "OpenRouter Auto Router",
        "default_kwargs": {
            "api_base": os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
        },
        "capabilities": {
            "supports_native_tool_calling": True,
            "supports_temperature": True,
            "supports_top_p": True,
            "supports_multimodal": True,
            "supports_parallel_tool_calls": True,
            "supports_streaming_tool_calls": True,
            "supports_usage_in_stream": True,
        },
    },
}


_PROVIDER_ALIASES = {
    "mimo": "xiaomi_mimo",
    "xiaomi-mimo": "xiaomi_mimo",
    "open-router": "openrouter",
}


_PROVIDER_ENV_KEYS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "glm": "GLM_API_KEY",
    "moonshot": ["KIMI_API_KEY", "MOONSHOT_API_KEY"],
    "minimax": "MINIMAX_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xiaomi_mimo": ["XIAOMI_MIMO_API_KEY", "MIMO_API_KEY"],
}


_PROVIDER_DEFAULT_KWARGS = {
    "glm": {"api_base": "https://open.bigmodel.cn/api/paas/v4/"},
    "moonshot": {"api_base": "https://api.moonshot.cn/v1"},
    "minimax": {
        "api_base": os.environ.get("MINIMAX_API_BASE", "https://api.minimaxi.com/v1"),
    },
    "openrouter": {
        "api_base": os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
    },
}


_PROVIDER_NATIVE_TOOL_SUPPORT = {
    "xiaomi_mimo": True,
}

_PROVIDER_MODEL_CAPABILITIES = {
    "openai": {
        "supports_native_tool_calling": True,
        "supports_temperature": True,
        "supports_top_p": True,
        "supports_parallel_tool_calls": True,
        "supports_streaming_tool_calls": True,
        "supports_usage_in_stream": True,
    },
    "moonshot": {
        "supports_native_tool_calling": True,
        "supports_temperature": True,
        "supports_top_p": True,
        "supports_parallel_tool_calls": True,
        "supports_streaming_tool_calls": True,
        "supports_usage_in_stream": True,
    },
    "minimax": {
        "supports_native_tool_calling": True,
        "supports_temperature": True,
        "supports_top_p": True,
        "supports_parallel_tool_calls": True,
        "supports_streaming_tool_calls": True,
        "supports_usage_in_stream": True,
    },
    "openrouter": {
        "supports_native_tool_calling": True,
        "supports_temperature": True,
        "supports_top_p": True,
        "supports_multimodal": True,
        "supports_parallel_tool_calls": True,
        "supports_streaming_tool_calls": True,
        "supports_usage_in_stream": True,
    },
    "xiaomi_mimo": {
        "supports_native_tool_calling": True,
        "supports_temperature": True,
        "supports_top_p": True,
        "supports_parallel_tool_calls": False,
        "supports_streaming_tool_calls": True,
        "supports_usage_in_stream": False,
        "allowed_openai_params": ["tools", "tool_choice"],
    },
}


def _normalize_provider_name(provider: Optional[str]) -> str:
    normalized = str(provider or "").strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def _infer_provider_name(model_string: str) -> str:
    candidate = str(model_string or "").strip()
    if "/" not in candidate:
        return _normalize_provider_name(candidate)
    return _normalize_provider_name(candidate.split("/", 1)[0])


def _find_registry_entry_by_model_route(model_route: str) -> Optional[Dict[str, Any]]:
    target = str(model_route or "").strip().lower()
    if not target:
        return None
    for alias, config in MODEL_REGISTRY.items():
        if not isinstance(config, dict):
            continue
        route = str(config.get("model", "") or "").strip().lower()
        if route == target:
            hit = dict(config)
            hit["_alias"] = alias
            return hit
    return None


def _provider_env_keys(provider_name: str):
    return _PROVIDER_ENV_KEYS.get(_normalize_provider_name(provider_name))


def _provider_default_kwargs(provider_name: str) -> Dict[str, Any]:
    defaults = _PROVIDER_DEFAULT_KWARGS.get(_normalize_provider_name(provider_name), {})
    return defaults.copy()


def _provider_supports_native_tool_calling(provider_name: str, default: bool = True) -> bool:
    normalized = _normalize_provider_name(provider_name)
    if normalized in _PROVIDER_NATIVE_TOOL_SUPPORT:
        return bool(_PROVIDER_NATIVE_TOOL_SUPPORT[normalized])
    return bool(default)


def _provider_model_capabilities(provider_name: str) -> Dict[str, Any]:
    normalized = _normalize_provider_name(provider_name)
    capabilities = _PROVIDER_MODEL_CAPABILITIES.get(normalized, {})
    return dict(capabilities) if isinstance(capabilities, dict) else {}


def _default_model_capabilities(model_string: str, supports_native_tool_calling: bool) -> Dict[str, Any]:
    lowered = str(model_string or "").strip().lower()
    controls_supported = not any(marker in lowered for marker in ("reasoner", "o1", "kimi-k2.5"))
    return {
        "supports_native_tool_calling": bool(supports_native_tool_calling),
        "supports_temperature": bool(controls_supported),
        "supports_top_p": bool(controls_supported),
        "supports_parallel_tool_calls": False,
        "supports_streaming_tool_calls": True,
        "supports_usage_in_stream": False,
    }


def _resolve_model_capabilities(
    *,
    model_string: str,
    provider_name: str,
    supports_native_tool_calling: bool,
    configured_capabilities: Optional[Dict[str, Any]] = None,
    runtime_capabilities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    capabilities = _default_model_capabilities(
        model_string=model_string,
        supports_native_tool_calling=supports_native_tool_calling,
    )
    capabilities = _merge_model_kwargs(capabilities, _provider_model_capabilities(provider_name))
    capabilities = _merge_model_kwargs(capabilities, configured_capabilities or {})
    capabilities = _merge_model_kwargs(capabilities, runtime_capabilities or {})
    return capabilities


def _merge_model_kwargs(default_kwargs: Dict[str, Any], runtime_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    merged = default_kwargs.copy()
    for key, value in runtime_kwargs.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            nested = merged[key].copy()
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _format_env_keys(env_keys: Any) -> str:
    if isinstance(env_keys, (list, tuple)):
        cleaned = [str(item).strip() for item in env_keys if str(item or "").strip()]
        return " / ".join(cleaned) if cleaned else "N/A"
    if env_keys:
        return str(env_keys)
    return "N/A"


def setup(provider: str, model_name: str = None, api_key: str = None, **kwargs) -> LiteLLMChat:
    """
    Build a model instance by provider/model or alias.

    Supported forms:
    1. setup("deepseek", "deepseek-reasoner")
    2. setup("gpt-4o")
    3. setup("openai", "gpt-4o", "sk-xxx")
    """

    if model_name is None:
        alias = str(provider or "").strip()
        if alias in MODEL_REGISTRY:
            config = MODEL_REGISTRY[alias]
            actual_model_string = config["model"]
            desc = config["description"]
            env_keys = config.get("env_key")
            default_kwargs = config.get("default_kwargs", {})
            registry_max_context_tokens = config.get("max_context_tokens", 128000)
            supports_native_tool_calling = config.get("supports_native_tool_calling", True)
            configured_capabilities = config.get("capabilities", {})
            resolved_provider = _infer_provider_name(actual_model_string)
        else:
            route_entry = _find_registry_entry_by_model_route(alias)
            if route_entry:
                actual_model_string = str(route_entry.get("model") or alias)
                desc = str(route_entry.get("description") or f"Model Route: {actual_model_string}")
                env_keys = route_entry.get("env_key")
                default_kwargs = route_entry.get("default_kwargs", {})
                registry_max_context_tokens = route_entry.get("max_context_tokens", 128000)
                supports_native_tool_calling = route_entry.get("supports_native_tool_calling", True)
                configured_capabilities = route_entry.get("capabilities", {})
                resolved_provider = _infer_provider_name(actual_model_string)
            else:
                actual_model_string = alias
                desc = f"Custom Model: {alias}"
                inferred_provider = _infer_provider_name(actual_model_string)
                env_keys = _provider_env_keys(inferred_provider)
                default_kwargs = _provider_default_kwargs(inferred_provider)
                registry_max_context_tokens = 128000
                supports_native_tool_calling = _provider_supports_native_tool_calling(inferred_provider)
                configured_capabilities = _provider_model_capabilities(inferred_provider)
                resolved_provider = inferred_provider
    else:
        if model_name in MODEL_REGISTRY:
            config = MODEL_REGISTRY[model_name]
            actual_model_string = config["model"]
            desc = config["description"]
            env_keys = config.get("env_key")
            default_kwargs = config.get("default_kwargs", {})
            registry_max_context_tokens = config.get("max_context_tokens", 128000)
            supports_native_tool_calling = config.get("supports_native_tool_calling", True)
            configured_capabilities = config.get("capabilities", {})
            resolved_provider = _infer_provider_name(actual_model_string)
        else:
            normalized_provider = _normalize_provider_name(provider)
            actual_model_string = f"{normalized_provider}/{model_name}"
            desc = f"Custom Model: {actual_model_string}"
            env_keys = _provider_env_keys(normalized_provider)
            default_kwargs = _provider_default_kwargs(normalized_provider)
            registry_max_context_tokens = 128000
            supports_native_tool_calling = _provider_supports_native_tool_calling(normalized_provider)
            configured_capabilities = _provider_model_capabilities(normalized_provider)
            resolved_provider = normalized_provider

    final_api_key = api_key
    if not final_api_key and env_keys:
        if isinstance(env_keys, (list, tuple)):
            for key in env_keys:
                final_api_key = os.environ.get(str(key).strip())
                if final_api_key:
                    break
        else:
            final_api_key = os.environ.get(str(env_keys).strip())

    if not final_api_key:
        env_hint = _format_env_keys(env_keys)
        raise ValueError(
            f"Missing API key for {desc}.\n"
            "Pass `api_key` explicitly, for example:\n"
            "setup_model('provider', 'model_name', 'sk-xxx')\n"
            f"Or set environment variable(s): {env_hint}"
        )

    final_kwargs = _merge_model_kwargs(default_kwargs, kwargs)
    runtime_capabilities = final_kwargs.pop("model_capabilities", None)
    if not isinstance(runtime_capabilities, dict):
        runtime_capabilities = {}

    model_capabilities = _resolve_model_capabilities(
        model_string=actual_model_string,
        provider_name=resolved_provider,
        supports_native_tool_calling=bool(supports_native_tool_calling),
        configured_capabilities=configured_capabilities if isinstance(configured_capabilities, dict) else {},
        runtime_capabilities=runtime_capabilities,
    )

    max_tokens = final_kwargs.pop("max_context_tokens", registry_max_context_tokens)
    api_base = final_kwargs.pop("api_base", None)
    temperature = final_kwargs.pop("temperature", 0.1)
    top_p = final_kwargs.pop("top_p", None)
    supports_native_tool_calling = bool(
        final_kwargs.pop(
            "supports_native_tool_calling",
            model_capabilities.get("supports_native_tool_calling", supports_native_tool_calling),
        )
    )
    model_capabilities["supports_native_tool_calling"] = supports_native_tool_calling
    model_backend = str(
        final_kwargs.pop("model_backend", final_kwargs.pop("backend", "litellm")) or "litellm"
    ).strip().lower()

    print(f"[Model] {desc} (liteLLM route: {actual_model_string}, backend: {model_backend})")

    model = LiteLLMChat(
        model=actual_model_string,
        api_key=final_api_key,
        api_base=api_base,
        temperature=temperature,
        top_p=top_p,
        backend=model_backend,
        model_capabilities=model_capabilities,
        **final_kwargs,
    )
    model.max_context_tokens = max_tokens
    model.supports_native_tool_calling = supports_native_tool_calling
    return model


def list_models() -> None:
    """Print all registered aliases and API key readiness."""
    print("\nRegistered model aliases:")
    print("-" * 75)
    for name, config in MODEL_REGISTRY.items():
        env_keys = config.get("env_key", "")
        if isinstance(env_keys, (list, tuple)):
            env_list = [str(item).strip() for item in env_keys if str(item or "").strip()]
            env_str = " / ".join(env_list)
            has_key = any(bool(os.getenv(item)) for item in env_list)
        else:
            env_str = str(env_keys or "N/A")
            has_key = bool(os.getenv(env_str)) if env_str != "N/A" else False

        status = "OK" if has_key else "MISSING"
        print(f"  [{status:7s}] {name:20s} | {config['description']:30s} | Key: {env_str}")
    print("-" * 75)
    print('Tip: you can also call setup("provider", "model_name") for models not listed here.')
