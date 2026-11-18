import os
import uuid
import sqlite3
import hashlib
import hmac
import time
import json
from datetime import datetime, timezone

import requests
from flask import (
    Flask,
    request,
    jsonify,
    make_response,
    render_template_string,
)
from flask_cors import CORS


# ============================================================
# VEYA V18
# Multi-Tier / Multi-Provider AI Platform
# ============================================================

APP_NAME = "Veya"
VERSION = "18.0"

PORT = int(os.getenv("PORT", "10000"))
DB_PATH = os.getenv("DATABASE_PATH", "veya.db")

SECRET_KEY = (
    os.getenv("SECRET_KEY", "").strip()
    or os.urandom(32).hex()
)

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY

CORS(app)


# ============================================================
# HELPERS
# ============================================================

def env(name, default=""):
    return os.getenv(name, default).strip()


def env_list(name):
    value = env(name)

    if not value:
        return []

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# PROVIDERS
# ============================================================

PROVIDERS = [
    "openai",
    "anthropic",
    "google",
    "xai",
    "deepseek",
    "mistral",
    "nvidia",
    "microsoft",
    "meta",
    "groq",
]


API_KEYS = {
    "openai": env("OPENAI_API_KEY"),
    "anthropic": env("ANTHROPIC_API_KEY"),
    "google": env("GOOGLE_API_KEY"),
    "xai": env("XAI_API_KEY"),
    "deepseek": env("DEEPSEEK_API_KEY"),
    "mistral": env("MISTRAL_API_KEY"),
    "nvidia": env("NVIDIA_API_KEY"),
    "microsoft": env("MICROSOFT_API_KEY"),
    "meta": env("META_API_KEY"),
    "groq": env("GROQ_API_KEY"),
}


BASE_URLS = {
    "openai": env(
        "OPENAI_BASE_URL",
        "https://api.openai.com/v1",
    ),

    "anthropic": env(
        "ANTHROPIC_BASE_URL",
        "https://api.anthropic.com",
    ),

    "google": env(
        "GOOGLE_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta",
    ),

    "xai": env(
        "XAI_BASE_URL",
        "https://api.x.ai/v1",
    ),

    "deepseek": env(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
    ),

    "mistral": env(
        "MISTRAL_BASE_URL",
        "https://api.mistral.ai/v1",
    ),

    "nvidia": env(
        "NVIDIA_BASE_URL",
        "https://integrate.api.nvidia.com/v1",
    ),

    "microsoft": env(
        "MICROSOFT_BASE_URL",
    ),

    "meta": env(
        "META_BASE_URL",
    ),

    "groq": env(
        "GROQ_BASE_URL",
        "https://api.groq.com/openai/v1",
    ),
}


# ============================================================
# CURRENT FRONTIER MODEL PREFERENCES
#
# These are preferences, NOT blind assumptions.
# Veya still verifies them against the live provider catalogue.
# ============================================================

FRONTIER_PREFERENCES = {

    "openai": [
        "gpt-6.2-astra",
        "gpt-6.2-sol",
        "gpt-6.2-luna",
        "gpt-6-astra",
        "gpt-6-sol",
        "gpt-6-luna",
    ],

    "anthropic": [
        "claude-fable-5-2",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
        "claude-fable-5-1",
    ],

    "google": [
        "gemini-4-argon",
        "gemini-4-carbon",
        "gemini-4-hydrogen",
        "gemini-3.8-pro",
        "gemini-3.8-flash",
        "gemini-3.1-pro",
        "gemini-3.1-flash",
        "gemini-3.1-flash-lite",
    ],

    "xai": [
        "grok-4.7",
        "grok-4.7-fast",
        "grok-4",
        "grok-4-fast",
    ],

    "deepseek": [
        "deepseek-v4-pro",
        "deepseek-flash",
    ],

    "mistral": [
        "mistral-large-4",
        "mistral-medium-3.5",
        "mistral-small-4",
        "codestral",
        "devstral",
    ],

    "nvidia": [
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    ],

    "microsoft": [
        "gpt-6.1-sol",
        "gpt-6.1-terra",
        "gpt-6-astra",
        "gpt-6-sol",
        "gpt-6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
    ],

    "meta": [
        "muse-spark",
        "Llama-4-Maverick",
        "Llama-4-Scout",
        "llama-4-maverick",
        "llama-4-scout",
    ],

    "groq": [
        "openai/gpt-oss-120b",
        "minimaxai/minimax-m2.7",
        "qwen/qwen3.8-27b",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
    ],
}


# ============================================================
# CHEAP / FREE-TIER PREFERENCES
# ============================================================

FREE_PROVIDER_ORDER = [
    "deepseek",
    "groq",
    "mistral",
    "google",
    "nvidia",
]


FREE_MODEL_PREFERENCES = {

    "deepseek": [
        "deepseek-flash",
    ],

    "groq": [
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
    ],

    "mistral": [
        "mistral-small-4",
    ],

    "google": [
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.1-flash",
    ],

    "nvidia": [
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "nvidia/nemotron-3.5-super-vl-120b-a12b",
    ],
}


# ============================================================
# MODEL OVERRIDES
# ============================================================

MODEL_OVERRIDES = {
    provider: env_list(
        f"{provider.upper()}_MODELS"
    )
    for provider in PROVIDERS
}


# ============================================================
# MODEL CACHE
# ============================================================

MODEL_CACHE = {}

CACHE_SECONDS = int(
    env(
        "MODEL_CACHE_SECONDS",
        "300",
    )
)


def cache_get(provider):
    item = MODEL_CACHE.get(provider)

    if not item:
        return None

    timestamp, models = item

    if time.time() - timestamp >= CACHE_SECONDS:
        return None

    return models


def cache_set(provider, models):
    MODEL_CACHE[provider] = (
        time.time(),
        models,
    )


def cache_clear(provider=None):
    if provider:
        MODEL_CACHE.pop(provider, None)
    else:
        MODEL_CACHE.clear()


# ============================================================
# DATABASE
# ============================================================

def db():
    connection = sqlite3.connect(
        DB_PATH,
        timeout=30,
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_db():

    connection = db()

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            plan TEXT NOT NULL DEFAULT 'free',
            created_at TEXT NOT NULL,
            subscription_id TEXT,
            subscription_status TEXT
        );

        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            agent_role TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            agent_role TEXT,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            estimated_cost_usd REAL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS payment_events (
            event_id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )

    connection.commit()
    connection.close()


# ============================================================
# PLANS
# ============================================================

PLANS = {

    "free": {
        "name": "Veya Free",
        "price_inr": 0,
        "daily_messages": 20,
        "monthly_provider_budget_usd": 0.10,

        "providers": FREE_PROVIDER_ORDER,

        "model_count_per_provider": 1,

        "tier": "free",

        "model_picker": False,

        "verification": False,
    },

    "core": {
        "name": "Veya Core",
        "price_inr": 399,
        "daily_messages": 100,
        "monthly_provider_budget_usd": 2.50,

        "providers": [
            "openai",
            "anthropic",
            "google",
        ],

        "model_count_per_provider": 1,

        "tier": "frontier_3",

        "model_picker": True,

        "verification": False,
    },

    "all_ai": {
        "name": "Veya All AI",
        "price_inr": 999,
        "daily_messages": 300,
        "monthly_provider_budget_usd": 7.50,

        "providers": PROVIDERS,

        "model_count_per_provider": 1,

        "tier": "frontier_10",

        "model_picker": True,

        "verification": True,
    },

    "ultra": {
        "name": "Veya Ultra",
        "price_inr": 2499,
        "daily_messages": 1000,
        "monthly_provider_budget_usd": 25.00,

        "providers": PROVIDERS,

        "model_count_per_provider": 5,

        "tier": "frontier_50",

        "model_picker": True,

        "verification": True,
    },
}


# ============================================================
# HTTP
# ============================================================

def request_json(
    method,
    url,
    headers=None,
    params=None,
    payload=None,
    timeout=60,
):

    response = requests.request(
        method,
        url,
        headers=headers or {},
        params=params or {},
        json=payload,
        timeout=timeout,
    )

    if not response.ok:

        raise RuntimeError(
            f"HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    if not response.text.strip():
        return {}

    return response.json()


# ============================================================
# PROVIDER CONFIG
# ============================================================

def provider_configured(provider):

    if provider == "local":
        return bool(
            env("LOCAL_BASE_URL")
        )

    if not API_KEYS.get(provider):
        return False

    if provider in (
        "microsoft",
        "meta",
    ):
        return bool(
            BASE_URLS.get(provider)
        )

    return True


def provider_headers(provider):

    key = API_KEYS.get(
        provider,
        "",
    )

    if provider == "anthropic":

        return {
            "x-api-key": key,
            "anthropic-version":
                "2023-06-01",
            "Content-Type":
                "application/json",
        }

    if provider == "microsoft":

        return {
            "api-key": key,
            "Content-Type":
                "application/json",
        }

    return {
        "Authorization":
            f"Bearer {key}",
        "Content-Type":
            "application/json",
    }


# ============================================================
# MODEL FILTER
# ============================================================

NON_CHAT_TERMS = (
    "embedding",
    "embed",
    "rerank",
    "moderation",
    "whisper",
    "tts",
    "transcrib",
    "speech-only",
    "audio-only",
    "image-generation",
    "ocr",
)


def looks_like_chat_model(model):

    name = model.lower()

    return not any(
        term in name
        for term in NON_CHAT_TERMS
    )


# ============================================================
# GOOGLE DISCOVERY
# ============================================================

def discover_google_models():

    provider = "google"

    cached = cache_get(provider)

    if cached is not None:
        return cached

    if not provider_configured(provider):
        return []

    data = request_json(
        "GET",
        BASE_URLS[provider].rstrip("/")
        + "/models",
        params={
            "key":
                API_KEYS["google"]
        },
        timeout=30,
    )

    models = []

    for item in data.get(
        "models",
        [],
    ):

        name = item.get(
            "name",
            "",
        )

        if name.startswith("models/"):
            name = name[7:]

        methods = item.get(
            "supportedGenerationMethods",
            [],
        )

        if not name:
            continue

        if (
            methods
            and
            "generateContent"
            not in methods
        ):
            continue

        if not looks_like_chat_model(name):
            continue

        models.append(name)

    models = list(
        dict.fromkeys(models)
    )

    cache_set(
        provider,
        models,
    )

    return models


# ============================================================
# ANTHROPIC DISCOVERY
# ============================================================

def discover_anthropic_models():

    provider = "anthropic"

    cached = cache_get(provider)

    if cached is not None:
        return cached

    if not provider_configured(provider):
        return []

    data = request_json(
        "GET",
        BASE_URLS[provider].rstrip("/")
        + "/v1/models",
        headers=provider_headers(provider),
        timeout=30,
    )

    models = []

    for item in data.get(
        "data",
        [],
    ):

        model_id = item.get("id")

        if (
            model_id
            and
            looks_like_chat_model(model_id)
        ):
            models.append(model_id)

    models = list(
        dict.fromkeys(models)
    )

    cache_set(
        provider,
        models,
    )

    return models


# ============================================================
# OPENAI-COMPATIBLE DISCOVERY
# ============================================================

def discover_compatible_models(provider):

    cached = cache_get(provider)

    if cached is not None:
        return cached

    if not provider_configured(provider):
        return []

    base = BASE_URLS.get(
        provider,
        "",
    ).rstrip("/")

    if not base:
        return []

    data = request_json(
        "GET",
        base + "/models",
        headers=provider_headers(provider),
        timeout=30,
    )

    models = []

    for item in data.get(
        "data",
        [],
    ):

        model_id = item.get("id")

        if (
            model_id
            and
            looks_like_chat_model(model_id)
        ):
            models.append(model_id)

    models = list(
        dict.fromkeys(models)
    )

    cache_set(
        provider,
        models,
    )

    return models


# ============================================================
# UNIVERSAL MODEL DISCOVERY
# ============================================================

def discover_models(provider):

    try:

        if provider == "google":

            live = discover_google_models()

        elif provider == "anthropic":

            live = discover_anthropic_models()

        else:

            live = discover_compatible_models(
                provider
            )

    except Exception:

        live = []

    override = MODEL_OVERRIDES.get(
        provider,
        [],
    )

    if live:

        if override:

            verified = [
                model
                for model in override
                if model in live
            ]

            if verified:
                return verified

        return live

    return override


# ============================================================
# MODEL RANKING
# ============================================================

def preference_score(
    provider,
    model,
):

    preferences = FRONTIER_PREFERENCES.get(
        provider,
        [],
    )

    try:

        index = preferences.index(model)

        return 10000 - index * 100

    except ValueError:

        return 0


def model_score(
    provider,
    model,
    task="general",
):

    name = model.lower()

    score = preference_score(
        provider,
        model,
    )

    frontier_terms = (
        "frontier",
        "flagship",
        "ultra",
        "pro",
        "opus",
        "fable",
        "argon",
        "astra",
        "large",
        "reason",
        "thinking",
        "max",
    )

    for term in frontier_terms:

        if term in name:
            score += 20

    if task == "coding":

        for term in (
            "code",
            "coder",
            "reason",
            "thinking",
            "opus",
            "fable",
            "astra",
            "pro",
        ):

            if term in name:
                score += 30

    elif task == "research":

        for term in (
            "research",
            "reason",
            "thinking",
            "fable",
            "opus",
            "pro",
            "large",
        ):

            if term in name:
                score += 30

    elif task == "math":

        for term in (
            "reason",
            "thinking",
            "pro",
            "large",
            "fable",
            "astra",
        ):

            if term in name:
                score += 30

    return score


def rank_models(
    provider,
    models,
    task="general",
):

    return sorted(
        models,
        key=lambda model: (
            model_score(
                provider,
                model,
                task,
            ),
            len(model),
        ),
        reverse=True,
    )


# ============================================================
# PLAN MODEL CATALOG
# ============================================================

def top_models_for_provider(
    provider,
    count,
    task="general",
):

    live = discover_models(provider)

    if not live:
        return []

    ranked = rank_models(
        provider,
        live,
        task,
    )

    return ranked[:count]


def free_model_for_provider(provider):

    live = discover_models(provider)

    if not live:
        return None

    preferred = FREE_MODEL_PREFERENCES.get(
        provider,
        [],
    )

    for wanted in preferred:

        if wanted in live:
            return wanted

    return rank_models(
        provider,
        live,
        "general",
    )[-1]


def build_plan_catalog(
    plan,
    task="general",
):

    if plan not in PLANS:
        return []

    output = []

    plan_info = PLANS[plan]

    for provider in plan_info["providers"]:

        if not provider_configured(provider):
            continue

        if plan == "free":

            model = free_model_for_provider(
                provider
            )

            if model:

                output.append({
                    "provider":
                        provider,

                    "slot":
                        1,

                    "model":
                        model,

                    "unique_model":
                        True,

                    "tier":
                        "economical",
                })

            continue

        models = top_models_for_provider(
            provider,
            plan_info[
                "model_count_per_provider"
            ],
            task,
        )

        for index, model in enumerate(models):

            output.append({

                "provider":
                    provider,

                "slot":
                    index + 1,

                "model":
                    model,

                "unique_model":
                    True,

                "tier":
                    plan_info["tier"],
            })

    return output


# ============================================================
# PREMIUM AGENT ROLES
# ============================================================

PREMIUM_ROLES = [
    (
        "reasoning",
        "Deep Reasoning",
    ),

    (
        "coding",
        "Advanced Coding",
    ),

    (
        "research",
        "Research & Analysis",
    ),

    (
        "creative",
        "Creative Intelligence",
    ),

    (
        "verifier",
        "Verification",
    ),
]


def catalog_with_roles(
    plan,
    task="general",
):

    catalog = build_plan_catalog(
        plan,
        task,
    )

    result = []

    for item in catalog:

        slot = item["slot"]

        role_id, role_name = (
            PREMIUM_ROLES[
                min(
                    slot - 1,
                    len(PREMIUM_ROLES) - 1,
                )
            ]
        )

        result.append({

            **item,

            "agent_id":
                f'{item["provider"]}-'
                f'{role_id}-'
                f'{slot}',

            "agent_role":
                role_id,

            "agent_name":
                role_name,
        })

    return result


# ============================================================
# TASK ROUTER
# ============================================================

def route_task(text):

    value = text.lower()

    if any(
        word in value
        for word in (
            "python",
            "javascript",
            "typescript",
            "coding",
            "code",
            "debug",
            "docker",
            "api",
            "program",
            "github",
            "software",
        )
    ):
        return "coding"

    if any(
        word in value
        for word in (
            "calculate",
            "equation",
            "math",
            "mathematics",
            "proof",
            "algebra",
        )
    ):
        return "math"

    if any(
        word in value
        for word in (
            "research",
            "latest",
            "current",
            "today",
            "compare",
            "analysis",
        )
    ):
        return "research"

    if any(
        word in value
        for word in (
            "write",
            "story",
            "article",
            "email",
            "essay",
            "rewrite",
        )
    ):
        return "writing"

    return "general"


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """
You are Veya.

Veya is an orchestration platform over configured AI providers.

Do not claim to literally be OpenAI, Anthropic, Google,
xAI, DeepSeek, Mistral, NVIDIA, Microsoft, Meta or Groq.

Be accurate.

Virtual agents are orchestration roles.
They are not independent foundation models.

If a model/provider is unavailable, do not pretend that
it is working.

Do not invent sources, capabilities or tool usage.
"""


def role_instruction(role):

    instructions = {

        "reasoning": """
Reason carefully.
Separate facts from assumptions.
Check important conclusions.
""",

        "coding": """
Act as a senior software engineer.
Check syntax, architecture, dependencies,
security and edge cases.
""",

        "research": """
Act as a research and analysis specialist.
Do not invent sources or unsupported facts.
""",

        "creative": """
Act as a creative specialist while remaining accurate.
""",

        "verifier": """
Check the answer for contradictions,
unsupported claims and technical errors.
""",
    }

    return instructions.get(
        role,
        instructions["reasoning"],
    )


# ============================================================
# PROVIDER CALLS
# ============================================================

def call_openai(
    model,
    messages,
):

    data = request_json(
        "POST",
        BASE_URLS["openai"].rstrip("/")
        + "/responses",
        headers=provider_headers("openai"),
        payload={
            "model":
                model,

            "input":
                messages,
        },
        timeout=180,
    )

    answer = data.get(
        "output_text",
        "",
    )

    if not answer:

        parts = []

        for item in data.get(
            "output",
            [],
        ):

            for content in item.get(
                "content",
                [],
            ):

                if (
                    content.get("type")
                    == "output_text"
                ):

                    parts.append(
                        content.get(
                            "text",
                            "",
                        )
                    )

        answer = "\n".join(parts).strip()

    usage = data.get(
        "usage",
        {},
    )

    return (
        answer
        or "The model returned no text.",
        int(
            usage.get(
                "input_tokens",
                0,
            )
        ),
        int(
            usage.get(
                "output_tokens",
                0,
            )
        ),
    )


def call_anthropic(
    model,
    messages,
):

    system = []
    clean = []

    for message in messages:

        if message["role"] == "system":

            system.append(
                message["content"]
            )

        else:

            clean.append({
                "role":
                    message["role"],

                "content":
                    message["content"],
            })

    payload = {

        "model":
            model,

        "max_tokens":
            4096,

        "messages":
            clean,
    }

    if system:

        payload["system"] = "\n".join(
            system
        )

    data = request_json(
        "POST",
        BASE_URLS["anthropic"].rstrip("/")
        + "/v1/messages",
        headers=provider_headers("anthropic"),
        payload=payload,
        timeout=180,
    )

    parts = []

    for block in data.get(
        "content",
        [],
    ):

        if block.get("type") == "text":

            parts.append(
                block.get(
                    "text",
                    "",
                )
            )

    usage = data.get(
        "usage",
        {},
    )

    return (
        "\n".join(parts).strip()
        or "The model returned no text.",

        int(
            usage.get(
                "input_tokens",
                0,
            )
        ),

        int(
            usage.get(
                "output_tokens",
                0,
            )
        ),
    )


def call_google(
    model,
    messages,
):

    contents = []
    system_parts = []

    for message in messages:

        if message["role"] == "system":

            system_parts.append(
                message["content"]
            )

            continue

        role = message["role"]

        if role == "assistant":
            role = "model"

        contents.append({

            "role":
                role,

            "parts": [
                {
                    "text":
                        message["content"]
                }
            ],
        })

    payload = {

        "contents":
            contents,

        "generationConfig": {

            "maxOutputTokens":
                4096,
        },
    }

    if system_parts:

        payload["systemInstruction"] = {

            "parts": [

                {
                    "text":
                        "\n".join(
                            system_parts
                        )
                }
            ]
        }

    data = request_json(
        "POST",
        BASE_URLS["google"].rstrip("/")
        + "/models/"
        + model
        + ":generateContent",

        params={
            "key":
                API_KEYS["google"]
        },

        headers={
            "Content-Type":
                "application/json"
        },

        payload=payload,

        timeout=180,
    )

    parts = []

    for candidate in data.get(
        "candidates",
        [],
    ):

        content = candidate.get(
            "content",
            {},
        )

        for part in content.get(
            "parts",
            [],
        ):

            if part.get("text"):

                parts.append(
                    part["text"]
                )

    usage = data.get(
        "usageMetadata",
        {},
    )

    return (
        "\n".join(parts).strip()
        or "The model returned no text.",

        int(
            usage.get(
                "promptTokenCount",
                0,
            )
        ),

        int(
            usage.get(
                "candidatesTokenCount",
                0,
            )
        ),
    )


def call_compatible(
    provider,
    model,
    messages,
):

    base = BASE_URLS.get(
        provider,
        "",
    ).rstrip("/")

    if not base:

        raise RuntimeError(
            f"{provider} base URL is not configured."
        )

    payload = {

        "model":
            model,

        "messages":
            messages,

        "max_tokens":
            4096,
    }

    # NVIDIA reasoning configuration.
    if provider == "nvidia":

        payload["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": True
            }
        }

    data = request_json(
        "POST",
        base + "/chat/completions",
        headers=provider_headers(provider),
        payload=payload,
        timeout=180,
    )

    choices = data.get(
        "choices",
        [],
    )

    if not choices:

        raise RuntimeError(
            f"{provider} returned no choices."
        )

    content = choices[0].get(
        "message",
        {},
    ).get(
        "content",
        "",
    )

    if isinstance(content, list):

        content = "\n".join(

            item.get(
                "text",
                "",
            )

            for item in content

            if isinstance(
                item,
                dict,
            )
        )

    usage = data.get(
        "usage",
        {},
    )

    return (

        content
        or "The model returned no text.",

        int(
            usage.get(
                "prompt_tokens",
                0,
            )
        ),

        int(
            usage.get(
                "completion_tokens",
                0,
            )
        ),
    )


def call_provider(
    provider,
    model,
    messages,
):

    if provider == "openai":

        return call_openai(
            model,
            messages,
        )

    if provider == "anthropic":

        return call_anthropic(
            model,
            messages,
        )

    if provider == "google":

        return call_google(
            model,
            messages,
        )

    return call_compatible(
        provider,
        model,
        messages,
    )


# ============================================================
# USERS
# ============================================================

def get_user_id():

    user_id = request.cookies.get(
        "veya_user"
    )

    if not user_id:

        user_id = str(
            uuid.uuid4()
        )

    connection = db()

    row = connection.execute(
        """
        SELECT id
        FROM users
        WHERE id = ?
        """,
        (user_id,),
    ).fetchone()

    if not row:

        connection.execute(
            """
            INSERT INTO users(
                id,
                plan,
                created_at
            )
            VALUES (?, 'free', ?)
            """,
            (
                user_id,
                utc_now(),
            ),
        )

        connection.commit()

    connection.close()

    return user_id


def get_plan(user_id):

    connection = db()

    row = connection.execute(
        """
        SELECT plan
        FROM users
        WHERE id = ?
        """,
        (user_id,),
    ).fetchone()

    connection.close()

    if not row:
        return "free"

    if row["plan"] not in PLANS:
        return "free"

    return row["plan"]


# ============================================================
# CHAT DATABASE
# ============================================================

def create_chat(
    user_id,
    title="New chat",
):

    chat_id = str(
        uuid.uuid4()
    )

    timestamp = utc_now()

    connection = db()

    connection.execute(
        """
        INSERT INTO chats(
            id,
            user_id,
            title,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            chat_id,
            user_id,
            title,
            timestamp,
            timestamp,
        ),
    )

    connection.commit()
    connection.close()

    return chat_id


def ensure_chat(
    user_id,
    chat_id,
):

    connection = db()

    row = connection.execute(
        """
        SELECT id
        FROM chats
        WHERE id = ?
        AND user_id = ?
        """,
        (
            chat_id,
            user_id,
        ),
    ).fetchone()

    connection.close()

    if not row:

        raise RuntimeError(
            "Chat not found."
        )


def save_message(
    chat_id,
    role,
    content,
    provider=None,
    model=None,
    agent_role=None,
):

    timestamp = utc_now()

    connection = db()

    connection.execute(
        """
        INSERT INTO messages(
            chat_id,
            role,
            content,
            provider,
            model,
            agent_role,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chat_id,
            role,
            content,
            provider,
            model,
            agent_role,
            timestamp,
        ),
    )

    connection.execute(
        """
        UPDATE chats
        SET updated_at = ?
        WHERE id = ?
        """,
        (
            timestamp,
            chat_id,
        ),
    )

    connection.commit()
    connection.close()


def get_messages(
    chat_id,
    limit=30,
):

    connection = db()

    rows = connection.execute(
        """
        SELECT
            role,
            content,
            provider,
            model,
            agent_role,
            created_at
        FROM messages
        WHERE chat_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (
            chat_id,
            limit,
        ),
    ).fetchall()

    connection.close()

    return [
        dict(row)
        for row in reversed(rows)
    ]


# ============================================================
# USAGE
# ============================================================

def daily_message_count(user_id):

    today = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d")

    connection = db()

    row = connection.execute(
        """
        SELECT COUNT(*)
        FROM messages m
        JOIN chats c
        ON c.id = m.chat_id
        WHERE c.user_id = ?
        AND m.role = 'user'
        AND substr(
            m.created_at,
            1,
            10
        ) = ?
        """,
        (
            user_id,
            today,
        ),
    ).fetchone()

    connection.close()

    return int(row[0] or 0)


def monthly_cost(user_id):

    month = datetime.now(
        timezone.utc
    ).strftime("%Y-%m")

    connection = db()

    row = connection.execute(
        """
        SELECT COALESCE(
            SUM(estimated_cost_usd),
            0
        )
        FROM usage
        WHERE user_id = ?
        AND substr(
            created_at,
            1,
            7
        ) = ?
        """,
        (
            user_id,
            month,
        ),
    ).fetchone()

    connection.close()

    return float(row[0] or 0)


def estimate_cost(
    input_tokens,
    output_tokens,
):

    input_price = float(
        env(
            "DEFAULT_INPUT_COST_USD_PER_M",
            "2",
        )
    )

    output_price = float(
        env(
            "DEFAULT_OUTPUT_COST_USD_PER_M",
            "8",
        )
    )

    return (
        input_tokens
        / 1_000_000
        * input_price
        +
        output_tokens
        / 1_000_000
        * output_price
    )


def record_usage(
    user_id,
    provider,
    model,
    agent_role,
    input_tokens,
    output_tokens,
):

    cost = estimate_cost(
        input_tokens,
        output_tokens,
    )

    connection = db()

    connection.execute(
        """
        INSERT INTO usage(
            user_id,
            provider,
            model,
            agent_role,
            input_tokens,
            output_tokens,
            estimated_cost_usd,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            provider,
            model,
            agent_role,
            input_tokens,
            output_tokens,
            cost,
            utc_now(),
        ),
    )

    connection.commit()
    connection.close()

    return cost


# ============================================================
# 5000 VIRTUAL AGENTS
# ============================================================

BASE_ROLES = [
    "reasoning",
    "coding",
    "research",
    "mathematics",
    "science",
    "writing",
    "editing",
    "planning",
    "analysis",
    "fact_checking",
    "criticism",
    "summarization",
    "translation",
    "teaching",
    "product_design",
    "business_analysis",
    "security_review",
    "data_analysis",
    "architecture",
    "strategy",
]


VIRTUAL_AGENTS = []

for index in range(5000):

    role = BASE_ROLES[
        index % len(BASE_ROLES)
    ]

    number = (
        index
        // len(BASE_ROLES)
        + 1
    )

    VIRTUAL_AGENTS.append({

        "id":
            f"veya-agent-{index + 1:04d}",

        "role":
            f"{role}-{number}",

        "index":
            index + 1,
    })


# ============================================================
# MODEL SELECTION
# ============================================================

def choose_model(
    plan,
    task,
    requested_provider=None,
    requested_model=None,
):

    catalog = catalog_with_roles(
        plan,
        task,
    )

    if not catalog:

        raise RuntimeError(
            "No verified live model is available "
            "for this plan."
        )

    if (
        requested_provider
        and requested_model
    ):

        for item in catalog:

            if (
                item["provider"]
                == requested_provider
                and
                item["model"]
                == requested_model
            ):

                return item

        raise RuntimeError(
            "The selected model is not currently "
            "available for this plan."
        )

    return max(
        catalog,
        key=lambda item:
            model_score(
                item["provider"],
                item["model"],
                task,
            ),
    )


# ============================================================
# API: HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return jsonify({

        "ok":
            True,

        "app":
            APP_NAME,

        "version":
            VERSION,

        "time":
            utc_now(),
    })


# ============================================================
# API: ME
# ============================================================

@app.get("/api/me")
def me():

    user_id = get_user_id()
    plan = get_plan(user_id)

    response = jsonify({

        "user_id":
            user_id,

        "plan":
            plan,

        "plan_info":
            PLANS[plan],

        "daily_used":
            daily_message_count(user_id),

        "monthly_cost_usd":
            monthly_cost(user_id),
    })

    response.set_cookie(
        "veya_user",
        user_id,
        httponly=True,
        samesite="Lax",
    )

    return response


# ============================================================
# API: PLANS
# ============================================================

@app.get("/api/plans")
def plans():

    return jsonify(PLANS)


# ============================================================
# API: PROVIDERS
# ============================================================

@app.get("/api/providers")
def providers():

    result = []

    for provider in PROVIDERS:

        configured = provider_configured(
            provider
        )

        models = []

        if configured:

            try:

                models = discover_models(
                    provider
                )

            except Exception:

                models = []

        result.append({

            "provider":
                provider,

            "configured":
                configured,

            "model_count":
                len(models),

            "models":
                models,

            "frontier_preferences":
                FRONTIER_PREFERENCES.get(
                    provider,
                    [],
                ),
        })

    return jsonify(result)


# ============================================================
# API: MODELS
# ============================================================

@app.get("/api/models")
def models():

    user_id = get_user_id()
    plan = get_plan(user_id)

    task = request.args.get(
        "task",
        "general",
    )

    return jsonify({

        "plan":
            plan,

        "models":
            catalog_with_roles(
                plan,
                task,
            ),

        "providers":
            PLANS[plan]["providers"],

        "models_per_provider":
            PLANS[plan][
                "model_count_per_provider"
            ],
    })


# ============================================================
# API: AGENTS
# ============================================================

@app.get("/api/agents")
def agents():

    return jsonify({

        "count":
            5000,

        "agents":
            VIRTUAL_AGENTS[:100],

        "premium_roles": [

            {
                "id":
                    role_id,

                "name":
                    role_name,
            }

            for role_id, role_name
            in PREMIUM_ROLES
        ],

        "note":
            "5000 are virtual orchestration roles, "
            "not 5000 independent foundation models.",
    })


# ============================================================
# API: CHATS
# ============================================================

@app.get("/api/chats")
def chats():

    user_id = get_user_id()

    connection = db()

    rows = connection.execute(
        """
        SELECT
            id,
            title,
            created_at,
            updated_at
        FROM chats
        WHERE user_id = ?
        ORDER BY updated_at DESC
        """,
        (user_id,),
    ).fetchall()

    connection.close()

    return jsonify([
        dict(row)
        for row in rows
    ])


@app.post("/api/chats/new")
def new_chat():

    return jsonify({

        "chat_id":
            create_chat(
                get_user_id()
            )
    })


@app.get("/api/chats/<chat_id>")
def chat_detail(chat_id):

    user_id = get_user_id()

    ensure_chat(
        user_id,
        chat_id,
    )

    return jsonify({

        "chat_id":
            chat_id,

        "messages":
            get_messages(
                chat_id
            ),
    })


# ============================================================
# API: MAIN CHAT
# ============================================================

@app.post("/api/chat")
def chat():

    user_id = get_user_id()
    plan = get_plan(user_id)

    plan_data = PLANS[plan]

    body = (
        request.get_json(
            silent=True
        )
        or {}
    )

    text = str(
        body.get(
            "message",
            "",
        )
    ).strip()

    if not text:

        return jsonify({
            "error":
                "Message is required."
        }), 400

    if len(text) > 30000:

        return jsonify({
            "error":
                "Message is too long."
        }), 400

    if (
        daily_message_count(user_id)
        >=
        plan_data["daily_messages"]
    ):

        return jsonify({
            "error":
                "Daily message limit reached."
        }), 429

    chat_id = body.get(
        "chat_id"
    )

    if not chat_id:

        chat_id = create_chat(
            user_id,
            text[:60],
        )

    ensure_chat(
        user_id,
        chat_id,
    )

    task = route_task(text)

    requested_provider = body.get(
        "provider"
    )

    requested_model = body.get(
        "model"
    )


    # ========================================================
    # FREE
    # ========================================================

    if plan == "free":

        provider = None
        model = None

        for candidate_provider in FREE_PROVIDER_ORDER:

            if not provider_configured(
                candidate_provider
            ):
                continue

            candidate_model = (
                free_model_for_provider(
                    candidate_provider
                )
            )

            if candidate_model:

                provider = candidate_provider
                model = candidate_model

                break

        if not provider or not model:

            return jsonify({

                "error":
                    "No economical/free-tier "
                    "provider is currently configured.",

                "hint":
                    "Configure at least one supported "
                    "provider API key.",
            }), 503

        agent_role = (
            "free-economical-router"
        )


    # ========================================================
    # PAID
    # ========================================================

    else:

        try:

            selected = choose_model(

                plan,

                task,

                requested_provider,

                requested_model,
            )

            provider = selected["provider"]
            model = selected["model"]
            agent_role = selected["agent_id"]

        except Exception as exc:

            return jsonify({
                "error":
                    str(exc)
            }), 503


    # ========================================================
    # BUDGET
    # ========================================================

    budget = plan_data[
        "monthly_provider_budget_usd"
    ]

    if (
        budget > 0
        and
        monthly_cost(user_id)
        >= budget
    ):

        return jsonify({

            "error":
                "Monthly AI usage budget reached.",

            "plan":
                plan,

            "budget_usd":
                budget,
        }), 402


    role_name = "reasoning"

    for role_id, _ in PREMIUM_ROLES:

        if role_id in str(agent_role):

            role_name = role_id

            break


    messages = [

        {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT
                + "\n\n"
                + role_instruction(
                    role_name
                ),
        }
    ]


    history = get_messages(
        chat_id,
        20,
    )

    for item in history:

        if item["role"] in (
            "user",
            "assistant",
        ):

            messages.append({

                "role":
                    item["role"],

                "content":
                    item["content"],
            })


    messages.append({

        "role":
            "user",

        "content":
            text,
    })


    # ========================================================
    # CALL
    # ========================================================

    try:

        (
            answer,
            input_tokens,
            output_tokens,
        ) = call_provider(

            provider,
            model,
            messages,
        )

    except Exception as first_error:

        cache_clear(provider)

        fresh_models = rank_models(

            provider,

            discover_models(
                provider
            ),

            task,
        )

        alternatives = [

            candidate

            for candidate in fresh_models

            if candidate != model
        ]

        success = False

        last_error = first_error

        for candidate in alternatives[:3]:

            try:

                (
                    answer,
                    input_tokens,
                    output_tokens,
                ) = call_provider(

                    provider,
                    candidate,
                    messages,
                )

                model = candidate

                success = True

                break

            except Exception as exc:

                last_error = exc

        if not success:

            return jsonify({

                "error":
                    f"{provider}/{model} failed.",

                "details":
                    str(last_error),

                "hint":
                    "Check /api/providers to "
                    "see the live models available "
                    "to your API key.",
            }), 502


    cost = record_usage(

        user_id,

        provider,

        model,

        agent_role,

        input_tokens,

        output_tokens,
    )


    save_message(
        chat_id,
        "user",
        text,
    )

    save_message(

        chat_id,

        "assistant",

        answer,

        provider,

        model,

        agent_role,
    )


    return jsonify({

        "chat_id":
            chat_id,

        "answer":
            answer,

        "provider":
            provider,

        "model":
            model,

        "agent":
            agent_role,

        "task":
            task,

        "plan":
            plan,

        "usage": {

            "input_tokens":
                input_tokens,

            "output_tokens":
                output_tokens,

            "estimated_cost_usd":
                cost,

            "monthly_cost_usd":
                monthly_cost(
                    user_id
                ),

            "monthly_budget_usd":
                budget,
        },
    })


# ============================================================
# MODEL HEALTH TEST
# ============================================================

@app.post("/api/test-model")
def test_model():

    provider = None
    model = None

    body = (
        request.get_json(
            silent=True
        )
        or {}
    )

    provider = body.get(
        "provider"
    )

    model = body.get(
        "model"
    )

    if not provider or not model:

        return jsonify({

            "ok":
                False,

            "error":
                "provider and model are required.",
        }), 400

    if model not in discover_models(provider):

        return jsonify({

            "ok":
                False,

            "error":
                "Model is not present in "
                "the provider's live catalogue.",
        }), 403

    try:

        (
            answer,
            input_tokens,
            output_tokens,
        ) = call_provider(

            provider,

            model,

            [
                {
                    "role":
                        "system",

                    "content":
                        "Reply exactly OK.",
                },

                {
                    "role":
                        "user",

                    "content":
                        "OK",
                },
            ],
        )

        return jsonify({

            "ok":
                True,

            "provider":
                provider,

            "model":
                model,

            "answer":
                answer,

            "input_tokens":
                input_tokens,

            "output_tokens":
                output_tokens,
        })

    except Exception as exc:

        return jsonify({

            "ok":
                False,

            "provider":
                provider,

            "model":
                model,

            "error":
                str(exc),
        }), 502


# ============================================================
# RAZORPAY
# ============================================================

RAZORPAY_KEY_ID = env(
    "RAZORPAY_KEY_ID"
)

RAZORPAY_KEY_SECRET = env(
    "RAZORPAY_KEY_SECRET"
)

RAZORPAY_WEBHOOK_SECRET = env(
    "RAZORPAY_WEBHOOK_SECRET"
)

RAZORPAY_PLAN_CORE = env(
    "RAZORPAY_PLAN_CORE"
)

RAZORPAY_PLAN_ALL_AI = env(
    "RAZORPAY_PLAN_ALL_AI"
)

RAZORPAY_PLAN_ULTRA = env(
    "RAZORPAY_PLAN_ULTRA"
)


RAZORPAY_PLAN_MAP = {

    "core":
        RAZORPAY_PLAN_CORE,

    "all_ai":
        RAZORPAY_PLAN_ALL_AI,

    "ultra":
        RAZORPAY_PLAN_ULTRA,
}


def razorpay_configured():

    return bool(

        RAZORPAY_KEY_ID
        and
        RAZORPAY_KEY_SECRET
    )


def set_user_plan(
    user_id,
    plan,
    subscription_id=None,
    subscription_status=None,
):

    if plan not in PLANS:
        return

    connection = db()

    connection.execute(

        """
        UPDATE users
        SET
            plan = ?,
            subscription_id = COALESCE(?, subscription_id),
            subscription_status = COALESCE(
                ?,
                subscription_status
            )
        WHERE id = ?
        """,

        (
            plan,
            subscription_id,
            subscription_status,
            user_id,
        ),
    )

    connection.commit()
    connection.close()


# ============================================================
# PAYMENT STATUS
# ============================================================

@app.get("/api/payment/status")
def payment_status():

    return jsonify({

        "razorpay_configured":
            razorpay_configured(),

        "razorpay_plans_configured":
            {
                plan:
                    bool(plan_id)
                for plan, plan_id
                in RAZORPAY_PLAN_MAP.items()
            },

        "credentials_exposed":
            False,
    })


# ============================================================
# CREATE RAZORPAY SUBSCRIPTION
# ============================================================

@app.post("/api/upgrade")
def upgrade():

    user_id = get_user_id()

    body = (
        request.get_json(
            silent=True
        )
        or {}
    )

    plan = body.get("plan")

    if plan not in (
        "core",
        "all_ai",
        "ultra",
    ):

        return jsonify({

            "error":
                "Invalid paid plan."
        }), 400


    if not razorpay_configured():

        return jsonify({

            "error":
                "Razorpay is not configured "
                "on the server."
        }), 503


    razorpay_plan_id = RAZORPAY_PLAN_MAP.get(
        plan
    )

    if not razorpay_plan_id:

        return jsonify({

            "error":
                f"RAZORPAY plan ID for {plan} "
                "is not configured."
        }), 503


    try:

        response = requests.post(

            "https://api.razorpay.com/v1/subscriptions",

            auth=(
                RAZORPAY_KEY_ID,
                RAZORPAY_KEY_SECRET,
            ),

            json={

                "plan_id":
                    razorpay_plan_id,

                "total_count":
                    120,

                "customer_notify":
                    1,
            },

            timeout=30,
        )

        if not response.ok:

            return jsonify({

                "error":
                    "Razorpay subscription creation failed.",

                "details":
                    response.text[:1000],
            }), 502


        data = response.json()


        return jsonify({

            "ok":
                True,

            "plan":
                plan,

            "subscription_id":
                data.get("id"),

            "razorpay_key_id":
                RAZORPAY_KEY_ID,

            "amount_inr":
                PLANS[plan]["price_inr"],

            "name":
                PLANS[plan]["name"],
        })


    except Exception as exc:

        return jsonify({

            "error":
                str(exc)
        }), 502


# ============================================================
# RAZORPAY WEBHOOK
# ============================================================

@app.post("/api/payment/razorpay/webhook")
def razorpay_webhook():

    if not RAZORPAY_WEBHOOK_SECRET:

        return jsonify({
            "error":
                "Webhook secret is not configured."
        }), 503


    raw_body = request.get_data()

    signature = request.headers.get(
        "X-Razorpay-Signature",
        "",
    )

    event_id = request.headers.get(
        "x-razorpay-event-id",
        "",
    )


    if not signature:

        return jsonify({
            "error":
                "Missing webhook signature."
        }), 400


    expected = hmac.new(

        RAZORPAY_WEBHOOK_SECRET.encode(),

        raw_body,

        hashlib.sha256,
    ).hexdigest()


    if not hmac.compare_digest(
        expected,
        signature,
    ):

        return jsonify({
            "error":
                "Invalid webhook signature."
        }), 400


    try:

        payload = json.loads(
            raw_body.decode(
                "utf-8"
            )
        )

    except Exception:

        return jsonify({
            "error":
                "Invalid JSON."
        }), 400


    event_type = payload.get(
        "event",
        "unknown",
    )


    if event_id:

        connection = db()

        existing = connection.execute(

            """
            SELECT event_id
            FROM payment_events
            WHERE event_id = ?
            """,

            (event_id,),
        ).fetchone()

        if existing:

            connection.close()

            return jsonify({
                "ok":
                    True,

                "duplicate":
                    True,
            })


        connection.execute(

            """
            INSERT INTO payment_events(
                event_id,
                event_type,
                payload,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,

            (
                event_id,
                event_type,
                raw_body.decode(
                    "utf-8"
                ),
                utc_now(),
            ),
        )

        connection.commit()
        connection.close()


    # --------------------------------------------------------
    # Subscription events
    # --------------------------------------------------------

    entity = (
        payload
        .get("payload", {})
        .get("subscription", {})
        .get("entity", {})
    )

    subscription_id = entity.get(
        "id"
    )

    notes = entity.get(
        "notes",
        {}
    )

    user_id = notes.get(
        "veya_user_id"
    )

    plan = notes.get(
        "veya_plan"
    )


    if (
        user_id
        and
        plan in (
            "core",
            "all_ai",
            "ultra",
        )
    ):

        if event_type in (
            "subscription.activated",
            "subscription.charged",
        ):

            set_user_plan(

                user_id,

                plan,

                subscription_id,

                "active",
            )


        elif event_type in (
            "subscription.halted",
            "subscription.cancelled",
            "subscription.completed",
        ):

            set_user_plan(

                user_id,

                "free",

                subscription_id,

                event_type,
            )


    return jsonify({
        "ok":
            True
    })


# ============================================================
# HTML
# ============================================================

HTML = """
<!doctype html>

<html>

<head>

<meta charset="utf-8">

<meta
name="viewport"
content="width=device-width,initial-scale=1">

<title>Veya</title>

<style>

*{
    box-sizing:border-box;
}

html,
body{
    margin:0;
    width:100%;
    height:100%;
    font-family:
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
    background:#f7f9fc;
    color:#111827;
}

button,
textarea,
select{
    font:inherit;
}

button{
    cursor:pointer;
}

#app{
    width:100%;
    height:100vh;
    display:flex;
    overflow:hidden;
}

#sidebar{
    position:fixed;
    left:0;
    top:0;
    bottom:0;
    width:300px;
    background:#fff;
    border-right:1px solid #e5e7eb;
    transform:translateX(-100%);
    transition:.25s;
    z-index:50;
    display:flex;
    flex-direction:column;
}

#sidebar.open{
    transform:translateX(0);
}

#overlay{
    display:none;
    position:fixed;
    inset:0;
    background:rgba(15,23,42,.35);
    z-index:40;
}

#overlay.show{
    display:block;
}

.sidebar-brand{
    padding:22px;
    font-size:24px;
    font-weight:850;
}

.new-chat{
    margin:0 16px 12px;
    width:calc(100% - 32px);
    border:1px solid #dbe3ef;
    border-radius:14px;
    padding:13px;
    background:#fff;
    font-weight:750;
}

.history{
    flex:1;
    overflow:auto;
    padding:10px;
}

.history-item{
    padding:12px;
    border-radius:11px;
    white-space:nowrap;
    overflow:hidden;
    text-overflow:ellipsis;
}

.history-item:hover{
    background:#eef2f7;
}

#main{
    width:100%;
    height:100vh;
    display:flex;
    flex-direction:column;
}

.topbar{
    height:70px;
    display:flex;
    align-items:center;
    gap:12px;
    padding:10px 18px;
    background:rgba(255,255,255,.96);
    border-bottom:1px solid #e5e7eb;
}

.menu{
    width:44px;
    height:44px;
    border:0;
    background:transparent;
    border-radius:12px;
    font-size:25px;
}

.brand{
    font-size:22px;
    font-weight:850;
}

.spacer{
    flex:1;
}

.upgrade{
    border:0;
    background:#111827;
    color:#fff;
    padding:9px 14px;
    border-radius:12px;
    font-weight:800;
}

.plan{
    font-size:12px;
    padding:7px 11px;
    border-radius:20px;
    background:#eef2ff;
    color:#4f46e5;
    font-weight:750;
}

.chat{
    flex:1;
    overflow:auto;
    padding:45px 18px 150px;
}

.welcome{
    max-width:850px;
    margin:8vh auto;
    text-align:center;
}

.welcome h1{
    font-size:
        clamp(38px,7vw,70px);
    line-height:1;
    letter-spacing:-3px;
}

.gradient{
    background:
        linear-gradient(
            90deg,
            #2563eb,
            #7c3aed,
            #db2777
        );
    -webkit-background-clip:text;
    color:transparent;
}

.welcome p{
    color:#64748b;
    font-size:17px;
}

.message{
    max-width:900px;
    margin:0 auto 24px;
    line-height:1.7;
    white-space:pre-wrap;
}

.user{
    max-width:720px;
    margin-left:auto;
    background:#eef2f7;
    border-radius:20px;
    padding:14px 17px;
}

.tag{
    color:#64748b;
    font-size:11px;
    font-weight:750;
    margin-bottom:7px;
}

.composer-wrap{
    position:fixed;
    left:0;
    right:0;
    bottom:0;
    padding:16px;
    background:
        linear-gradient(
            transparent,
            #f7f9fc 30%
        );
}

.composer{
    max-width:900px;
    margin:auto;
    background:#fff;
    border:1px solid #dbe2ec;
    border-radius:24px;
    padding:8px;
    box-shadow:
        0 15px 45px
        rgba(15,23,42,.10);
}

.row{
    display:flex;
    gap:8px;
    align-items:end;
}

textarea{
    flex:1;
    min-height:45px;
    max-height:150px;
    resize:none;
    border:0;
    outline:0;
    padding:12px 5px;
}

.send{
    width:44px;
    height:44px;
    border:0;
    border-radius:50%;
    background:#111827;
    color:#fff;
    font-size:19px;
}

.picker{
    display:none;
    padding:8px;
    border-bottom:1px solid #eef2f7;
}

.picker.show{
    display:block;
}

select{
    width:100%;
    border:1px solid #dbe2ec;
    border-radius:12px;
    padding:10px;
    background:#fff;
}

/* ============================================================
   UPGRADE MODAL
   ============================================================ */

.modal{
    display:none;
    position:fixed;
    inset:0;
    background:rgba(15,23,42,.55);
    z-index:100;
    align-items:center;
    justify-content:center;
    padding:18px;
}

.modal.show{
    display:flex;
}

.modal-box{
    width:min(980px,100%);
    max-height:90vh;
    overflow:auto;
    background:#fff;
    border-radius:24px;
    padding:24px;
}

.modal-head{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:12px;
}

.close{
    border:0;
    background:#f1f5f9;
    width:40px;
    height:40px;
    border-radius:50%;
}

.plans{
    display:grid;
    grid-template-columns:
        repeat(3,1fr);
    gap:14px;
    margin-top:20px;
}

.plan-card{
    border:1px solid #e2e8f0;
    border-radius:18px;
    padding:18px;
}

.plan-card.best{
    border:2px solid #6366f1;
}

.price{
    font-size:30px;
    font-weight:900;
    margin:10px 0;
}

.plan-card ul{
    padding-left:20px;
    color:#475569;
    line-height:1.8;
}

.buy{
    width:100%;
    border:0;
    border-radius:12px;
    padding:12px;
    background:#111827;
    color:#fff;
    font-weight:800;
}

@media(max-width:760px){

    .plans{
        grid-template-columns:1fr;
    }

    .upgrade{
        padding:8px 10px;
        font-size:12px;
    }

}

</style>

</head>

<body>

<div id="app">

<aside id="sidebar">

<div class="sidebar-brand">
Veya
</div>

<button
class="new-chat"
onclick="newChat()">
＋ New chat
</button>

<div
id="history"
class="history">
</div>

</aside>

<div
id="overlay"
onclick="toggleSidebar()">
</div>

<main id="main">

<header class="topbar">

<button
class="menu"
onclick="toggleSidebar()">
☰
</button>

<div class="brand">
Veya
</div>

<div class="spacer"></div>

<button
class="upgrade"
onclick="openUpgrade()">
Upgrade
</button>

<div
id="plan"
class="plan">
Veya Free
</div>

</header>

<section
id="chat"
class="chat">

<div
id="welcome"
class="welcome">

<h1>
One workspace.<br>
<span class="gradient">
Many AI minds.
</span>
</h1>

<p>
Veya verifies configured models and routes
your task to the right AI provider.
</p>

</div>

</section>

<div class="composer-wrap">

<div class="composer">

<div
id="picker"
class="picker">

<select
id="modelSelect">
</select>

</div>

<div class="row">

<textarea
id="input"
placeholder="Message Veya..."
onkeydown="handleKey(event)">
</textarea>

<button
class="send"
onclick="sendMessage()">
↑
</button>

</div>

</div>

</div>

</main>

</div>


<!-- ==========================================================
     UPGRADE MODAL
     ========================================================== -->

<div
id="upgradeModal"
class="modal">

<div class="modal-box">

<div class="modal-head">

<div>

<h2>
Upgrade Veya
</h2>

<p>
Choose the AI power level you want.
</p>

</div>

<button
class="close"
onclick="closeUpgrade()">
×
</button>

</div>


<div class="plans">


<div class="plan-card">

<h3>
Veya Core
</h3>

<div class="price">
₹399
<span style="font-size:13px">
/month
</span>
</div>

<ul>
<li>OpenAI flagship</li>
<li>Anthropic flagship</li>
<li>Google flagship</li>
<li>Model selection</li>
<li>100 messages/day</li>
</ul>

<button
class="buy"
onclick="buyPlan('core')">
Choose Core
</button>

</div>


<div class="plan-card best">

<h3>
Veya All AI
</h3>

<div class="price">
₹999
<span style="font-size:13px">
/month
</span>
</div>

<ul>
<li>All 10 AI providers</li>
<li>Best live model from each</li>
<li>Multi-provider routing</li>
<li>Verification mode</li>
<li>300 messages/day</li>
</ul>

<button
class="buy"
onclick="buyPlan('all_ai')">
Choose All AI
</button>

</div>


<div class="plan-card">

<h3>
Veya Ultra
</h3>

<div class="price">
₹2499
<span style="font-size:13px">
/month
</span>
</div>

<ul>
<li>All 10 providers</li>
<li>Up to 5 best live models/provider</li>
<li>Advanced model picker</li>
<li>Verification</li>
<li>1000 messages/day</li>
</ul>

<button
class="buy"
onclick="buyPlan('ultra')">
Choose Ultra
</button>

</div>

</div>

<p
style="
margin-top:20px;
font-size:12px;
color:#64748b;
">

Availability depends on the provider API keys,
account permissions, model catalogue and billing.

</p>

</div>

</div>


<script>

let currentChat = null;
let currentPlan = "free";
let models = [];


async function api(
    url,
    options = {}
){

    const response =
        await fetch(
            url,
            {
                credentials:
                    "same-origin",

                ...options,

                headers:{
                    "Content-Type":
                        "application/json",

                    ...(options.headers || {})
                }
            }
        );

    const data =
        await response.json();

    if(!response.ok){

        throw new Error(
            data.error
            || data.message
            || "Request failed"
        );
    }

    return data;
}


function toggleSidebar(){

    document
        .getElementById(
            "sidebar"
        )
        .classList.toggle(
            "open"
        );

    document
        .getElementById(
            "overlay"
        )
        .classList.toggle(
            "show"
        );
}


function openUpgrade(){

    document
        .getElementById(
            "upgradeModal"
        )
        .classList.add(
            "show"
        );
}


function closeUpgrade(){

    document
        .getElementById(
            "upgradeModal"
        )
        .classList.remove(
            "show"
        );
}


async function buyPlan(plan){

    try{

        const data =
            await api(
                "/api/upgrade",
                {
                    method:
                        "POST",

                    body:
                        JSON.stringify({
                            plan:
                                plan
                        })
                }
            );

        /*
         * Razorpay Checkout can be added here using
         * the returned subscription_id and razorpay_key_id.
         *
         * Do NOT put RAZORPAY_KEY_SECRET in this browser.
         */

        alert(
            "Subscription created: "
            + data.subscription_id
        );

    }catch(error){

        alert(
            error.message
        );
    }
}


async function loadMe(){

    const data =
        await api(
            "/api/me"
        );

    currentPlan =
        data.plan;

    document
        .getElementById(
            "plan"
        )
        .textContent =
        data.plan_info.name;

    await loadModels();
}


async function loadModels(){

    const data =
        await api(
            "/api/models"
        );

    models =
        data.models || [];

    const select =
        document.getElementById(
            "modelSelect"
        );

    select.innerHTML = "";

    if(!models.length){

        const option =
            document.createElement(
                "option"
            );

        option.textContent =
            "Automatic model routing";

        select.appendChild(
            option
        );

        return;
    }


    for(
        const item
        of models
    ){

        const option =
            document.createElement(
                "option"
            );

        option.value =
            JSON.stringify({

                provider:
                    item.provider,

                model:
                    item.model
            });

        option.textContent =
            item.provider.toUpperCase()
            + " — "
            + item.agent_name
            + " — "
            + item.model;

        select.appendChild(
            option
        );
    }
}


async function loadHistory(){

    const data =
        await api(
            "/api/chats"
        );

    const history =
        document.getElementById(
            "history"
        );

    history.innerHTML = "";

    for(
        const item
        of data
    ){

        const div =
            document.createElement(
                "div"
            );

        div.className =
            "history-item";

        div.textContent =
            item.title;

        div.onclick =
            () => openChat(
                item.id
            );

        history.appendChild(
            div
        );
    }
}


async function newChat(){

    const data =
        await api(
            "/api/chats/new",
            {
                method:
                    "POST"
            }
        );

    currentChat =
        data.chat_id;

    document
        .getElementById(
            "chat"
        )
        .innerHTML = "";

    await loadHistory();

    toggleSidebar();
}


async function openChat(id){

    currentChat = id;

    const data =
        await api(
            "/api/chats/"
            + id
        );

    const chat =
        document.getElementById(
            "chat"
        );

    chat.innerHTML = "";

    for(
        const message
        of data.messages
    ){

        addMessage(

            message.role,

            message.content,

            message.provider,

            message.model,

            message.agent_role
        );
    }

    toggleSidebar();
}


function addMessage(
    role,
    content,
    provider,
    model,
    agent
){

    const welcome =
        document.getElementById(
            "welcome"
        );

    if(welcome){
        welcome.remove();
    }

    const chat =
        document.getElementById(
            "chat"
        );

    const wrapper =
        document.createElement(
            "div"
        );

    wrapper.className =
        "message "
        + (
            role === "user"
            ? "user"
            : "assistant"
        );


    if(
        role === "assistant"
        && provider
        && model
    ){

        const tag =
            document.createElement(
                "div"
            );

        tag.className =
            "tag";

        tag.textContent =
            provider.toUpperCase()
            + " · "
            + model
            + (
                agent
                ? " · " + agent
                : ""
            );

        wrapper.appendChild(
            tag
        );
    }


    const body =
        document.createElement(
            "div"
        );

    body.textContent =
        content;

    wrapper.appendChild(
        body
    );

    chat.appendChild(
        wrapper
    );

    chat.scrollTop =
        chat.scrollHeight;
}


function handleKey(event){

    if(
        event.key === "Enter"
        && !event.shiftKey
    ){

        event.preventDefault();

        sendMessage();
    }
}


async function sendMessage(){

    const input =
        document.getElementById(
            "input"
        );

    const text =
        input.value.trim();

    if(!text){
        return;
    }

    input.value = "";

    addMessage(
        "user",
        text
    );


    let selected = null;


    if(
        currentPlan !== "free"
        &&
        models.length
    ){

        try{

            selected =
                JSON.parse(

                    document
                        .getElementById(
                            "modelSelect"
                        )
                        .value
                );

        }catch(error){

            selected = null;
        }
    }


    try{

        const data =
            await api(
                "/api/chat",
                {

                    method:
                        "POST",

                    body:
                        JSON.stringify({

                            chat_id:
                                currentChat,

                            message:
                                text,

                            provider:
                                selected
                                ? selected.provider
                                : null,

                            model:
                                selected
                                ? selected.model
                                : null,
                        })
                }
            );


        currentChat =
            data.chat_id;


        addMessage(

            "assistant",

            data.answer,

            data.provider,

            data.model,

            data.agent
        );


        await loadHistory();


    }catch(error){

        addMessage(

            "assistant",

            "Error: "
            + error.message
        );
    }
}


async function boot(){

    try{

        await loadMe();

        await loadHistory();

    }catch(error){

        addMessage(

            "assistant",

            "Startup error: "
            + error.message
        );
    }
}


boot();

</script>

</body>

</html>
"""


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    user_id = get_user_id()

    response = make_response(
        render_template_string(
            HTML
        )
    )

    response.set_cookie(
        "veya_user",
        user_id,
        httponly=True,
        samesite="Lax",
    )

    return response


# ============================================================
# STARTUP
# ============================================================

init_db()


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
    )