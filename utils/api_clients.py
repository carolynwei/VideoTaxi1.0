from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional

import requests

try:
    import streamlit as st  # type: ignore
except ImportError:  # pragma: no cover - streamlit may not be installed in all environments
    st = None  # type: ignore

try:
    # OpenAI Python SDK (used for DeepSeek API, which is compatible with OpenAI)
    from openai import OpenAI
except ImportError:  # pragma: no cover - SDK may not be installed in all environments
    OpenAI = None  # type: ignore


ARK_RESPONSES_URL = "https://ark.cn-beijing.volces.com/api/v3/responses"
TIANAPI_DOUYIN_URL = "https://apis.tianapi.com/douyinhot/index"


class ArkAPIError(RuntimeError):
    """Raised when the Ark (Doubao) API returns an error or unexpected payload."""


class TianAPIError(RuntimeError):
    """Raised when the TianAPI (Douyin hot trends) API returns an error."""


class DeepSeekAPIError(RuntimeError):
    """Raised when the DeepSeek API returns an error or unexpected payload."""


def _get_config_value(key: str) -> str:
    """
    Read configuration from Streamlit secrets first, then environment variables.

    Raises:
        KeyError: If the key is missing in both places.
    """
    # Prefer Streamlit secrets when available
    if st is not None:
        try:
            if key in st.secrets:
                value = str(st.secrets[key])
                if value:
                    return value
        except Exception:
            # Ignore secrets access issues and fall back to environment variables
            pass

    value = os.getenv(key, "").strip()
    if not value:
        raise KeyError(f"Required configuration '{key}' is not set in env or Streamlit secrets.")
    return value


def _extract_ark_output_text(data: Mapping[str, Any]) -> str:
    """
    Extract the main model output text from Ark / Doubao responses.

    The exact schema can vary slightly between versions; this function
    defensively supports the common layouts used by the v3 Responses API.
    """
    # Preferred: Responses API-style: output -> message -> content[*].text
    output = data.get("output")
    if isinstance(output, Mapping):
        message = output.get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, list):
                texts: List[str] = []
                for item in content:
                    if not isinstance(item, Mapping):
                        continue
                    item_type = item.get("type")
                    if item_type in ("output_text", "text", "message"):
                        text_val = item.get("text")
                        if isinstance(text_val, str) and text_val.strip():
                            texts.append(text_val.strip())
                if texts:
                    return "\n".join(texts)

        # Fallback: output -> choices[0] -> message -> content[*].text
        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, Mapping):
                msg = first_choice.get("message")
                if isinstance(msg, Mapping):
                    content = msg.get("content")
                    if isinstance(content, list):
                        for item in content:
                            if not isinstance(item, Mapping):
                                continue
                            text_val = item.get("text")
                            if isinstance(text_val, str) and text_val.strip():
                                return text_val.strip()

    # Legacy-style: top-level choices[0].message.content[0].text
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, Mapping):
            msg = first_choice.get("message")
            if isinstance(msg, Mapping):
                content = msg.get("content")
                if isinstance(content, list) and content:
                    first_item = content[0]
                    if isinstance(first_item, Mapping):
                        text_val = first_item.get("text")
                        if isinstance(text_val, str) and text_val.strip():
                            return text_val.strip()

    # If we reach here, the structure is unexpected.
    raise ArkAPIError(
        "Unable to extract text from Ark API response. "
        f"Top-level keys: {list(data.keys())}"
    )


def _call_ark_api(payload: Dict[str, Any], *, timeout: int) -> Dict[str, Any]:
    """
    Low-level helper to call the Ark Responses API and return parsed JSON.

    Shared by higher-level generation helpers to ensure consistent
    error handling and logging.
    """
    api_key = _get_config_value("ARK_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            ARK_RESPONSES_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:  # Network or low-level error
        raise ArkAPIError(f"Failed to call Ark API: {exc}") from exc

    if not response.ok:
        # Print status and body for easier debugging during development
        print(f"[ArkAPIError] HTTP {response.status_code}: {response.text}")
        raise ArkAPIError(
            f"Ark API returned HTTP {response.status_code}. "
            "See printed response body for details."
        )

    try:
        data: Dict[str, Any] = response.json()
    except json.JSONDecodeError as exc:
        print(f"[ArkAPIError] Non-JSON response body: {response.text}")
        raise ArkAPIError("Ark API returned invalid JSON.") from exc

    return data


def generate_script_with_search(query: str, *, timeout: int = 30) -> str:
    """
    Call Doubao (Ark) model with web_search tool enabled to generate a short-video script.

    The model and API key are read from configuration:
        - ARK_API_KEY
        - ARK_MODEL_ID

    Args:
        query: User's request text, e.g. "结合今日热点写一个短视频脚本".
        timeout: Optional requests timeout in seconds (default 30).

    Returns:
        The generated script text from the model.

    Raises:
        ArkAPIError: If the HTTP request fails or payload is malformed.
        KeyError: If required configuration values are missing.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    model_id = _get_config_value("ARK_MODEL_ID")

    payload: Dict[str, Any] = {
        "model": model_id,
        "stream": False,
        "tools": [
            {"type": "web_search"},
        ],
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": query.strip(),
                    }
                ],
            }
        ],
    }

    data = _call_ark_api(payload, timeout=timeout)
    return _extract_ark_output_text(data)


def generate_video_script(topic: str, *, timeout: int = 30) -> Dict[str, Any]:
    """
    Generate a structured, JSON-format short-video script based on a hot topic,
    using Doubao (Ark) with web_search enabled.

    The System Prompt explicitly instructs the model to return ONLY a JSON
    object with the schema:
        {
          "title": "视频标题",
          "narration": "旁白文案（30秒左右，要搞笑幽默）",
          "visual_scenes": ["分镜1的中文描述", "分镜2的中文描述", "分镜3的中文描述"],
          "bgm_style": "搞笑/反转/悬疑"
        }

    Args:
        topic: The selected hot topic from Douyin trends.
        timeout: Optional requests timeout in seconds (default 30).

    Returns:
        A Python dict parsed from the JSON returned by the model.

    Raises:
        ArkAPIError: If the HTTP request fails, payload is malformed,
                     or the model output is not valid JSON.
        KeyError: If required configuration values are missing.
    """
    if not topic or not topic.strip():
        raise ValueError("Topic must be a non-empty string.")

    model_id = _get_config_value("ARK_MODEL_ID")

    system_prompt = (
        "你是一个短视频脚本创作助手。请根据给定的热点话题，生成一个大约30秒、"
        "搞笑幽默风格的中文短视频方案。你必须严格按照下面的 JSON 结构返回结果，"
        "并且只返回 JSON，不能包含任何额外说明、注释或 Markdown 代码块：\n\n"
        '{\n'
        '  "title": "视频标题",\n'
        '  "narration": "旁白文案（30秒左右，要搞笑幽默）",\n'
        '  "visual_scenes": ["分镜1的中文描述", "分镜2的中文描述", "分镜3的中文描述"],\n'
        '  "bgm_style": "搞笑/反转/悬疑"\n'
        "}\n\n"
        "请确保返回的是合法的 JSON 对象字符串，字段名必须为英文，且不要在 JSON 外再添加其他文字。"
    )

    user_prompt = (
        f"当前热点话题是：{topic.strip()}。\n"
        "请围绕这个话题创作一个适合抖音平台的短视频脚本，"
        "要求风格搞笑幽默，时长约30秒。"
    )

    payload: Dict[str, Any] = {
        "model": model_id,
        "stream": False,
        "tools": [
            {"type": "web_search"},
        ],
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": system_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_prompt,
                    }
                ],
            },
        ],
    }

    data = _call_ark_api(payload, timeout=timeout)
    raw_text = _extract_ark_output_text(data)

    try:
        parsed: Dict[str, Any] = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        # Print raw text for easier debugging of prompt/format issues
        print("[ArkAPIError] Model did not return valid JSON. Raw output:")
        print(raw_text)
        raise ArkAPIError("Model output is not valid JSON.") from exc

    return parsed


def get_douyin_hot_trends(limit: int = 10, *, timeout: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch current Douyin hot trends (top N) from TianAPI.

    Reads API key from configuration:
        - TIANAPI_KEY

    Returns:
        A list of dicts, each containing at least:
            {
              "title": <话题标题>,
              "hot": <热度指数>
            }

    Raises:
        TianAPIError: If the HTTP request fails or payload is malformed.
        KeyError: If required configuration values are missing.
    """
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")

    api_key = _get_config_value("TIANAPI_KEY")

    params = {"key": api_key}

    try:
        response = requests.get(
            TIANAPI_DOUYIN_URL,
            params=params,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise TianAPIError(f"Failed to call TianAPI Douyin hot trends: {exc}") from exc

    if not response.ok:
        print(f"[TianAPIError] HTTP {response.status_code}: {response.text}")
        raise TianAPIError(
            f"TianAPI returned HTTP {response.status_code}. "
            "See printed response body for details."
        )

    try:
        data: Dict[str, Any] = response.json()
    except json.JSONDecodeError as exc:
        print(f"[TianAPIError] Non-JSON response body: {response.text}")
        raise TianAPIError("TianAPI returned invalid JSON.") from exc

    code = data.get("code")
    if code != 200:
        msg = data.get("msg") or data.get("message") or "unknown error"
        print(f"[TianAPIError] code={code}, msg={msg}, body={data}")
        raise TianAPIError(f"TianAPI business error: code={code}, msg={msg}")

    result = data.get("result")
    items: Optional[List[Any]] = None

    if isinstance(result, Mapping):
        # Newer TianAPI pattern: result.list is the array
        list_obj = result.get("list") or result.get("newslist") or result.get("data")
        if isinstance(list_obj, list):
            items = list_obj

    # Fallback for older/alternative structures
    if items is None:
        for key in ("list", "newslist", "data"):
            maybe_items = data.get(key)
            if isinstance(maybe_items, list):
                items = maybe_items
                break

    if not items:
        raise TianAPIError("TianAPI response does not contain any hot trend items.")

    trends: List[Dict[str, Any]] = []
    for item in items[:limit]:
        if not isinstance(item, Mapping):
            continue
        # TianAPI douyinhot uses 'word' for topic, 'hotindex' for hotness
        title_val = item.get("word") or item.get("title") or ""
        hot_val = item.get("hotindex") or item.get("hot") or 0

        if isinstance(title_val, str):
            title = title_val.strip()
        else:
            title = str(title_val)

        trends.append(
            {
                "title": title,
                "hot": hot_val,
            }
        )

    return trends


def optimize_visual_prompt(chinese_scenes_list: List[str], *, temperature: float = 0.4) -> List[str]:
    """
    Optimize a list of Chinese visual scenes into high-quality English prompts
    for AI video generation using DeepSeek (compatible with OpenAI SDK).

    This function calls the DeepSeek `deepseek-chat` model via the official
    `openai` Python package, with base_url set to `https://api.deepseek.com`
    and API key read from configuration:
        - DEEPSEEK_API_KEY

    Each input scene corresponds to exactly one English prompt in the output,
    and the prompts are enriched with cinematic camera language such as:
    "cinematic, 4k, hyper-realistic, dynamic lighting".

    Args:
        chinese_scenes_list: List of Chinese scene descriptions from Doubao's
                             `visual_scenes` output.
        temperature: Sampling temperature for DeepSeek generation.

    Returns:
        A list of English prompts with the same length as `chinese_scenes_list`.

    Raises:
        DeepSeekAPIError: If the SDK is missing, the HTTP call fails,
                          or the model output is not valid JSON list of strings.
        KeyError: If required configuration values are missing.
    """
    if not chinese_scenes_list:
        raise ValueError("chinese_scenes_list must be a non-empty list of strings.")

    if any(not isinstance(scene, str) or not scene.strip() for scene in chinese_scenes_list):
        raise ValueError("Each scene in chinese_scenes_list must be a non-empty string.")

    if OpenAI is None:
        raise DeepSeekAPIError(
            "The 'openai' Python package is required to call the DeepSeek API. "
            "Please install it with 'pip install openai'."
        )

    api_key = _get_config_value("DEEPSEEK_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    scenes_json = json.dumps(chinese_scenes_list, ensure_ascii=False)

    system_prompt = (
        "You are an expert prompt engineer for AI video generation. "
        "Given a list of Chinese short-video scenes, you must convert each scene "
        "into a high-quality English prompt suitable for text-to-video models. "
        "For each scene, include rich cinematic camera language and visual details, "
        "such as: cinematic, 4k, hyper-realistic, dynamic lighting, depth of field, "
        "shot type (close-up / medium shot / wide shot), camera movement, etc. "
        "The number of English prompts must be exactly the same as the number of "
        "input scenes, and they must be in the same order."
        "\n\n"
        "IMPORTANT FORMAT REQUIREMENT:\n"
        "- You MUST return ONLY a JSON array of strings, like:\n"
        '  ["prompt for scene 1", "prompt for scene 2", "..."]\n'
        "- Do NOT add any extra text, explanations, or Markdown formatting outside the JSON array."
    )

    user_prompt = (
        "下面是一个中文短视频分镜列表，请逐条将其转化为适合 AI 视频生成的英文提示词（Prompt），"
        "并加入高质量的摄影机语言，例如：cinematic, 4k, hyper-realistic, dynamic lighting。\n\n"
        f"中文分镜列表（JSON 数组）如下：\n{scenes_json}"
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
    except Exception as exc:
        # DeepSeek (OpenAI-compatible) SDK wraps HTTP errors into exceptions
        raise DeepSeekAPIError(f"Failed to call DeepSeek API: {exc}") from exc

    try:
        choice = response.choices[0]
        raw_content = choice.message.content  # type: ignore[assignment]
    except (AttributeError, IndexError, KeyError) as exc:
        raise DeepSeekAPIError(
            f"Unexpected DeepSeek response structure: {response!r}"
        ) from exc

    if not isinstance(raw_content, str):
        raise DeepSeekAPIError(
            f"Unexpected DeepSeek message content type: {type(raw_content)!r}"
        )

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        print("[DeepSeekAPIError] Model did not return valid JSON array. Raw output:")
        print(raw_content)
        raise DeepSeekAPIError("Model output is not valid JSON.") from exc

    if not isinstance(parsed, list):
        raise DeepSeekAPIError(
            f"Model output JSON is not a list: {type(parsed)!r}"
        )

    if any(not isinstance(item, str) for item in parsed):
        raise DeepSeekAPIError("Model output JSON list must contain only strings.")

    if len(parsed) != len(chinese_scenes_list):
        # Not fatal to print mismatch; still raise for strictness
        print(
            f"[DeepSeekAPIError] Output length ({len(parsed)}) does not match "
            f"input length ({len(chinese_scenes_list)})."
        )
        raise DeepSeekAPIError(
            "Model output list length does not match input scenes list length."
        )

    return parsed


__all__ = [
    "ArkAPIError",
    "generate_script_with_search",
    "generate_video_script",
    "TianAPIError",
    "get_douyin_hot_trends",
    "DeepSeekAPIError",
    "optimize_visual_prompt",
]


if __name__ == "__main__":
    # Simple manual test for DeepSeek visual prompt optimization.
    sample_scenes = [
        "镜头1：早晨的城市街道，年轻人背着书包快步走在路上。",
        "镜头2：办公室里，主角一脸绝望地看着堆成山的文件。",
        "镜头3：夜晚屋顶，主角仰望星空，突然露出轻松的笑容。",
    ]

    print("=== Testing optimize_visual_prompt with DeepSeek ===")
    try:
        prompts = optimize_visual_prompt(sample_scenes)
    except Exception as exc:  # noqa: BLE001 - manual test, catch-all is acceptable here
        print(f"DeepSeek optimize_visual_prompt test failed: {exc}")
    else:
        from pprint import pprint

        pprint(prompts)

