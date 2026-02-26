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

try:
    # Exa 实时搜索 SDK，用于在 Kimi 生成脚本前先抓一批事实材料
    from exa_py import Exa
except ImportError:  # pragma: no cover - SDK may不一定安装
    Exa = None  # type: ignore


ARK_RESPONSES_URL = "https://ark.cn-beijing.volces.com/api/v3/responses"
ARK_CHAT_COMPLETIONS_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
TIANAPI_DOUYIN_URL = "https://apis.tianapi.com/douyinhot/index"


class ArkAPIError(RuntimeError):
    """Raised when the Ark (Doubao) API returns an error or unexpected payload."""


class TianAPIError(RuntimeError):
    """Raised when the TianAPI (Douyin hot trends) API returns an error."""


class DeepSeekAPIError(RuntimeError):
    """Raised when the DeepSeek API returns an error or unexpected payload."""


class ExaAPIError(RuntimeError):
    """Raised when the Exa web search API returns an error or unexpected payload."""


_EXA_CLIENT: Optional["Exa"] = None


def _get_exa_client() -> "Exa":
    """
    Lazily construct a global Exa client.

    优先从 Streamlit secrets 读取 EXA_API_KEY，其次从环境变量 EXA_API_KEY / WEB_KEY。
    """
    global _EXA_CLIENT
    if _EXA_CLIENT is not None:
        return _EXA_CLIENT

    if Exa is None:
        raise ExaAPIError(
            "Exa SDK 未安装，请先执行: pip install exa-py"
        )

    api_key = ""
    # 优先 secrets
    if st is not None:
        try:
            if "EXA_API_KEY" in st.secrets:
                api_key = str(st.secrets["EXA_API_KEY"]).strip()
        except Exception:
            api_key = ""
    # 其次环境变量（兼容 EXA_API_KEY / WEB_KEY）
    if not api_key:
        api_key = os.getenv("EXA_API_KEY", "").strip() or os.getenv("WEB_KEY", "").strip()

    if not api_key:
        raise ExaAPIError("EXA_API_KEY / WEB_KEY 未配置，无法使用 Exa 联网搜索。")

    _EXA_CLIENT = Exa(api_key=api_key)
    return _EXA_CLIENT


def fetch_topic_facts_with_exa(topic: str, *, max_chars: int = 8000) -> str:
    """
    使用 Exa 对话题做一次聚合搜索，返回一段“事实材料”文本，供 Kimi 参考。

    返回的是一段纯文本，包含若干条「来源 + 摘要」，长度控制在 max_chars 内。
    """
    topic = (topic or "").strip()
    if not topic:
        return ""

    try:
        client = _get_exa_client()
    except ExaAPIError as exc:
        print(f"[ExaAPIError] {exc}")
        return ""

    try:
        results = client.search(
            query=f"与「{topic}」相关的最新事实、新闻事件、数据和背景信息，中文结果优先",
            type="auto",
            num_results=6,
            contents={"text": {"max_characters": 2000}},
        )
    except Exception as exc:  # pragma: no cover - 网络/服务错误
        print(f"[ExaAPIError] 搜索失败: {exc}")
        return ""

    lines: List[str] = []
    for i, r in enumerate(results.results, start=1):
        title = (getattr(r, "title", "") or "").strip()
        url = (getattr(r, "url", "") or "").strip()
        snippet = (getattr(r, "text", "") or "").strip()
        if not (title or snippet):
            continue
        block = f"[资料 {i}] {title}\n来源：{url}\n内容概括：{snippet}\n"
        lines.append(block)

    if not lines:
        return ""

    combined = "\n\n".join(lines).strip()
    if len(combined) > max_chars:
        combined = combined[: max_chars] + "\n\n…（其余内容已截断）"
    return combined


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


def _get_doubao_api_key() -> str:
    """
    Return API key for Doubao / Ark endpoints.

    Prefer ARK_API_KEY (火山方舟豆包专用 Key)，如果用户沿用旧文档配置了 LAS_API_KEY，
    则作为兜底兼容。
    """
    try:
        return _get_config_value("ARK_API_KEY")
    except KeyError:
        return _get_config_value("LAS_API_KEY")


def _extract_ark_output_text(data: Mapping[str, Any]) -> str:
    """
    Extract the main model output text from Ark / Doubao responses.

    The exact schema can vary slightly between versions; this function
    defensively supports the common layouts used by the v3 Responses API.
    """
    def _collect_text_from_content_list(content_list: Any) -> List[str]:
        texts: List[str] = []
        if not isinstance(content_list, list):
            return texts

        for item in content_list:
            if not isinstance(item, Mapping):
                continue
            item_type = item.get("type")

            # Standard pattern: {"type": "output_text", "text": "xxx"}
            if item_type in ("output_text", "text", "message", "output_message"):
                text_val = item.get("text")
                # Newer schema may use a list of segments:
                # {"type": "output_text", "text": [{"type": "text", "text": "..."}, ...]}
                if isinstance(text_val, str) and text_val.strip():
                    texts.append(text_val.strip())
                elif isinstance(text_val, list):
                    segments: List[str] = []
                    for seg in text_val:
                        if not isinstance(seg, Mapping):
                            continue
                        seg_text = seg.get("text")
                        if isinstance(seg_text, str) and seg_text.strip():
                            segments.append(seg_text.strip())
                    if segments:
                        texts.append("".join(segments))

        return texts

    def _find_first_string(obj: Any) -> Optional[str]:
        """
        Very defensive fallback: walk the structure and return
        the first reasonably long string we find.
        """
        if isinstance(obj, str):
            text = obj.strip()
            if text:
                return text
            return None
        if isinstance(obj, Mapping):
            for value in obj.values():
                found = _find_first_string(value)
                if found:
                    return found
        if isinstance(obj, list):
            for value in obj:
                found = _find_first_string(value)
                if found:
                    return found
        return None

    # Preferred: Responses API-style: output -> message -> content[*].text
    output = data.get("output")
    if isinstance(output, Mapping):
        message = output.get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
            texts = _collect_text_from_content_list(content)
            if texts:
                return "\n".join(texts)

        # Some models may put text directly under output["text"]
        direct_text = output.get("text")
        if isinstance(direct_text, str) and direct_text.strip():
            return direct_text.strip()

        # Fallback: output -> choices[0] -> message -> content[*].text
        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, Mapping):
                msg = first_choice.get("message")
                if isinstance(msg, Mapping):
                    content = msg.get("content")
                    texts = _collect_text_from_content_list(content)
                    if texts:
                        return "\n".join(texts)

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

    # As a very defensive fallback, try to walk through `output`
    # (if present) or the whole payload to find the first string.
    if isinstance(output, Mapping):
        fallback = _find_first_string(output)
        if fallback:
            return fallback

    fallback = _find_first_string(data)
    if fallback:
        return fallback

    # If we reach here, the structure is unexpected.
    raise ArkAPIError(
        "Unable to extract text from Ark API response. "
        f"Top-level keys: {list(data.keys())}"
    )


def _call_ark_responses_with_web_search(
    model_id: str,
    user_text: str,
    *,
    system_prefix: Optional[str] = None,
    timeout: int = 90,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    调用 Ark v3 Responses API，使用 web_search 工具（适用于 Kimi 等模型）。
    请求格式参考：model + tools[web_search] + input[user content]。
    """
    api_key = _get_doubao_api_key()
    full_text = (system_prefix + "\n\n【用户请求】\n" + user_text) if system_prefix else user_text
    payload: Dict[str, Any] = {
        "model": model_id,
        "stream": stream,
        "tools": [{"type": "web_search", "max_keyword": 3}],
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": full_text}],
            }
        ],
    }
    if stream:
        return _call_ark_responses_stream(api_key, payload, timeout=timeout)
    return _call_ark_api(payload, timeout=timeout)


def _call_ark_responses_stream(
    api_key: str, payload: Dict[str, Any], *, timeout: int
) -> Dict[str, Any]:
    """
    以流式方式调用 Ark Responses API，聚合所有 output_text 为一条完整响应，
    返回与非流式兼容的结构：{"output": {"message": {"content": [{"type": "output_text", "text": "..."}]}}}.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        ARK_RESPONSES_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
        stream=True,
    )
    if not resp.ok:
        print(f"[ArkAPIError] HTTP {resp.status_code}: {resp.text}")
        raise ArkAPIError(
            f"Ark Responses API returned HTTP {resp.status_code}. See printed body."
        )
    chunks: List[str] = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.strip():
            continue
        if line.startswith("data:"):
            data_str = line[5:].strip()
            if data_str == "[DONE]" or data_str == "":
                continue
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            output = event.get("output") if isinstance(event, dict) else None
            if not isinstance(output, dict):
                continue
            content = output.get("message", {}).get("content") if isinstance(output.get("message"), dict) else output.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "output_text":
                        t = item.get("text")
                        if isinstance(t, str) and t:
                            chunks.append(t)
            text = output.get("text")
            if isinstance(text, str) and text:
                chunks.append(text)
    combined = "".join(chunks).strip()
    if not combined:
        raise ArkAPIError(
            "Ark Responses API (stream) returned no output_text. "
            "Check model id and stream event format."
        )
    return {"output": {"message": {"content": [{"type": "output_text", "text": combined}]}}}


def _call_ark_api(payload: Dict[str, Any], *, timeout: int) -> Dict[str, Any]:
    """
    Low-level helper to call the Ark Responses API and return parsed JSON.

    Shared by higher-level generation helpers to ensure consistent
    error handling and logging.
    """
    api_key = _get_doubao_api_key()

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


def _call_doubao_chat_completions(
    messages: List[Dict[str, Any]],
    *,
    timeout: int,
    model_id: Optional[str] = None,
    temperature: float = 0.7,
    enable_web_search: bool = False,
) -> Dict[str, Any]:
    """
    Call Doubao chat completions endpoint
    (https://ark.cn-beijing.volces.com/api/v3/chat/completions)
    and return parsed JSON.

    This follows the OpenAI-compatible Chat Completions schema and is
    generally the recommended way to call豆包 models now.
    """
    if not messages:
        raise ValueError("messages must be a non-empty list.")

    if model_id is None:
        model_id = _get_config_value("ARK_MODEL_ID")

    api_key = _get_doubao_api_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
    }
    if enable_web_search:
        # Ark 扩展字段：开启联网搜索能力
        payload["enable_web_search"] = True

    try:
        response = requests.post(
            ARK_CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise ArkAPIError(f"Failed to call Doubao Chat API: {exc}") from exc

    if not response.ok:
        print(f"[ArkAPIError] HTTP {response.status_code}: {response.text}")
        raise ArkAPIError(
            f"Doubao Chat API returned HTTP {response.status_code}. "
            "See printed response body for details."
        )

    try:
        data: Dict[str, Any] = response.json()
    except json.JSONDecodeError as exc:
        print(f"[ArkAPIError] Non-JSON response body: {response.text}")
        raise ArkAPIError("Doubao Chat API returned invalid JSON.") from exc

    return data


def _extract_doubao_chat_content(data: Mapping[str, Any]) -> str:
    """
    Extract assistant message content from Doubao chat completions response.
    """
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, Mapping):
            message = first_choice.get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()

    raise ArkAPIError(
        "Unexpected Doubao Chat API response structure. "
        f"Top-level keys: {list(data.keys())}"
    )


def generate_script_with_search(query: str, *, timeout: int = 90) -> str:
    """
    Call Doubao model via Chat Completions API to generate a short-video script.

    The model and API key are read from configuration:
        - ARK_API_KEY or LAS_API_KEY
        - ARK_MODEL_ID

    Args:
        query: User's request text, e.g. "结合今日热点写一个短视频脚本".
        timeout: Optional requests timeout in seconds (default 90; 豆包联网生成约 30–60s).

    Returns:
        The generated script text from the model.

    Raises:
        ArkAPIError: If the HTTP request fails or payload is malformed.
        KeyError: If required configuration values are missing.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    model_id = _get_config_value("ARK_MODEL_ID")

    system_prompt = (
        "你是一个具备联网搜索能力的短视频脚本创作助手，后端模型已开启 enable_web_search。"
        "在回答前，你应优先通过联网搜索获取与用户问题相关的最新信息、事实数据、"
        "社交媒体热梗和新闻背景，再基于这些真实信息进行创作。\n\n"
        "请遵循以下原则：\n"
        "1. 必要时使用联网搜索补充最新信息，尤其是涉及热点事件、人物、数据或实时趋势的问题。\n"
        "2. 避免编造具体数据和事实，如确实查不到，请直接说明不确定，而不要凭空捏造。\n"
        "3. 输出内容仍需是连贯的中文短视频脚本，可以包含分镜提示和旁白文案，适合抖音平台的表达风格。\n"
        "4. 不要在回答中解释你是如何调用搜索，只展示最终创作结果即可。"
    )

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": query.strip(),
        },
    ]

    data = _call_doubao_chat_completions(
        messages,
        timeout=timeout,
        model_id=model_id,
        temperature=0.8,
        enable_web_search=True,
    )
    return _extract_doubao_chat_content(data)


def generate_video_script(topic: str, *, timeout: int = 90) -> Dict[str, Any]:
    """
    Generate a structured, JSON-format short-video script based on a hot topic.
    When ARK_MODEL_ID contains \"kimi\", uses Ark Responses API with web_search tool (Kimi);
    otherwise uses Ark Chat Completions API (e.g. Doubao).

    The System Prompt explicitly instructs the model to return ONLY a JSON
    object with the schema:
        {
          "title": "视频标题",
          "narration": "旁白文案（约30秒；会原样用于 TTS 配音与画面字幕，需断句清晰、每句适中）",
          "visual_scenes": ["画面要点1", ...],
          "bgm_style": "搞笑/反转/悬疑"
        }

    Downstream use of the returned dict:
        - narration → passed to TTS (generate_tts_audio) and to video assembler (script_text)
          for both voiceover and on-screen subtitles; same text for both.
        - visual_scenes → passed to DeepSeek (optimize_visual_prompt) for Kling prompts.

    Args:
        topic: The selected hot topic from Douyin trends.
        timeout: Optional requests timeout in seconds (default 90; 豆包联网生成约 30–60s).

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

    # 若是 Kimi 模型，先用 Exa 补一批事实材料，供后续脚本生成参考
    exa_facts = ""
    if "kimi" in model_id.lower():
        try:
            exa_facts = fetch_topic_facts_with_exa(topic)
        except Exception as exc:  # pragma: no cover
            print(f"[ExaAPIError] 获取话题事实失败: {exc}")
            exa_facts = ""

    system_prompt = (
        "你的角色是「联网事实与素材采集员」，后端已开启 enable_web_search。"
        "核心任务：针对给定的抖音热点话题，通过联网搜索获取客观、可验证的事实与素材，"
        "并整理成结构化输出，供后续环节做镜头设计与旁白撰写。\n\n"
        "【以下是已为你整理好的外部事实材料（来自 Exa 实时搜索），请优先基于这些材料创作，"
        "如与你自己的搜索结果不一致，以这些材料为准，避免凭空编造：】\n"
        f"{exa_facts if exa_facts else '（未能额外获取到外部事实，请仅在确信的范围内回答，不要编造具体事实。）'}\n\n"
        "【必须执行的步骤】\n"
        "1. 联网搜索：针对该话题搜索最新报道、数据、时间线、当事人/机构、网友热议点、官方说法等。\n"
        "2. 只采用可验证的客观信息：人物、时间、地点、数字、事件经过、结果等尽量来自搜索结果；"
        "若某条信息搜不到或无法核实，不要编造，可在表述中保留「据报」「有说法称」等限定。\n"
        "3. 当前时间背景为 2026 年：理解为「故事发生在 2026 年左右」即可，"
        "避免写出具体的日历日期或精确时间（例如「2026年3月12日上午九点」），"
        "如需提到时间，请用「最近」「这段时间」「不久前」等模糊表达，不要编造具体年月日和几点几分。\n\n"
        "【输出要求】严格按下列 JSON 结构输出，且只输出 JSON，无额外说明或 Markdown：\n\n"
        "{\n"
        '  "title": "视频标题",\n'
        '  "narration": "旁白文案（约30秒，口语化、搞笑幽默；内容须基于你搜到的客观事实，可含具体数据、时间、人物）",\n'
        '  "visual_scenes": ["画面要点1", "画面要点2", "画面要点3", "画面要点4", "画面要点5"],\n'
        '  "bgm_style": "搞笑/反转/悬疑"\n'
        "}\n\n"
        "【narration 的用途与写法】\n"
        "narration 是你输出的「旁白文案」，会原样传递给后续流程，同时用于：① 语音合成（TTS 配音）；② 画面上的字幕。"
        "因此必须按「适合配音 + 字幕」的写法：用句号、感叹号、问号明确断句，每句长度适中（建议 7～20 字一句），"
        "便于下游按句切字幕或按字打轴，观众听和看的是同一份文案。\n\n"
        "字段说明：\n"
        "  - title：基于该话题与搜索到的最新热梗/讨论角度拟标题，不泄露个人隐私。\n"
        "  - narration：整段旁白，与 5 个画面要点顺序对应；尽量塞入你搜到的客观事实（数据、时间、人物、结果），保持口语化和幽默感；"
        "断句清晰、每句不宜过长，以便直接用于配音与字幕。\n"
        "  - visual_scenes：共 5 条。每条是基于「搜索到的客观事实」提炼出的画面要点/素材点（例如：某场景、某事件瞬间、某数据对应的画面联想），"
        "    不要求你写完整分镜剧本，具体镜头设计由下游完成；但要点之间要有逻辑顺序（起因→发展→转折或递进），便于后续做成连贯小故事。\n"
        "  - bgm_style：根据话题情绪选一个风格。\n\n"
        "请确保返回合法 JSON，字段名为英文，且不在 JSON 外添加任何文字。"
    )

    user_prompt = (
        f"当前热点话题：{topic.strip()}。（当前时间：2026年）\n"
        "请先联网搜索该话题的客观事实、最新动态、数据与热议点，再基于搜索结果填写上述 JSON。"
        "visual_scenes 共 5 条，每条为基于事实提炼的画面要点。"
        "narration 将原样用于配音与字幕，请融入具体事实、断句清晰、每句长度适中。时间表述一律用 2026 年。"
    )

    # Kimi 等使用 Ark Responses API（web_search 工具）；豆包使用 Chat Completions API。
    # 为了兼容不同流式事件格式，这里统一使用非流式 Responses 调用，避免解析 SSE 事件。
    use_responses_api = "kimi" in model_id.lower()
    if use_responses_api:
        try:
            data = _call_ark_responses_with_web_search(
                model_id,
                user_prompt,
                system_prefix=system_prompt,
                timeout=timeout,
                stream=False,
            )
            raw_text = _extract_ark_output_text(data)
        except ArkAPIError as exc:
            # 某些账号 / 模型可能暂不支持 Responses 接口，容错退回 Chat Completions，至少保证流程可跑通。
            print(
                f"[ArkAPIError] Kimi Responses 调用失败，将回退到 Chat Completions：{exc}"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            data = _call_doubao_chat_completions(
                messages,
                timeout=timeout,
                model_id=model_id,
                temperature=0.5,
                enable_web_search=False,  # 我们已经通过 Exa 提供事实材料，这里不再强行开启内置搜索
            )
            raw_text = _extract_doubao_chat_content(data)
    else:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        data = _call_doubao_chat_completions(
            messages,
            timeout=timeout,
            model_id=model_id,
            temperature=0.7,
            enable_web_search=True,
        )
        raw_text = _extract_doubao_chat_content(data)

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
        "You are an expert prompt engineer specialized in dynamic multi‑shot text‑to‑video generation "
        "for models like Kling. Given a list of Chinese short‑video scenes, you must convert each scene "
        "into a high‑quality English prompt that drives MOVEMENT and CINEMATIC ACTION on screen, "
        "not static slideshow images.\n\n"
        "NARRATIVE & LOGIC: Treat the list as one short story. Scenes must connect logically—cause and effect, "
        "same character journey, or clear before/after. Each shot should feel like a story beat (setup, tension, "
        "reaction, payoff), not isolated images. Keep continuity in character, place, and mood so the video feels "
        "like one coherent mini‑narrative.\n\n"
        "DEPTH & STORY IN EACH SHOT: Every prompt should have a sense of depth and story:\n"
        "- 立体感 (three‑dimensionality): Describe foreground / mid / background layers, depth of field, "
        "spatial clarity (where the character is in the room or space), so the frame feels like a real place.\n"
        "- 故事感 (story feel): The shot should imply a moment in a story—a character doing something with purpose, "
        "reacting to something, or the environment changing (e.g. light shift, crowd reaction). Use vivid verbs: "
        "walking, turning, rushing, laughing, spinning, looking back, etc.\n\n"
        "CAMERA: You MUST use varied, cinematic camera movement. Where it fits the story, prefer:\n"
        "- Camera orbiting or circling around the character (orbit around subject, arc shot, camera circles the person, "
        "slow 360 around the character), so the audience feels they are moving around the scene.\n"
        "- Also use: dolly in/out, push‑in, pull‑back, pan, tilt, handheld, tracking shot, following the character.\n"
        "Vary shot types across the list: wide establishing, medium, close‑up, over‑the‑shoulder, so the edit feels "
        "dynamic and the character stays the visual anchor.\n\n"
        "For each scene, you MUST:\n"
        "1) Start with a stable visual anchor so the main subject is consistent across shots (e.g. same person "
        "with brief appearance note, or same location).\n"
        "2) Describe concrete action, environment and emotion for THIS scene only, with clear depth (layers, space) "
        "and a story moment (what the character is doing or feeling).\n"
        "3) Add Kling‑friendly details: ultra realistic, 4k, dynamic lighting, shallow depth of field, cinematic, "
        "shot type, camera movement (including orbit/circle around character when it fits), composition, atmosphere.\n"
        "4) Ensure each prompt implies temporal flow (e.g. \"as the camera orbits around her\", \"while she turns\", "
        "\"the camera pushes in as he reacts\").\n"
        "5) Do NOT describe any on‑screen text, signs, captions, labels, or text overlays (to avoid garbled text).\n"
        "6) Keep content safe and suitable for mainstream short‑video platforms.\n\n"
        "FORMAT: Return ONLY a JSON array of strings, one string per input scene, same order. "
        "Each string at most 512 characters. No extra text or Markdown outside the JSON array."
    )

    user_prompt = (
        "下面是一个中文短视频分镜列表，请逐条转化为适合可灵 Kling 的英文 Prompt。\n\n"
        "要求：\n"
        "- 场景之间要有明确逻辑与故事线（起因→发展→转折/结果），人物与场景前后统一，像一条完整小故事。\n"
        "- 每个镜头要有立体感（前景/中景/背景层次、空间关系、景深）和故事感（人物在做什么、情绪或反应、环境变化）。\n"
        "- 镜头语言要丰富：适当使用「镜头围绕人物转动」（orbit around the character / camera circles the subject / arc shot）"
        "以及推拉摇移、跟拍等，让画面动起来；景别要有变化（远景、中景、特写等）。\n"
        "- 每条只描述一个分镜，包含：主体、场景与环境、动作、氛围，以及镜头运动与构图；不要描述画面中的文字。\n"
        "- 每条英文 Prompt 不超过 512 字符，精简有力；整体风格真实、电影感，适合短视频。\n\n"
        f"中文分镜列表（JSON 数组）：\n{scenes_json}"
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

    # 部分模型会返回 ```json\n...\n``` 包裹的内容，解析前先剥掉
    text = raw_content.strip()
    if text.startswith("```"):
        start = text.find("\n")
        if start != -1:
            text = text[start + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
        text = text.strip()

    try:
        parsed = json.loads(text)
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

