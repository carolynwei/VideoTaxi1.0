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

    # 为严肃新闻 / 深度科普模式优化的 Exa 检索：
    # - 使用 deep 模式做多视角深度搜索
    # - 扩大 num_results，覆盖更多权威来源
    # - 通过自然语言 query 明确要求“权威新闻/官方/研究机构”，过滤低质内容
    # - 使用 highlights 而不是整页全文，减小下游大模型的上下文压力
    query = (
        f"与「{topic}」相关的最新客观事实、权威新闻报道、官方公告和核心数据，"
        "优先返回来自政府机构、主流新闻媒体、研究机构和官方数据网站的页面，"
        "尽量避免内容农场、自媒体软文和纯营销广告，中文结果优先。"
    )

    try:
        # 使用 Exa 默认检索模式（非 deep），更接近“正常模式”
        results = client.search(
            query=query,
            type="deep",
            num_results=12,
            # 使用 highlights 返回每个页面中最相关的精华片段，避免整页长文造成 token 爆炸
            contents={
                "highlights": {
                    "max_characters": 600,
                }
            },
        )
    except Exception as exc:  # pragma: no cover - 网络/服务错误
        print(f"[ExaAPIError] 搜索失败: {exc}")
        return ""

    lines: List[str] = []
    for i, r in enumerate(results.results, start=1):
        title = (getattr(r, "title", "") or "").strip()
        url = (getattr(r, "url", "") or "").strip()
        # 新版 Exa 在使用 contents.highlights 时，会把高亮文本挂在 highlights 或 text 字段上
        highlights = getattr(r, "highlights", None)
        if isinstance(highlights, list) and highlights:
            snippet = " ".join(str(h).strip() for h in highlights if str(h).strip())
        else:
            snippet = (getattr(r, "text", "") or "").strip()

        if not (title or snippet):
            continue

        # 为 Kimi 提供尽量“干净”的事实材料，只保留标题和摘要内容，
        # 避免类似「[资料 1]」「来源：」等包装文本干扰后续 JSON 输出格式。
        if snippet:
            block = snippet
        else:
            block = title
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
    # 先检查是否存在显式错误信息
    error_obj = data.get("error")
    if isinstance(error_obj, Mapping):
        msg = error_obj.get("message") or error_obj.get("msg") or str(error_obj)
        raise ArkAPIError(f"Ark API error: {msg}")

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

    # 如果没有拿到任何文本，进一步针对常见「只返回 resp_xxx id」的情况给出更清晰的报错，
    # 避免误把 id 当成模型正文返回，导致后续 JSON 解析异常。
    resp_id = data.get("id")
    if not output and isinstance(resp_id, str):
        raise ArkAPIError(
            "Ark Responses API 调用未返回 output 内容，仅返回了任务 ID，"
            "当前实现不支持这种异步模式。已放弃本次 Responses 调用，将回退到 Chat Completions。"
        )

    # 走到这里说明结构完全不符合预期，直接抛错而不是返回任意字符串。
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
        # 提高 max_keyword，鼓励模型为 web_search 拆分出更多检索词，以覆盖更多信息源
        "tools": [{"type": "web_search", "max_keyword": 6}],
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
        # 对超时错误做更友好的提示，便于前端展示
        if "Read timed out" in str(exc) or "read timeout" in str(exc).lower():
            raise ArkAPIError(
                f"Doubao Chat API 请求在 {timeout} 秒内未返回结果（超时）。"
                "这通常是网络波动或模型负载过高导致的，请稍后重试一次。"
            ) from exc
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


def generate_script_with_search(query: str, *, timeout: int = 180) -> str:
    """
    Call Doubao model via Chat Completions API to generate a short-video script.

    The model and API key are read from configuration:
        - ARK_API_KEY or LAS_API_KEY
        - ARK_MODEL_ID

    Args:
        query: User's request text, e.g. "结合今日热点写一个短视频脚本".
        timeout: Optional requests timeout in seconds (default 180; 豆包/Kimi 联网生成约 30–120s).

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


def generate_video_script(
    topic: str,
    *,
    timeout: int = 180,
    exa_facts: Optional[str] = None,
) -> Dict[str, Any]:
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
        timeout: Optional requests timeout in seconds (default 180; 豆包/Kimi 联网生成约 30–120s).
        exa_facts: Optional external facts string collected via Exa. If provided,
            this value will be embedded到 system prompt 中；如果为 None，则由本函数
            内部调用 fetch_topic_facts_with_exa 自动获取。

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

    use_responses_api = "kimi" in model_id.lower()

    if use_responses_api:
        # Kimi 场景：不做资料收集，负责将热点话题转化为具有强烈现实共鸣的电影级短片脚本
        system_prompt = (
            "你是一名顶级的抖音短视频编剧兼视觉总监。你的任务是不做任何资料收集，"
            "直接把给定的【新闻热点话题】转化为一个具有极强现实共鸣、高级写实感且逻辑自洽的约 30 秒短故事脚本。\n\n"
            "特别提醒：这个剧本是写给「AI 视频模型」拍的。整体质感要像【电影级写实纪录片】或【高端品牌商业广告】——"
            "画面极简、写实、充满真实生活或商业社会的气息，但光影和构图要具有好莱坞级别的电影感。\n\n"
            "【核心创作策略：基于现实主义的美学风格】\n"
            "1. 科技/财经/商业类（如政策、磋商、企业动态、机器人、汽车、芯片）：\n"
            "   - 采用【高端商业大片 / 华尔街写实风】审美。\n"
            "   - 场景优先：现代玻璃幕墙写字楼、极简的高管办公室、深夜亮着台灯的办公桌、充满质感的皮革沙发。\n"
            "   - 表现手法：用人物的微表情（如皱眉、凝视、自信微笑）和高级道具（如钢笔、咖啡杯、平板电脑的冷光）来隐喻商业博弈，绝对禁止魔法或科幻光效。\n"
            "   ⚠️ 致命警告（实体锚定）：如果新闻涉及具体的科技产品（如宇树机器人），必须在画面中保留该实体的物理形态（如：流线型的银色机械狗、极具工业设计感的机械臂等），禁止把它替换成抽象光球、模糊能量团或任何魔法/超自然元素。\n"
            "2. 体育/竞技类（如中国男篮 vs 日本男篮）：\n"
            "   - 采用【高燃运动广告片】风格（参考耐克、阿迪达斯广告）。\n"
            "   - 强调现实物理质感：皮肤上的汗水特写、肌肉的紧绷、刺眼的场馆聚光灯、深呼吸的胸腔起伏。\n"
            "3. 生活/教育/社会民生类（如四六级成绩公布）：\n"
            "   - 采用【电影级现实主义】。真实、接地气，但打光极其讲究（如王家卫式的氛围光或日系文艺逆光）。\n"
            "   - 场景优先：真实的大学宿舍（但整理得极简整洁）、深夜的图书馆、清晨的街道、咖啡馆靠窗座位。\n"
            "   - 表现手法：用极其真实的动作引发共鸣（如紧张地握紧双手、看着远方如释重负地松了一口气）。\n\n"
            "【美学与动作限制（AI 视频物理学——必须严格遵守）】\n"
            "创作时必须默认画面将由 AI 视频模型（如可灵、MiniMax 等）生成，务必遵守以下硬性规则：\n"
            "1. 动作极简与真实原则：\n"
            "   - 动作必须符合现实物理规律。禁止飞天遁地、魔法发光等夸张设定。\n"
            "   - 每个镜头的动作控制在 1 个极简单动作：例如“缓慢抬起头”、“深吸一口气”、“握紧拳头”、“缓慢转身”、“端起咖啡杯”等。\n"
            "2. 场景纯净与质感原则：\n"
            "   - 场景描述要“极简、高端、电影光影”，拒绝杂乱、破旧、堆满小物件的无序空间。\n"
            "   - 优先使用：大面积留白、几何结构、侧逆光、轮廓光、雨后地面的反光等视觉语言。\n"
            "3. 单一主体原则：\n"
            "   - 每个画面尽量只突出一个核心主体（一个人 / 一个关键实体对象）。禁止复杂群戏。\n"
            "4. 文字与细节可见度限制（极其致命）：\n"
            "   - 绝对禁止依赖“屏幕里的小字 / 牌匾上的字 / 成绩单上的数字”等可读文字来讲故事。\n"
            "   - 必须用动作、光影变化和表情来传达信息（例如：不要写“屏幕上显示不及格”，要写“他看着前方，脸色瞬间苍白，脱力般靠在椅背上”）。\n"
            "5. 镜头调度简化：\n"
            "   - 镜头运动描述要简单、明确，如“缓慢推近(Dolly in)”、“特写(Close-up)”、“静止构图”。\n\n"
            "【叙事结构与文案要求】\n"
            "1. 结构紧凑：起（真实悬念） -> 承（情绪放大） -> 转（态度或视角的转变） -> 合（引人深思或极具力量的定格）。\n"
            "2. 旁白文案（narration）：极度口语化、接地气。像是一个成熟的讲述者在跟你聊天，有情绪起伏，多用短句、设问句。坚决不要播音腔或新闻复述。\n\n"
            "【输出格式硬性要求】严格按下列 JSON 结构输出，且只输出 JSON，无额外说明或 Markdown：\n"
            "{\n"
            '  "title": "视频标题（要有网感和现实痛点，可略带标题党）",\n'
            '  "narration": "旁白文案（约30秒，用讲真实故事或犀利点评的语气）",\n'
            '  "visual_scenes": ["镜头 1 画面要点", "镜头 2 画面要点", "镜头 3 画面要点", "镜头 4 画面要点", "镜头 5 画面要点"],\n'
            '  "bgm_style": "沉浸剧情/高燃踩点/情绪低吟 等中任选其一"\n'
            "}\n\n"
            "【输出格式防错守则】\n"
            "1. 只输出一个合法的 JSON 值，绝不允许在外部有任何文字、解释、Markdown 符号（如 ```json）。\n"
            "2. 字段名和字符串必须用英文双引号包裹。\n"
            "3. 严禁出现结尾多余逗号（trailing comma）。\n"
            "请确保最终返回的内容是语法严格合法的 JSON 文本。"
        )

        user_prompt = (
            f"当前抖音热点话题：【{topic.strip()}】。\n"
            "请把这个话题当成灵感内核，创作一个围绕该话题的现实主义短视频脚本。不要搞科幻或脱离实际的脑洞！\n"
            "要让人觉得这是真实发生的、极具质感的电影级画面。请扎根于现实生活、真实商业或体育场景进行延展。\n"
            "请按照 JSON 结构返回结果：visual_scenes 必须是 5 条符合「极简、写实、无文字」原则的画面描述；"
            "narration 写成口语化、情绪饱满的爆款旁白。"
        )
    else:
        # 非 Kimi 场景仍然可以使用事实材料与联网搜索的思路
        facts_text = exa_facts
        if facts_text is None:
            try:
                facts_text = fetch_topic_facts_with_exa(topic)
            except Exception as exc:  # pragma: no cover
                print(f"[ExaAPIError] 获取话题事实失败: {exc}")
                facts_text = ""

        system_prompt = (
            "你的角色是「联网事实与素材采集员」，后端已开启 enable_web_search。"
            "核心任务：针对给定的抖音热点话题，通过联网搜索获取客观、可验证的事实与素材，"
            "并整理成结构化输出，供后续环节做镜头设计与旁白撰写。\n\n"
            "【以下是已为你整理好的外部事实材料（来自 Exa 实时搜索），请优先基于这些材料创作，"
            "如与你自己的搜索结果不一致，以这些材料为准，避免凭空编造：】\n"
            f"{facts_text if facts_text else '（未能额外获取到外部事实，请仅在确信的范围内回答，不要编造具体事实。）'}\n\n"
            "【必须执行的步骤】\n"
            "1. 联网搜索：务必先多轮使用 web_search 工具，针对该话题拆分出若干子问题，"
            "用不同的中文/英文关键词搜索最新报道、数据、时间线、当事人/机构、网友热议点、官方说法等；"
            "优先选择权威新闻媒体、官方渠道和数据网站，忽略广告、营销软文和无来源的自媒体内容。\n"
            "2. 只采用可验证的客观信息：人物、时间、地点、数字、事件经过、结果等必须以多个搜索结果为依据，"
            "关键事实尽量做到至少 2 个以上独立来源相互印证；若某条信息搜不到或无法核实，不要编造，"
            "可在表述中保留「据报」「有说法称」等限定。\n"
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
            "【输出格式硬性要求——务必严格遵守】\n"
            "1. 你必须只输出一个 JSON 值，不允许在 JSON 外多输出任何文字、解释、注释、示例、Markdown 代码块等内容。\n"
            "2. JSON 顶层类型必须是对象，字段名一律使用英文双引号包裹。\n"
            "3. 所有字符串必须用双引号包裹，禁止使用单引号；禁止在 JSON 中添加注释。\n"
            "4. 禁止出现结尾多余逗号（trailing comma），数组和对象的最后一个元素后面都不能有逗号。\n"
            "5. 如果无法完全满足业务要求，也要返回一个语法上完全合法的 JSON，并在某些字段里用简短中文说明原因，而不是乱写格式。\n"
            "6. 严禁输出形如 ```json、``` 之类的 Markdown 包裹，只能输出纯 JSON 文本本身。\n"
            "7. 如果你想输出多段内容，一律合并进同一个 JSON 对象中，不要分多段输出。\n"
            "请确保最终返回的内容是语法严格合法的 JSON。"
        )

        user_prompt = (
            f"当前热点话题：{topic.strip()}。（当前时间：2026年）\n"
            "请先联网搜索该话题的客观事实、最新动态、数据与热议点，再基于搜索结果填写上述 JSON。"
            "visual_scenes 共 5 条，每条为基于事实提炼的画面要点。"
            "narration 将原样用于配音与字幕，请融入具体事实、断句清晰、每句长度适中。时间表述一律用 2026 年。"
        )

    # Kimi 等使用 Ark Responses API（web_search 工具）；豆包使用 Chat Completions API。
    # 为了兼容不同流式事件格式，这里统一使用非流式 Responses 调用，避免解析 SSE 事件。
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

    # ---------- 清洗与解析 JSON 输出，兼容 ```json 代码块与额外说明文字 ----------
    cleaned = (raw_text or "").strip()

    # 去掉 ``` 或 ```json 包裹
    if cleaned.startswith("```"):
        start = cleaned.find("\n")
        if start != -1:
            cleaned = cleaned[start + 1 :]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3].rstrip()
        cleaned = cleaned.strip()

    # 如果模型在 JSON 前后加了解释性文字，尝试从第一个 { 起做「括号配对」提取首个完整 JSON 对象，
    # 比简单的 first/last rfind 更健壮（可处理字符串内的 { } 以及嵌套对象）。
    def _extract_first_json_object(text: str) -> str:
        start = text.find("{")
        if start == -1:
            return text
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        # 如果没能完整配对，退回原始文本，后续解析会给出清晰报错
        return text

    json_candidate = _extract_first_json_object(cleaned)

    try:
        parsed: Dict[str, Any] = json.loads(json_candidate)
    except json.JSONDecodeError:
        # 退回尝试解析原始文本，以防误截断
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
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
        "You are an elite Prompt Engineer specialized in text-to-video models like MiniMax Hailuo and Kling. "
        "Your task is to translate a JSON array of Chinese trending news headlines and abstract events into a highly optimized JSON array of English video prompts.\n\n"
        "### CORE OBJECTIVE: VISUAL METAPHORS & B-ROLL\n"
        "News headlines are abstract. You MUST NOT generate literal newsrooms, news anchors, or screens with text. "
        "Instead, translate each news item into a concrete, text-free CINEMATIC B-ROLL or VISUAL METAPHOR.\n"
        "- Example 1 (Crypto/Finance): Instead of 'A report about crypto', generate 'A glowing digital vault absorbing golden holographic coins from a futuristic fiber-optic network, high-tech macro shot.'\n"
        "- Example 2 (Exam Scores): Instead of 'Scores announced', generate 'Close-up of a nervous student's hands pressing a glowing keyboard in a dark room, cinematic dramatic lighting.'\n"
        "- Example 3 (Political Talks): Instead of 'Meeting', generate 'Two businessmen in sharp suits shaking hands in silhouette against a massive window overlooking a modern cyberpunk cityscape, slow motion.'\n\n"
        "### 0. SPEED & KINETIC ENERGY (CRITICAL)\n"
        "Since each video clip is only 6 seconds, you MUST prioritize FAST-PACED and HIGH-ENERGY actions. "
        "Use verbs that imply speed and intensity such as: rushing, spinning rapidly, flicking, swiftly, bursting, slamming, snapping, dynamic transition. "
        "Every shot must have a clear 'start-to-end' kinetic energy shift to prevent a slow-motion feel (for example: calm → sudden burst → impact within 6 seconds).\n"
        "Within the 6-second window, design at least two distinct action beats or visual changes. "
        "Explicitly use patterns like 'First ..., then ...' or 'While ..., ...' so that time feels compressed and eventful.\n\n"
        "### PROMPT FORMULA\n"
        "Construct EVERY prompt strictly using this flow:\n"
        "[Camera Movement] + [Fast multi-stage Action] + [Concrete Visual Metaphor / Subject] + [Environment & Depth] + [Cinematic Style & Lighting] + [Anti-Gibberish Constraints]\n\n"
        "### 1. CAMERA & MOVEMENT (Dynamic & Premium)\n"
        "- Force premium commercial camera movement: 'fast dolly-in', 'quick zoom-out', 'hyper-lapse style movement', 'aggressive tracking shot', 'dynamic macro tracking shot', 'drone sweeping over', 'orbit around subject'.\n"
        "- The camera should almost never be static; even subtle shots must include panning, pushing, orbiting, or reframing.\n"
        "- The movement must match the news tone (e.g., fast and dynamic for sports, slow and tense but still kinetic for politics).\n"
        "- When appropriate, explicitly mention 'high-energy cinematography' or 'fast-paced editing style'.\n"
        "- When featuring VIPs or generic important figures, frequently use Over-The-Shoulder (OTS) shots, silhouettes, back shots, or close-ups of hands/objects to maintain a premium, mysterious cinematic feel without showing full faces.\n\n"
        "### 2. AESTHETICS BY CATEGORY\n"
        "- Tech/Finance: 'Futuristic, holographic glowing data streams, neon blue and gold lighting, sharp focus, 8k resolution, commercial macro photography.'\n"
        "- Sports (e.g., Basketball): 'High-contrast stadium lighting, sweat dripping, extreme slow motion, dynamic action, cinematic sports broadcast style.'\n"
        "- Politics/Economy: 'Serious moody lighting, shallow depth of field, sharp silhouettes, highly professional documentary style.'\n\n"
        "### 3. ANTI-TEXT & PURITY RULES (STRICT STRICT STRICT)\n"
        "- ABSOLUTELY NO readable text, signs, logos, chyrons, or captions.\n"
        "- NO newsroom desks or TV screens displaying text.\n"
        "- MUST append this exact negative constraint at the end of EVERY prompt: 'clean uncluttered background, devoid of typography, no text overlays, no random symbols, no visual noise, highly detailed.'\n\n"
        "### 5. IDENTITY TRANSLATION & CENSORSHIP AVOIDANCE (CRITICAL)\n"
        "- Video models CANNOT generate specific niche CEOs or politicians, and using political names will trigger API safety bans.\n"
        "- You MUST translate specific real-world names into generic cinematic archetypes.\n"
        "- Example 1 (Tech CEO): Translate '王兴兴' (Wang Xingxing) or '雷军' (Lei Jun) to 'A confident young Asian tech entrepreneur in a minimalist black t-shirt'.\n"
        "- Example 2 (Politician): Translate '德国总理' (German Chancellor) or '拜登' (Biden) to 'A distinguished elderly European statesman in a tailored suit' or 'A VIP government official'.\n"
        "- NEVER use real names of public figures, politicians, or specific brands in the output prompt. Use visual descriptions only.\n\n"
        "### OUTPUT FORMAT\n"
        "- Return ONLY a valid JSON array of strings.\n"
        "- Exactly one string per input scene, preserving the original order.\n"
        "- Maximum 512 characters per string.\n"
        "- NO conversational text, NO explanations, NO Markdown formatting outside the JSON array.\n\n"
        "### STRICT JSON RULES (MUST FOLLOW)\n"
        "1. You must output a single JSON value only, with no extra text, comments, examples, or Markdown code fences like ```json or ``` around it.\n"
        "2. The top-level JSON type must be an array; each element must be a string.\n"
        "3. All strings must use double quotes; never use single quotes inside the JSON syntax.\n"
        "4. Do NOT add trailing commas after the last element in arrays.\n"
        "5. If you cannot perfectly satisfy the content requirements, you still must return syntactically valid JSON and use short English explanations inside the strings themselves if needed.\n"
        "6. Never wrap the JSON array in any additional text or Markdown formatting; the output must be pure JSON."
    )

    user_prompt = (
        "Please strictly follow your system instructions to convert the following Chinese news headlines into English visual-metaphor video prompts.\n\n"
        f"Input Scenes (JSON Array):\n{scenes_json}"
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
        # 尽量容错：长度不一致时，打印告警并做截断/补齐，而不是直接中断整条生成链路。
        print(
            f"[DeepSeekAPIWarning] Output length ({len(parsed)}) does not match "
            f"input length ({len(chinese_scenes_list)}). Will normalize length."
        )
        if not parsed:
            raise DeepSeekAPIError(
                "Model output JSON list is empty; cannot build any visual prompts."
            )
        normalized: List[str] = []
        # 先按顺序匹配最短长度部分
        common_len = min(len(parsed), len(chinese_scenes_list))
        normalized.extend(parsed[:common_len])
        # 如果 DeepSeek 少返回了若干条，用最后一条 prompt 轻微变体补齐剩余镜头，保证长度一致
        if len(parsed) < len(chinese_scenes_list):
            last_prompt = parsed[-1]
            for idx in range(common_len, len(chinese_scenes_list)):
                normalized.append(f"{last_prompt} // variation for extra scene {idx + 1}")
        # 如果 DeepSeek 多返回了若干条，直接丢弃多余的，保留与输入场景数一致的部分
        return normalized

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

