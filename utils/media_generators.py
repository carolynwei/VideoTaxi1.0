"""
多媒体生成层：可灵视频生成与豆包语音 2.0 TTS。

- generate_kling_video(prompt): 提交文生视频任务并轮询直至完成，返回视频下载 URL。
- generate_tts_audio(text, output_path): 调用豆包语音合成模型 2.0（WebSocket 双向流）合成音频并保存，可选返回字级时间戳供字幕打轴。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


def _config(key: str) -> str:
    """Read config from environment (and .env if python-dotenv is used)."""
    v = os.getenv(key, "").strip()
    if not v:
        raise KeyError(f"Required config '{key}' is not set.")
    return v


# --- 可灵 Turbo / Kling 文生视频 ---
# 官方调用域名：https://api-beijing.klingai.com，鉴权使用 AccessKey + SecretKey 生成 JWT
KLING_BASE_URL = os.getenv("KLING_BASE_URL", "https://api-beijing.klingai.com").rstrip("/")
KLING_TEXT2VIDEO_URL = f"{KLING_BASE_URL}/v1/videos/text2video"
KLING_POLL_INTERVAL = 10
KLING_POLL_TIMEOUT = 900  # 可灵多镜头生成较慢，默认等 15 分钟


class KlingAPIError(RuntimeError):
    """可灵 API 请求失败或返回错误时抛出。"""


def _kling_headers() -> Dict[str, str]:
    """
    可灵接口鉴权：用 AccessKey + SecretKey 按 JWT (HS256) 生成 Token，
    再组装为 Authorization: Bearer <token>。详见官方「接口鉴权」文档。
    """
    try:
        import jwt
    except ImportError as exc:
        raise KlingAPIError(
            "可灵鉴权需要 PyJWT，请执行: pip install PyJWT"
        ) from exc

    ak = _config("KLING_ACCESS_KEY")
    sk = _config("KLING_SECRET_KEY")
    headers_jwt = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 1800,
        "nbf": int(time.time()) - 5,
    }
    token = jwt.encode(payload, sk, headers=headers_jwt)
    if hasattr(token, "decode"):
        token = token.decode("utf-8")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def generate_kling_video(
    prompt: str,
    *,
    aspect_ratio: str = "9:16",
    duration: int = 5,
    poll_interval: int = KLING_POLL_INTERVAL,
    timeout: int = KLING_POLL_TIMEOUT,
) -> str:
    """
    使用可灵 Turbo 2.5 文生视频 API：提交任务后轮询状态，成功后返回视频下载 URL。

    Args:
        prompt: 文本提示词（正向）。
        aspect_ratio: 宽高比，如 "9:16", "16:9", "1:1"。
        duration: 视频时长（秒），通常 5 或 10。
        poll_interval: 轮询间隔（秒）。
        timeout: 最大等待时间（秒），例如 300 表示 5 分钟。

    Returns:
        视频文件的下载 URL。

    Raises:
        ValueError: prompt 为空。
        KlingAPIError: 请求失败、任务失败或超时。
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")

    headers = _kling_headers()
    body = {
        "prompt": prompt.strip(),
        "aspect_ratio": aspect_ratio,
        "duration": duration,
    }

    try:
        resp = requests.post(
            KLING_TEXT2VIDEO_URL,
            headers=headers,
            json=body,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise KlingAPIError(f"可灵提交任务请求失败: {exc}") from exc

    if not resp.ok:
        raise KlingAPIError(
            f"可灵 API 返回 HTTP {resp.status_code}: {resp.text}"
        )

    try:
        data: Dict[str, Any] = resp.json()
    except json.JSONDecodeError as exc:
        raise KlingAPIError(f"可灵 API 返回非 JSON: {resp.text}") from exc

    task_id = data.get("data", {}).get("task_id") or data.get("task_id")
    if not task_id:
        raise KlingAPIError(
            f"可灵 API 响应中未包含 task_id。响应: {data}"
        )

    query_url = f"{KLING_TEXT2VIDEO_URL.rstrip('/')}/{task_id}"
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            q = requests.get(query_url, headers=headers, timeout=30)
        except requests.RequestException as exc:
            raise KlingAPIError(f"可灵查询任务请求失败: {exc}") from exc

        if not q.ok:
            raise KlingAPIError(
                f"可灵查询任务返回 HTTP {q.status_code}: {q.text}"
            )

        try:
            qdata: Dict[str, Any] = q.json()
        except json.JSONDecodeError:
            raise KlingAPIError(f"可灵查询返回非 JSON: {q.text}")

        # 兼容 data.task_status / data.video?.url 或顶层 task_status / result
        inner = qdata.get("data") or qdata
        status = (
            inner.get("task_status")
            or inner.get("status")
            or qdata.get("task_status")
        )
        status = (status or "").lower()

        if status in ("completed", "success", "succeed"):
            url = (
                (inner.get("video") or {})
                if isinstance(inner.get("video"), dict)
                else {}
            )
            if isinstance(url, dict):
                url = url.get("url") or url.get("video_url") or ""
            elif not isinstance(url, str):
                url = ""
            if not url:
                url = (
                    inner.get("video_url")
                    or inner.get("url")
                    or qdata.get("video_url")
                    or qdata.get("url")
                    or ""
                )
            if url:
                return url
            raise KlingAPIError(
                f"可灵任务已完成但响应中无视频 URL。响应: {qdata}"
            )

        if status in ("failed", "error", "canceled", "cancelled"):
            msg = (
                inner.get("message")
                or inner.get("error")
                or qdata.get("message")
                or qdata.get("error")
                or str(qdata)
            )
            raise KlingAPIError(f"可灵视频任务失败: {msg}")

        time.sleep(poll_interval)

    raise KlingAPIError(
        f"可灵视频任务在 {timeout} 秒内未完成，task_id={task_id}"
    )


def generate_kling_multishot(
    prompts: list[str],
    *,
    aspect_ratio: str = "9:16",
    total_duration: int = 15,
    mode: str = "pro",
    sound: str = "on",
    model_name: str = "kling-v3",
    poll_interval: int = KLING_POLL_INTERVAL,
    timeout: int = KLING_POLL_TIMEOUT,
) -> str:
    """
    使用可灵多镜头文生视频接口，基于 multi_prompt 生成多镜头视频。

    Args:
        prompts: 每个分镜对应的一条英文 Prompt，顺序即为分镜顺序。
        aspect_ratio: 宽高比，如 "9:16", "16:9", "1:1"。
        total_duration: 视频总时长（秒），将平均分配到各分镜，余数加在前几镜。
        mode: 生成模式，std / pro。
        sound: 是否生成声音，on / off。
        model_name: 可灵模型名称，例如 "kling-v3"。
        poll_interval: 轮询间隔（秒）。
        timeout: 最大等待时间（秒）。

    Returns:
        最终视频文件的下载 URL。

    Raises:
        ValueError: prompts 为空或 total_duration 非法。
        KlingAPIError: 请求失败、任务失败或超时。
    """
    if not prompts:
        raise ValueError("prompts must be a non-empty list of strings.")
    if total_duration <= 0:
        raise ValueError("total_duration must be a positive integer.")

    clean_prompts = [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
    if not clean_prompts:
        raise ValueError("prompts must contain at least one non-empty string.")
    # 可灵 API：multi_prompt 数量必须在 1～6 之间
    clean_prompts = clean_prompts[:6]

    n = len(clean_prompts)
    base = max(total_duration // n, 1)
    durations = [base] * n
    remaining = total_duration - base * n
    idx = 0
    while remaining > 0:
        durations[idx] += 1
        remaining -= 1
        idx = (idx + 1) % n

    # 可灵 API 要求每个分镜 prompt 长度不超过 512 字符
    KLING_PROMPT_MAX_LEN = 512
    multi_prompt = [
        {
            "index": i + 1,
            "prompt": (clean_prompts[i][:KLING_PROMPT_MAX_LEN] if len(clean_prompts[i]) > KLING_PROMPT_MAX_LEN else clean_prompts[i]),
            "duration": str(durations[i]),
        }
        for i in range(n)
    ]

    headers = _kling_headers()
    body = {
        "model_name": model_name,
        "multi_shot": True,
        "shot_type": "customize",
        "multi_prompt": multi_prompt,
        "duration": str(total_duration),
        "mode": mode,
        "sound": sound,
        "aspect_ratio": aspect_ratio,
    }

    try:
        resp = requests.post(
            KLING_TEXT2VIDEO_URL,
            headers=headers,
            json=body,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise KlingAPIError(f"可灵多镜头任务提交失败: {exc}") from exc

    if not resp.ok:
        raise KlingAPIError(
            f"可灵 API 返回 HTTP {resp.status_code}: {resp.text}"
        )

    try:
        data: Dict[str, Any] = resp.json()
    except json.JSONDecodeError as exc:
        raise KlingAPIError(f"可灵 API 返回非 JSON: {resp.text}") from exc

    task_id = data.get("data", {}).get("task_id") or data.get("task_id")
    if not task_id:
        raise KlingAPIError(
            f"可灵 API 响应中未包含 task_id。响应: {data}"
        )

    query_url = f"{KLING_TEXT2VIDEO_URL.rstrip('/')}/{task_id}"
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            q = requests.get(query_url, headers=headers, timeout=30)
        except requests.RequestException as exc:
            raise KlingAPIError(f"可灵查询任务请求失败: {exc}") from exc

        if not q.ok:
            raise KlingAPIError(
                f"可灵查询任务返回 HTTP {q.status_code}: {q.text}"
            )

        try:
            qdata: Dict[str, Any] = q.json()
        except json.JSONDecodeError:
            raise KlingAPIError(f"可灵查询返回非 JSON: {q.text}")

        inner = qdata.get("data") or qdata
        status = (
            inner.get("task_status")
            or inner.get("status")
            or qdata.get("task_status")
        )
        status = (status or "").lower()

        if status in ("completed", "success", "succeed"):
            result = inner.get("task_result") or inner
            videos = result.get("videos") if isinstance(result, dict) else None
            url: str = ""
            if isinstance(videos, list) and videos:
                first = videos[0]
                if isinstance(first, dict):
                    url = (
                        first.get("url")
                        or first.get("video_url")
                        or ""
                    )
            if not url:
                url = (
                    inner.get("video_url")
                    or inner.get("url")
                    or qdata.get("video_url")
                    or qdata.get("url")
                    or ""
                )
            if url:
                return url
            raise KlingAPIError(
                f"可灵多镜头任务已完成但响应中无视频 URL。响应: {qdata}"
            )

        if status in ("failed", "error", "canceled", "cancelled"):
            msg = (
                inner.get("task_status_msg")
                or inner.get("message")
                or inner.get("error")
                or qdata.get("message")
                or qdata.get("error")
                or str(qdata)
            )
            raise KlingAPIError(f"可灵多镜头视频任务失败: {msg}")

        time.sleep(poll_interval)

    raise KlingAPIError(
        f"可灵多镜头视频任务在 {timeout} 秒内未完成，task_id={task_id}"
    )


# --- MiniMax 文生视频（Hailuo 系列）---

MINIMAX_API_BASE = os.getenv("MINIMAX_API_BASE", "https://api.minimax.io").rstrip("/")
MINIMAX_VIDEO_GEN_URL = f"{MINIMAX_API_BASE}/v1/video_generation"
MINIMAX_VIDEO_QUERY_URL = f"{MINIMAX_API_BASE}/v1/query/video_generation"
MINIMAX_FILES_RETRIEVE_URL = f"{MINIMAX_API_BASE}/v1/files/retrieve"


class MinimaxAPIError(RuntimeError):
    """MiniMax 视频生成 API 请求失败或返回错误时抛出。"""


def _minimax_api_key() -> str:
    """
    读取 MiniMax API Key。

    优先从 Streamlit secrets["ANTHROPIC_API_KEY"]，否则环境变量 ANTHROPIC_API_KEY / MINIMAX_API_KEY。
    """
    try:
        import streamlit as _st  # type: ignore

        if hasattr(_st, "secrets") and _st.secrets:
            v = (_st.secrets.get("ANTHROPIC_API_KEY") or "").strip()
            if v:
                return v
    except Exception:
        pass

    api_key = (
        os.getenv("ANTHROPIC_API_KEY", "").strip()
        or os.getenv("MINIMAX_API_KEY", "").strip()
    )
    if not api_key:
        raise KeyError("MiniMax API 需要 ANTHROPIC_API_KEY 或 MINIMAX_API_KEY。")
    return api_key


def generate_minimax_video(
    prompt: str,
    *,
    model: str = "MiniMax-Hailuo-2.3",
    duration: int = 6,
    resolution: str = "768P",
    poll_interval: int = 8,
    timeout: int = 900,
) -> str:
    """
    使用 MiniMax 文生视频 API 生成单段视频，返回下载 URL。

    目前默认使用 MiniMax-Hailuo-2.3，时长 6 秒，分辨率 768P。
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt must be a non-empty string.")

    api_key = _minimax_api_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body: Dict[str, Any] = {
        "model": model,
        "prompt": prompt[:2000],
        "duration": duration,
        "resolution": resolution,
        "prompt_optimizer": True,
        "fast_pretreatment": True,
    }

    try:
        resp = requests.post(
            MINIMAX_VIDEO_GEN_URL,
            headers=headers,
            json=body,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise MinimaxAPIError(f"MiniMax 文生视频创建任务失败: {exc}") from exc

    try:
        data: Dict[str, Any] = resp.json()
    except json.JSONDecodeError as exc:
        raise MinimaxAPIError(
            f"MiniMax 文生视频创建任务返回非 JSON: {resp.text}"
        ) from exc

    base_resp = data.get("base_resp") or {}
    if base_resp.get("status_code") != 0:
        raise MinimaxAPIError(
            f"MiniMax 创建任务失败: {base_resp.get('status_msg') or base_resp}"
        )

    task_id = (data.get("task_id") or "").strip()
    if not task_id:
        raise MinimaxAPIError(f"MiniMax 创建任务成功但未返回 task_id: {data}")

    # 轮询任务状态
    start = time.time()
    while time.time() - start < timeout:
        try:
            qresp = requests.get(
                MINIMAX_VIDEO_QUERY_URL,
                headers=headers,
                params={"task_id": task_id},
                timeout=30,
            )
        except requests.RequestException as exc:
            raise MinimaxAPIError(f"MiniMax 查询任务失败: {exc}") from exc

        try:
            qdata: Dict[str, Any] = qresp.json()
        except json.JSONDecodeError as exc:
            raise MinimaxAPIError(
                f"MiniMax 查询任务返回非 JSON: {qresp.text}"
            ) from exc

        qbase = qdata.get("base_resp") or {}
        if qbase.get("status_code") != 0:
            raise MinimaxAPIError(
                f"MiniMax 查询任务失败: {qbase.get('status_msg') or qbase}"
            )

        status = (qdata.get("status") or "").lower()
        if status in ("preparing", "queueing", "processing"):
            time.sleep(poll_interval)
            continue
        if status == "fail":
            raise MinimaxAPIError(f"MiniMax 视频任务失败: {qdata}")
        if status == "success":
            file_id = (qdata.get("file_id") or "").strip()
            if not file_id:
                raise MinimaxAPIError(
                    f"MiniMax 视频任务成功但未返回 file_id: {qdata}"
                )
            # 获取下载链接
            try:
                fresp = requests.get(
                    MINIMAX_FILES_RETRIEVE_URL,
                    headers=headers,
                    params={"file_id": file_id},
                    timeout=30,
                )
            except requests.RequestException as exc:
                raise MinimaxAPIError(
                    f"MiniMax 获取文件信息失败: {exc}"
                ) from exc

            try:
                fdata: Dict[str, Any] = fresp.json()
            except json.JSONDecodeError as exc:
                raise MinimaxAPIError(
                    f"MiniMax 获取文件信息返回非 JSON: {fresp.text}"
                ) from exc

            fbase = fdata.get("base_resp") or {}
            if fbase.get("status_code") != 0:
                raise MinimaxAPIError(
                    f"MiniMax 获取文件信息失败: {fbase.get('status_msg') or fbase}"
                )

            file_obj = fdata.get("file") or {}
            url = (file_obj.get("download_url") or "").strip()
            if not url:
                raise MinimaxAPIError(
                    f"MiniMax 文件信息中无 download_url: {fdata}"
                )
            return url

        # 未知状态，稍后重试
        time.sleep(poll_interval)

    raise MinimaxAPIError(
        f"MiniMax 视频任务在 {timeout} 秒内未完成，task_id={task_id}"
    )


# --- 火山引擎 TTS（豆包语音合成模型 2.0 WebSocket）---

VOLC_TTS_WS_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
VOLC_TTS_RESOURCE_ID = "seed-tts-2.0"

# 豆包语音 2.0 常用音色（文档「豆包语音合成模型2.0」与「视频配音」角色）
TTS_SPEAKER_FUNNY = "zh_male_taocheng_uranus_bigtts"  # 小天 2.0，默认搞笑/短视频男声

TTS_SPEAKER_PRESETS: Dict[str, str] = {
    # 通用 2.0 音色
    "小天 2.0（男，中配风，默认）": "zh_male_taocheng_uranus_bigtts",
    "Vivi 2.0（女，可爱一点）": "zh_female_vv_uranus_bigtts",
    "小何 2.0（女，温柔一点）": "zh_female_xiaohe_uranus_bigtts",
    "云舟 2.0（男，成熟稳重）": "zh_male_m191_uranus_bigtts",
    # 视频配音角色音色（saturn 系列）
    "儒雅逸辰（男，儒雅配音）": "zh_male_ruyayichen_saturn_bigtts",
    "可爱女生（女，角色扮演）": "saturn_zh_female_keainvsheng_tob",
    "调皮公主（女，角色扮演）": "saturn_zh_female_tiaopigongzhu_tob",
    "爽朗少年（男，角色扮演）": "saturn_zh_male_shuanglangshaonian_tob",
    "天才同桌（男，角色扮演）": "saturn_zh_male_tiancaitongzhuo_tob",
    "知性灿灿（女，角色扮演）": "saturn_zh_female_cancan_tob",
}

# 二进制协议：Event 码（与文档一致）
_EVT_START_CONNECTION = 1
_EVT_FINISH_CONNECTION = 2
_EVT_CONNECTION_STARTED = 50
_EVT_CONNECTION_FAILED = 51
_EVT_START_SESSION = 100
_EVT_FINISH_SESSION = 102
_EVT_TASK_REQUEST = 200
_EVT_SESSION_STARTED = 150
_EVT_SESSION_FINISHED = 152
_EVT_SESSION_FAILED = 153
_EVT_TTS_RESPONSE = 352
_EVT_TTS_SENTENCE_END = 351
_EVT_TTS_SUBTITLE = 353  # 文档：开启字幕后多次返回 TTSSubtitle（以常见扩展码占位，若不符再改）


class VolcTTSError(RuntimeError):
    """火山引擎 TTS 请求失败或返回错误时抛出。"""


def _volc_tts_credentials() -> Tuple[str, str]:
    """豆包语音 2.0：使用 VOLC_APPID 与 VOLC_ACCESS_TOKEN（控制台豆包语音产品下获取）。"""
    try:
        import streamlit as _st
        if hasattr(_st, "secrets") and _st.secrets:
            app_id = (_st.secrets.get("VOLC_APPID") or "").strip()
            token = (_st.secrets.get("VOLC_ACCESS_TOKEN") or "").strip()
            if app_id and token:
                return app_id, token
    except Exception:
        pass
    app_id = os.getenv("VOLC_APPID", "").strip()
    token = os.getenv("VOLC_ACCESS_TOKEN", "").strip()
    if not app_id or not token:
        raise KeyError(
            "豆包语音 2.0 需要 VOLC_APPID 与 VOLC_ACCESS_TOKEN（火山控制台-豆包语音）。"
        )
    return app_id, token


def _pack_volc_tts_frame(event: int, session_id: Optional[str], payload: Optional[bytes]) -> bytes:
    """组装豆包语音 WebSocket 二进制请求帧（大端）。"""
    import struct
    header = bytes([0x11, 0x14, 0x10, 0x00])  # v1, Full-client request with event, JSON, no compression
    buf = header + struct.pack(">I", event & 0xFFFFFFFF)
    if session_id is not None:
        sid = session_id.encode("utf-8")
        buf += struct.pack(">I", len(sid)) + sid
    if payload is not None:
        buf += struct.pack(">I", len(payload)) + payload
    return buf


def _parse_volc_tts_frame(data: bytes) -> Tuple[int, Optional[str], Optional[bytes], Optional[Dict[str, Any]]]:
    """
    解析服务端一帧。返回 (event, session_id, raw_payload, json_payload)。
    """
    import struct
    if len(data) < 8:
        return (-1, None, None, None)
    msg_type = data[1]
    event = struct.unpack_from(">I", data, 4)[0]
    pos = 8
    session_id: Optional[str] = None
    if len(data) >= pos + 4:
        sid_len = struct.unpack_from(">I", data, pos)[0]
        pos += 4
        if len(data) >= pos + sid_len:
            session_id = data[pos : pos + sid_len].decode("utf-8", errors="replace")
            pos += sid_len
    payload_bytes: Optional[bytes] = None
    if len(data) >= pos + 4:
        plen = struct.unpack_from(">I", data, pos)[0]
        pos += 4
        if len(data) >= pos + plen:
            payload_bytes = data[pos : pos + plen]
    json_payload: Optional[Dict[str, Any]] = None
    if payload_bytes and msg_type in (0x94, 0x90):
        try:
            json_payload = json.loads(payload_bytes.decode("utf-8"))
        except Exception:
            pass
    return (event, session_id, payload_bytes, json_payload)


def _tts_websocket_synthesize(
    text: str,
    *,
    app_id: str,
    access_token: str,
    speaker: str,
    audio_format: str,
    sample_rate: int,
    enable_subtitle: bool,
    timeout: float,
) -> Tuple[bytes, Optional[List[Dict[str, Any]]]]:
    """
    通过豆包语音 2.0 双向流 WebSocket 合成整段文本，返回 (音频字节, 字级时间戳列表或 None)。
    流程：StartConnection -> ConnectionStarted -> StartSession -> SessionStarted ->
    TaskRequest -> 收音频/字幕 -> FinishSession -> SessionFinished -> FinishConnection。
    """
    import struct
    import uuid
    import threading
    try:
        import websocket
    except ImportError as exc:
        raise VolcTTSError("豆包语音 WebSocket 需要 websocket-client，请执行: pip install websocket-client") from exc

    ws_url = VOLC_TTS_WS_URL
    header_dict = {
        "X-Api-App-Key": app_id,
        "X-Api-Access-Key": access_token,
        "X-Api-Resource-Id": VOLC_TTS_RESOURCE_ID,
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }
    audio_chunks: List[bytes] = []
    words_all: List[Dict[str, Any]] = []
    error_message: Optional[str] = None
    connection_started = False
    session_ready = False
    session_finished = False
    session_id = uuid.uuid4().hex[:12]

    def on_message(ws: Any, message: Any) -> None:
        nonlocal connection_started, session_ready, session_finished, error_message
        if isinstance(message, str):
            return
        data = message if isinstance(message, bytes) else bytes(message)
        if len(data) < 8:
            return
        msg_type = data[1]
        event = struct.unpack_from(">I", data, 4)[0]
        pos = 8
        if len(data) >= pos + 4:
            sid_len = struct.unpack_from(">I", data, pos)[0]
            pos += 4
            if len(data) >= pos + sid_len:
                pos += sid_len
        payload_bytes: Optional[bytes] = None
        if len(data) >= pos + 4:
            plen = struct.unpack_from(">I", data, pos)[0]
            pos += 4
            if len(data) >= pos + plen:
                payload_bytes = data[pos : pos + plen]

        if event == _EVT_CONNECTION_FAILED:
            try:
                obj = json.loads(payload_bytes.decode("utf-8")) if payload_bytes else {}
                error_message = obj.get("message", "Connection failed")
            except Exception:
                error_message = "Connection failed"
            return
        if event == _EVT_CONNECTION_STARTED:
            connection_started = True
            return
        if event == _EVT_SESSION_FAILED:
            try:
                obj = json.loads(payload_bytes.decode("utf-8")) if payload_bytes else {}
                error_message = obj.get("message", "Session failed")
            except Exception:
                error_message = "Session failed"
            return
        if event == _EVT_SESSION_STARTED:
            session_ready = True
            return
        if event == _EVT_TTS_RESPONSE and payload_bytes:
            audio_chunks.append(payload_bytes)
            return
        if event in (_EVT_TTS_SENTENCE_END, 353, 354):
            if payload_bytes:
                try:
                    obj = json.loads(payload_bytes.decode("utf-8"))
                    words = obj.get("words") or []
                    for w in words:
                        if isinstance(w, dict) and "word" in w:
                            words_all.append({
                                "word": str(w.get("word", "")),
                                "start_time": str(w.get("startTime", w.get("start_time", 0))),
                                "end_time": str(w.get("endTime", w.get("end_time", 0))),
                            })
                except Exception:
                    pass
            return
        if event == _EVT_SESSION_FINISHED:
            session_finished = True
            return

    def on_error(ws: Any, err: Any) -> None:
        nonlocal error_message
        if not err:
            return
        msg = str(err)
        # websocket-client 在正常关闭时可能抛出 "connection disconnected normally"，
        # 这种情况视为正常结束，不当作错误。
        if "connection disconnected normally" in msg.lower():
            return
        error_message = msg

    sock = websocket.WebSocketApp(
        ws_url,
        header=header_dict,
        on_message=on_message,
        on_error=on_error,
    )
    thread = threading.Thread(target=lambda: sock.run_forever())
    thread.daemon = True
    thread.start()

    deadline = time.time() + timeout
    while not sock.sock or not sock.sock.connected:
        if time.time() > deadline:
            raise VolcTTSError("豆包语音 WebSocket 建连超时")
        time.sleep(0.05)
    time.sleep(0.15)

    frame_conn = _pack_volc_tts_frame(_EVT_START_CONNECTION, None, b"{}")
    sock.send(frame_conn, opcode=websocket.ABNF.OPCODE_BINARY)
    while not connection_started and time.time() <= deadline:
        if error_message:
            raise VolcTTSError(f"豆包语音: {error_message}")
        time.sleep(0.05)
    if not connection_started:
        raise VolcTTSError("豆包语音 ConnectionStarted 未收到")

    payload_start = json.dumps({
        "user": {"uid": "1"},
        "event": _EVT_START_SESSION,
        "req_params": {
            "speaker": speaker,
            "audio_params": {
                "format": audio_format,
                "sample_rate": sample_rate,
                "enable_subtitle": enable_subtitle,
            },
        },
    }, ensure_ascii=False).encode("utf-8")
    frame_start = _pack_volc_tts_frame(_EVT_START_SESSION, session_id, payload_start)
    sock.send(frame_start, opcode=websocket.ABNF.OPCODE_BINARY)

    while not session_ready and time.time() <= deadline:
        if error_message:
            raise VolcTTSError(f"豆包语音: {error_message}")
        time.sleep(0.05)
    if not session_ready:
        raise VolcTTSError("豆包语音 StartSession 未就绪")

    payload_task = json.dumps({
        "user": {"uid": "1"},
        "event": _EVT_TASK_REQUEST,
        "req_params": {"text": text},
    }, ensure_ascii=False).encode("utf-8")
    frame_task = _pack_volc_tts_frame(_EVT_TASK_REQUEST, session_id, payload_task)
    sock.send(frame_task, opcode=websocket.ABNF.OPCODE_BINARY)

    time.sleep(0.3)
    frame_finish = _pack_volc_tts_frame(_EVT_FINISH_SESSION, session_id, b"{}")
    sock.send(frame_finish, opcode=websocket.ABNF.OPCODE_BINARY)

    while not session_finished and time.time() <= deadline:
        if error_message:
            raise VolcTTSError(f"豆包语音: {error_message}")
        time.sleep(0.05)
    time.sleep(0.2)
    frame_fin_conn = _pack_volc_tts_frame(_EVT_FINISH_CONNECTION, None, b"{}")
    sock.send(frame_fin_conn, opcode=websocket.ABNF.OPCODE_BINARY)
    time.sleep(0.2)
    sock.close()

    if error_message:
        raise VolcTTSError(f"豆包语音: {error_message}")

    audio = b"".join(audio_chunks)
    words_out = words_all if words_all else None
    return audio, words_out


def generate_tts_audio(
    text: str,
    output_path: str,
    *,
    speaker: str = TTS_SPEAKER_FUNNY,
    format: str = "mp3",
    enable_timestamp: bool = True,
    timeout: int = 60,
) -> Optional[Dict[str, Any]]:
    """
    调用豆包语音合成模型 2.0（WebSocket 双向流），将合成音频保存到 output_path，
    并可选返回字级时间戳（enable_subtitle），供字幕打轴使用。

    Args:
        text: 待合成文本。
        output_path: 本地保存路径，如 temp/audio.mp3。
        speaker: 发音人 ID，默认小天 2.0；可选 zh_female_vv_uranus_bigtts 等。
        format: 音频格式，mp3 / ogg_opus / pcm。
        enable_timestamp: 是否请求句级字时间戳（enable_subtitle）。
        timeout: 请求超时（秒）。

    Returns:
        若 enable_timestamp 且接口返回了 words，则返回 { "duration": 总秒数, "words": [...] }；
        否则返回 None。音频已写入 output_path。
    """
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string.")
    if not output_path or not output_path.strip():
        raise ValueError("output_path must be a non-empty string.")

    app_id, access_token = _volc_tts_credentials()
    sample_rate = 24000
    audio_bytes, words_list = _tts_websocket_synthesize(
        text.strip(),
        app_id=app_id,
        access_token=access_token,
        speaker=speaker,
        audio_format=format,
        sample_rate=sample_rate,
        enable_subtitle=enable_timestamp,
        timeout=float(timeout),
    )

    if not audio_bytes:
        raise VolcTTSError("豆包语音未返回音频数据")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(audio_bytes)

    ts_result: Optional[Dict[str, Any]] = None
    if words_list:
        duration = 0.0
        for w in words_list:
            try:
                end = float(w.get("end_time", 0))
                if end > duration:
                    duration = end
            except (TypeError, ValueError):
                pass
        ts_result = {"duration": duration, "words": words_list}
    return ts_result


# --- 测试与 temp 目录 ---

def _ensure_temp_dir() -> Path:
    """确保项目下的 temp/ 目录存在，用于存放生成的测试素材。"""
    d = Path(__file__).resolve().parent.parent / "temp"
    d.mkdir(parents=True, exist_ok=True)
    return d


__all__ = [
    "KlingAPIError",
    "generate_kling_video",
    "generate_kling_multishot",
    "MinimaxAPIError",
    "generate_minimax_video",
    "VolcTTSError",
    "generate_tts_audio",
    "TTS_SPEAKER_FUNNY",
    "TTS_SPEAKER_PRESETS",
    "_ensure_temp_dir",
]


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    _ensure_temp_dir()
    print("temp/ 目录已就绪。")

    print("\n=== 测试 generate_kling_video（仅提交+轮询，可能超时） ===")
    try:
        url = generate_kling_video(
            "A cute cat walking in the street, cinematic lighting.",
            timeout=60,
            poll_interval=10,
        )
        print("视频 URL:", url)
    except Exception as e:
        print("Kling 测试异常（可忽略，或检查 API Key/配额）:", e)

    print("\n=== 测试 generate_tts_audio ===")
    temp_dir = _ensure_temp_dir()
    out_audio = temp_dir / "audio.mp3"
    try:
        ts = generate_tts_audio(
            "今天天气真不错，适合拍一条搞笑短视频。",
            str(out_audio),
            enable_timestamp=True,
        )
        print("音频已保存:", out_audio)
        if ts:
            print("时长(秒):", ts.get("duration"))
            if ts.get("words"):
                print("字时间戳数量:", len(ts["words"]))
    except Exception as e:
        print("TTS 测试异常（请配置火山 TTS appkey/token）:", e)
