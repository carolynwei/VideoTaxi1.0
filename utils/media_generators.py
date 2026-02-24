"""
多媒体生成层：可灵 Turbo 2.5 视频生成与火山引擎 TTS 语音合成。

- generate_kling_video(prompt): 提交文生视频任务并轮询直至完成，返回视频下载 URL。
- generate_tts_audio(text, output_path): 调用火山 TTS 合成音频并保存，可选返回字级时间戳。
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests


def _config(key: str) -> str:
    """Read config from environment (and .env if python-dotenv is used)."""
    v = os.getenv(key, "").strip()
    if not v:
        raise KeyError(f"Required config '{key}' is not set.")
    return v


# --- 可灵 Turbo 2.5 文生视频 ---

KLING_TEXT2VIDEO_URL = "https://api.klingai.com/v1/videos/text2video"
KLING_POLL_INTERVAL = 10
KLING_POLL_TIMEOUT = 300


class KlingAPIError(RuntimeError):
    """可灵 API 请求失败或返回错误时抛出。"""


def _kling_api_key() -> str:
    try:
        return _config("KLING_API_KEY")
    except KeyError:
        return _config("KLING_ACCESS_KEY")


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

    api_key = _kling_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
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


# --- 火山引擎 TTS ---

VOLC_TTS_INVOKE_URL = "https://sami.bytedance.com/api/v1/invoke"
VOLC_TTS_NAMESPACE = "TTS"
VOLC_TTS_VERSION = "v4"

# 适合搞笑短视频的火山引擎音色（中文趣味）
TTS_SPEAKER_FUNNY = "zh_male_sunwukong_clone2"


class VolcTTSError(RuntimeError):
    """火山引擎 TTS 请求失败或返回错误时抛出。"""


def _volc_tts_credentials() -> Tuple[str, str]:
    appkey = os.getenv("VOLC_TTS_APPKEY", "").strip() or os.getenv(
        "VOLC_APPID", ""
    ).strip()
    token = os.getenv("VOLC_TTS_TOKEN", "").strip() or os.getenv(
        "VOLC_ACCESS_TOKEN", ""
    ).strip()
    if not appkey or not token:
        raise KeyError(
            "请设置 VOLC_TTS_APPKEY 与 VOLC_TTS_TOKEN，或 VOLC_APPID 与 VOLC_ACCESS_TOKEN。"
        )
    return appkey, token


def generate_tts_audio(
    text: str,
    output_path: str,
    *,
    speaker: str = TTS_SPEAKER_FUNNY,
    format: str = "mp3",
    enable_timestamp: bool = True,
    timeout: int = 30,
) -> Optional[Dict[str, Any]]:
    """
    调用火山引擎 TTS，将合成音频保存到 output_path，并可选返回字级时间戳。

    Args:
        text: 待合成文本。
        output_path: 本地保存路径，如 temp/audio.mp3。
        speaker: 发音人 ID，默认使用适合搞笑短视频的音色。
        format: 音频格式，如 mp3 / wav。
        enable_timestamp: 是否请求字与音素时间戳（用于后期字幕对齐）。
        timeout: 请求超时（秒）。

    Returns:
        若 enable_timestamp 且接口返回了 payload 中的 words/phonemes，
        则返回包含 duration、words、phonemes 的字典；否则返回 None。
        音频已写入 output_path。

    Raises:
        ValueError: text 为空或 output_path 无效。
        VolcTTSError: 请求失败或业务错误。
    """
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string.")
    if not output_path or not output_path.strip():
        raise ValueError("output_path must be a non-empty string.")

    appkey, token = _volc_tts_credentials()
    payload_obj: Dict[str, Any] = {
        "text": text.strip(),
        "speaker": speaker,
        "audio_config": {
            "format": format,
            "enable_timestamp": enable_timestamp,
        },
    }
    payload_str = json.dumps(payload_obj, ensure_ascii=False)

    url = (
        f"{VOLC_TTS_INVOKE_URL}"
        f"?version={VOLC_TTS_VERSION}&token={token}&appkey={appkey}&namespace={VOLC_TTS_NAMESPACE}"
    )
    body = {"payload": payload_str}

    try:
        resp = requests.post(
            url,
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise VolcTTSError(f"火山 TTS 请求失败: {exc}") from exc

    if not resp.ok:
        raise VolcTTSError(
            f"火山 TTS 返回 HTTP {resp.status_code}: {resp.text}"
        )

    try:
        data: Dict[str, Any] = resp.json()
    except json.JSONDecodeError as exc:
        raise VolcTTSError(f"火山 TTS 返回非 JSON: {resp.text}") from exc

    status_code = data.get("status_code")
    if status_code is not None and status_code != 20000000:
        msg = data.get("status_text") or data.get("message") or str(data)
        raise VolcTTSError(f"火山 TTS 业务错误: {msg}")

    raw_b64 = data.get("data")
    if raw_b64:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(raw_b64))

    ts_result: Optional[Dict[str, Any]] = None
    payload_raw = data.get("payload")
    if isinstance(payload_raw, str) and payload_raw.strip():
        try:
            ts_result = json.loads(payload_raw)
        except json.JSONDecodeError:
            pass
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
    "VolcTTSError",
    "generate_tts_audio",
    "TTS_SPEAKER_FUNNY",
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
