from __future__ import annotations

"""
视频合成工具：使用 MoviePy 将可灵视频 + 火山 TTS 音频合成最终成片，并叠加字幕。

核心入口：
- assemble_final_video(video_path, audio_path, script_text, output_path, timestamps=None)
"""

import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from moviepy.editor import (  # type: ignore
        AudioFileClip,
        CompositeVideoClip,
        SubtitlesClip,
        TextClip,
        VideoFileClip,
        concatenate_videoclips,
    )
except Exception as exc:  # pragma: no cover - import error is reported at call time
    AudioFileClip = None  # type: ignore
    CompositeVideoClip = None  # type: ignore
    SubtitlesClip = None  # type: ignore
    TextClip = None  # type: ignore
    VideoFileClip = None  # type: ignore
    concatenate_videoclips = None  # type: ignore
    _MOVIEPY_IMPORT_ERROR = exc
else:
    _MOVIEPY_IMPORT_ERROR = None


class VideoAssembleError(RuntimeError):
    """Raised when final video assembly fails."""


def _ensure_moviepy_available() -> None:
    if _MOVIEPY_IMPORT_ERROR is not None:
        raise VideoAssembleError(
            "MoviePy 未正确安装，请先在虚拟环境中执行 `pip install moviepy`.\n"
            f"原始错误：{_MOVIEPY_IMPORT_ERROR}"
        )


def _build_subtitle_timeline_from_timestamps(
    timestamps: Dict[str, Any],
    *,
    total_duration: float,
) -> List[Tuple[Tuple[float, float], str]]:
    """
    将 TTS 返回的字级时间戳转换为 SubtitlesClip 需要的片段列表。

    字级时间戳格式示例：
    {
        "duration": 3.0,
        "words": [{"word": "你", "start_time": "0", "end_time": "0.05"}, ...]
    }
    """
    words = timestamps.get("words") or []
    if not isinstance(words, list) or not words:
        return []

    segments: List[Tuple[Tuple[float, float], str]] = []
    for item in words:
        if not isinstance(item, dict):
            continue
        w = str(item.get("word", "")).strip()
        if not w:
            continue
        try:
            start = float(item.get("start_time", 0.0))
            end = float(item.get("end_time", start))
        except (TypeError, ValueError):
            continue
        if end <= start:
            end = start + 0.05
        start = max(0.0, start)
        end = min(total_duration, end)
        segments.append(((start, end), w))

    return segments


def _build_subtitle_timeline_from_script_text(
    script_text: str,
    *,
    total_duration: float,
) -> List[Tuple[Tuple[float, float], str]]:
    """
    在没有时间戳时，将整段文案按句子平均切分到整段视频时长上。
    """
    cleaned = script_text.strip()
    if not cleaned:
        return []

    # 按中文/英文句号和感叹号等进行粗略分句
    raw_sentences = re.split(r"[。！？!?]", cleaned)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        sentences = [cleaned]

    n = len(sentences)
    base = total_duration / float(n)
    segments: List[Tuple[Tuple[float, float], str]] = []
    for idx, sent in enumerate(sentences):
        start = base * idx
        end = total_duration if idx == n - 1 else base * (idx + 1)
        segments.append(((start, end), sent))
    return segments


def _make_subtitles_clip(
    base_video,
    subtitle_segments: Sequence[Tuple[Tuple[float, float], str]],
):
    """
    根据时间片段和文本生成 SubtitlesClip，并设置为底部居中黄色描边大字体。
    """
    if not subtitle_segments:
        return None

    # 根据底层环境字体情况，可能需要调整字体名称
    def _text_clip_generator(txt: str):
        # 使用 label / caption 自动换行，宽度为视频宽度 90%
        width = int(base_video.w * 0.9)
        try:
            return TextClip(
                txt,
                fontsize=48,
                font="SimHei",  # 常见中文字体；若不存在会在外层捕获异常
                color="yellow",
                stroke_color="black",
                stroke_width=2,
                method="caption",
                size=(width, None),
            )
        except Exception:
            # 回退到默认字体
            return TextClip(
                txt,
                fontsize=48,
                color="yellow",
                stroke_color="black",
                stroke_width=2,
                method="caption",
                size=(width, None),
            )

    try:
        subs = SubtitlesClip(list(subtitle_segments), _text_clip_generator)
    except Exception as exc:  # 字体或 PIL 等问题则直接放弃字幕
        raise VideoAssembleError(f"创建字幕失败：{exc}") from exc

    return subs.set_position(("center", "bottom"))


def assemble_final_video(
    video_path: str,
    audio_path: str,
    script_text: str,
    output_path: str,
    *,
    timestamps: Optional[Dict[str, Any]] = None,
) -> str:
    """
    将下载好的可灵视频与火山 TTS 音频合成最终成片，并根据情况叠加字幕。

    - 若传入 timestamps（TTS 返回的 payload），则按字级时间戳生成动态字幕。
    - 若未传入 timestamps，则按 script_text 粗略分句，平均铺满整段视频。

    如果视频总时长短于音频时长，会自动循环拼接视频片段，使其长度不短于音频；
    如果视频长于音频，则按音频时长裁剪视频，以保证画面和声音对齐。

    Returns:
        最终导出 mp4 文件的路径（字符串形式）。
    """
    _ensure_moviepy_available()

    v_path = Path(video_path)
    a_path = Path(audio_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not v_path.is_file():
        raise VideoAssembleError(f"视频文件不存在：{v_path}")
    if not a_path.is_file():
        raise VideoAssembleError(f"音频文件不存在：{a_path}")

    # 资源加载与拼接
    video = VideoFileClip(str(v_path))
    audio = AudioFileClip(str(a_path))

    try:
        video_duration = float(video.duration or 0.0)
        audio_duration = float(audio.duration or 0.0)
        if video_duration <= 0 or audio_duration <= 0:
            raise VideoAssembleError(
                f"非法的媒体时长：video={video_duration}, audio={audio_duration}"
            )

        # 对齐时长：视频短则循环，长则裁剪
        if video_duration < audio_duration:
            loops = max(1, int(math.ceil(audio_duration / video_duration)))
            clips = [video] * loops
            looped = concatenate_videoclips(clips, method="compose")
            base_video = looped.subclip(0, audio_duration)
        else:
            base_video = video.subclip(0, audio_duration)

        base_video = base_video.set_audio(audio)

        # 准备字幕时间线
        subtitle_segments: List[Tuple[Tuple[float, float], str]]
        if timestamps:
            subtitle_segments = _build_subtitle_timeline_from_timestamps(
                timestamps,
                total_duration=base_video.duration,
            )
        else:
            subtitle_segments = _build_subtitle_timeline_from_script_text(
                script_text,
                total_duration=base_video.duration,
            )

        subtitle_clip = None
        if subtitle_segments:
            subtitle_clip = _make_subtitles_clip(base_video, subtitle_segments)

        if subtitle_clip is not None:
            final_clip = CompositeVideoClip([base_video, subtitle_clip])
        else:
            final_clip = base_video

        # 写出最终 mp4
        final_clip.write_videofile(
            str(out_path),
            codec="libx264",
            audio_codec="aac",
            fps=base_video.fps or 25,
        )
    finally:
        # 确保资源释放
        try:
            video.close()
        except Exception:
            pass
        try:
            audio.close()
        except Exception:
            pass
        try:
            if "base_video" in locals():
                base_video.close()
        except Exception:
            pass
        try:
            if "subtitle_clip" in locals() and subtitle_clip is not None:
                subtitle_clip.close()
        except Exception:
            pass
        try:
            if "final_clip" in locals():
                final_clip.close()
        except Exception:
            pass

    return str(out_path)


__all__ = [
    "VideoAssembleError",
    "assemble_final_video",
]

