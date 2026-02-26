from __future__ import annotations

"""
视频合成工具：使用 MoviePy 将可灵视频 + TTS 音频合成最终成片，并叠加字幕。

核心入口：
- assemble_final_video(video_path, audio_path, script_text, output_path, timestamps=None)
"""

import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore

try:
    # 核心剪辑组件从 moviepy.editor 导入（1.x 兼容）
    from moviepy.editor import (  # type: ignore
        AudioFileClip,
        CompositeVideoClip,
        ImageClip,
        TextClip,
        VideoFileClip,
        concatenate_videoclips,
    )
    from moviepy.audio.AudioClip import CompositeAudioClip  # type: ignore
    try:
        from moviepy.audio.AudioClip import concatenate_audioclips  # type: ignore
    except ImportError:
        concatenate_audioclips = None  # type: ignore
    # SubtitlesClip 在 1.x 中位于 video.tools.subtitles，而不是 editor 顶层
    try:
        from moviepy.video.tools.subtitles import SubtitlesClip  # type: ignore
    except Exception:
        SubtitlesClip = None  # type: ignore
    from moviepy.config import change_settings  # type: ignore
except Exception as exc:  # pragma: no cover - import error is reported at call time
    AudioFileClip = None  # type: ignore
    CompositeVideoClip = None  # type: ignore
    CompositeAudioClip = None  # type: ignore
    concatenate_audioclips = None  # type: ignore
    SubtitlesClip = None  # type: ignore
    TextClip = None  # type: ignore
    VideoFileClip = None  # type: ignore
    concatenate_videoclips = None  # type: ignore
    ImageClip = None  # type: ignore
    change_settings = None  # type: ignore
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


def _configure_moviepy_binaries_from_env() -> None:
    """
    每次合成前，从当前环境变量读取 FFmpeg / ImageMagick 路径，
    这样可以配合 python-dotenv 在运行时加载 .env。
    """
    try:
        settings: Dict[str, str] = {}
        ffmpeg_cmd = os.getenv("FFMPEG_CMD", "").strip()
        if ffmpeg_cmd:
            settings["FFMPEG_BINARY"] = ffmpeg_cmd
        im_cmd = os.getenv("IMAGEMAGICK_BINARY", "").strip()
        if im_cmd:
            settings["IMAGEMAGICK_BINARY"] = im_cmd
        if settings:
            change_settings(settings)
    except Exception:
        # 配置出错时不终止，由后续错误信息提示
        pass


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

    # 字幕字体：优先使用 SUBTITLE_FONT_PATH（绝对路径），否则退回到项目根目录的 font.ttf，
    # 再否则使用系统默认字体。
    font_path_env = os.getenv("SUBTITLE_FONT_PATH", "").strip()
    if font_path_env and Path(font_path_env).is_file():
        subtitle_font = font_path_env
    else:
        project_root = Path(__file__).resolve().parent.parent
        default_font = project_root / "font.ttf"
        subtitle_font = str(default_font) if default_font.is_file() else "SimHei"

    # 优先使用 PIL 直接渲染字幕（不依赖 ImageMagick），确保中文可见，并在单行过长时自动换行
    def _text_clip_pil(txt: str):
        width = int(round(float(base_video.w) * 0.9))
        if ImageFont is None or Image is None or ImageDraw is None or ImageClip is None:
            raise RuntimeError("PIL 或 ImageClip 不可用")

        try:
            font = ImageFont.truetype(subtitle_font, 64)
        except Exception:
            font = ImageFont.load_default()

        # 先在虚拟画布上根据最大宽度做自动换行，避免长句被截断
        max_text_width = max(1, width - 80)
        dummy_img = Image.new("RGBA", (width, 1000))
        draw = ImageDraw.Draw(dummy_img)

        def _wrap_text_for_width(raw_text: str) -> str:
            """
            按像素宽度自动换行：
            - 保留原有换行符
            - 对中文/英文混排逐字符试探，超出宽度则换行
            """
            if not raw_text:
                return ""

            lines = []
            current = ""

            for ch in raw_text:
                # 手动换行符，直接断行
                if ch == "\n":
                    lines.append(current)
                    current = ""
                    continue

                test = current + ch
                bbox_line = draw.textbbox((0, 0), test, font=font)
                line_w = bbox_line[2] - bbox_line[0]

                # 超出最大宽度，则当前行收尾，开启新行
                if line_w > max_text_width and current:
                    lines.append(current)
                    # 开头若是空格就丢弃，避免新行首空格
                    current = ch if ch != " " else ""
                else:
                    current = test

            if current:
                lines.append(current)

            return "\n".join(lines)

        wrapped_txt = _wrap_text_for_width(txt)
        if not wrapped_txt:
            wrapped_txt = txt

        # 计算包围盒
        bbox = draw.multiline_textbbox((0, 0), wrapped_txt, font=font, align="center")
        text_w = max(1, int(bbox[2] - bbox[0]))
        text_h = max(1, int(bbox[3] - bbox[1]))

        img_w = int(min(width, text_w + 80))
        img_h = int(text_h + 60)

        # 半透明黑底
        img = Image.new("RGBA", (img_w, img_h))
        draw = ImageDraw.Draw(img)
        # 居中绘制白字+黑描边
        x = (img_w - text_w) // 2 - bbox[0]
        y = (img_h - text_h) // 2 - bbox[1]
        draw.multiline_text(
            (x, y),
            wrapped_txt,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=3,
            stroke_fill=(0, 0, 0, 255),
            align="center",
        )

        arr = np.array(img)
        return ImageClip(arr)

    subtitle_clips: List[Any] = []
    for (start, end), text in subtitle_segments:
        text = str(text).strip()
        if not text:
            continue
        try:
            clip = _text_clip_pil(text)
        except Exception as exc:
            print(f"[VideoAssembler] 单条字幕创建失败，将跳过该句：{exc}")
            continue
        dur = max(0.1, float(end - start))
        clip = (
            clip.set_start(float(start))
            .set_duration(dur)
            .set_position(("center", "bottom"))
        )
        subtitle_clips.append(clip)

    if not subtitle_clips:
        return None

    # 将所有字幕叠加成一个透明图层，时长与视频一致
    subs_layer = CompositeVideoClip(subtitle_clips, size=(base_video.w, base_video.h))
    subs_layer = subs_layer.set_duration(base_video.duration)
    return subs_layer


def assemble_final_video(
    video_path: str,
    audio_path: str,
    script_text: str,
    output_path: str,
    *,
    timestamps: Optional[Dict[str, Any]] = None,
    bgm_path: Optional[str] = None,
    target_aspect: Optional[str] = None,
) -> str:
    """
    将下载好的可灵视频与火山 TTS 音频合成最终成片，并叠加字幕；可选混入 BGM。

    - 若传入 timestamps（TTS 返回的 payload），则按字级时间戳生成动态字幕。
    - 若未传入 timestamps，则按 script_text 粗略分句，平均铺满整段视频。
    - 若传入 bgm_path（本地 BGM 文件路径），则将该音频循环至成片时长、压低音量后与现有音轨叠加作背景。

    如果视频总时长短于音频时长，会自动循环拼接视频片段，使其长度不短于音频；
    如果视频长于音频，则按音频时长裁剪视频，以保证画面和声音对齐。

    Returns:
        最终导出 mp4 文件的路径（字符串形式）。
    """
    _ensure_moviepy_available()
    _configure_moviepy_binaries_from_env()

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

        # 对齐时长：优先通过变速让整条视频与音频等长，避免重复循环段落。
        # 这样每个镜头在整体节奏上仍保持相对比例，更有利于旁白与画面对应。
        speed_factor = video_duration / audio_duration
        if speed_factor <= 0:
            raise VideoAssembleError(
                f"非法的速度因子：video={video_duration}, audio={audio_duration}"
            )

        # 使用 fx.speedx 调整全片速度
        from moviepy.editor import vfx  # type: ignore

        if abs(speed_factor - 1.0) < 1e-3:
            # 时长基本一致，直接裁剪到音频长度
            base_video = video.subclip(0, audio_duration)
        else:
            # speed_factor > 1：视频原本更长，整体加速；<1：视频原本更短，整体减速
            sped = video.fx(vfx.speedx, factor=speed_factor)
            base_video = sped.subclip(0, audio_duration)

        # # 若视频自带音轨（如可灵生成的音效/环境音），与 TTS 旁白叠加：视频音轨作背景，TTS 作前景
        # mixed_audio = None
        # video_audio = base_video.audio
        # if video_audio is not None and CompositeAudioClip is not None:
        #     try:
        #         kling_bg = video_audio.volumex(0.25)
        #         mixed_audio = CompositeAudioClip([kling_bg, audio])
        #         base_video = base_video.set_audio(mixed_audio)
        #         video_audio.close()
        #     except Exception:
        #         base_video = base_video.set_audio(audio)
        #         mixed_audio = None
        # else:
        #     base_video = base_video.set_audio(audio)

        # 暂时只保留 TTS 旁白，不混可灵环境音，避免复合音轨出错
        mixed_audio = None
        base_video = base_video.set_audio(audio)

        # 按需裁剪成目标画面比例（例如 9:16 / 16:9 / 1:1 等），优先保证人物不被严重裁掉：
        # - 若源视频更宽，则左右居中裁掉多余宽度；
        # - 若源视频更窄/更高，则上下居中裁掉多余高度。
        if target_aspect:
            try:
                parts = str(target_aspect).strip().split(":")
                if len(parts) == 2:
                    aw = float(parts[0])
                    ah = float(parts[1])
                    if aw > 0 and ah > 0:
                        target_ratio = aw / ah
                        vw = float(base_video.w)
                        vh = float(base_video.h)
                        if vw > 0 and vh > 0:
                            src_ratio = vw / vh
                            if abs(src_ratio - target_ratio) > 1e-3:
                                # 源更宽：裁掉左右；源更窄/更高：裁掉上下
                                if src_ratio > target_ratio:
                                    new_w = int(round(vh * target_ratio))
                                    new_w = max(1, min(int(vw), new_w))
                                    x1 = max(0, int((vw - new_w) / 2))
                                    x2 = x1 + new_w
                                    base_video = base_video.crop(x1=x1, x2=x2)
                                else:
                                    new_h = int(round(vw / target_ratio))
                                    new_h = max(1, min(int(vh), new_h))
                                    y1 = max(0, int((vh - new_h) / 2))
                                    y2 = y1 + new_h
                                    base_video = base_video.crop(y1=y1, y2=y2)
            except Exception:
                # 比例解析或裁剪异常时忽略，继续用原始画面比例
                pass

        # 可选：叠加 BGM（洗脑神曲等）作背景，音量压低
        target_duration = float(base_video.duration or 0.0)
        if bgm_path and Path(bgm_path).is_file() and target_duration > 0:
            bgm_audio = None
            bgm_looped = None
            try:
                bgm_audio = AudioFileClip(str(bgm_path))
                bgm_dur = float(bgm_audio.duration or 0.0)
                if bgm_dur > 0:
                    if bgm_dur >= target_duration:
                        bgm_looped = bgm_audio.subclip(0, target_duration)
                    elif concatenate_audioclips is not None:
                        n_loops = max(1, int(math.ceil(target_duration / bgm_dur)))
                        bgm_looped = concatenate_audioclips([bgm_audio] * n_loops).subclip(0, target_duration)
                    else:
                        bgm_looped = bgm_audio.subclip(0, min(bgm_dur, target_duration))
                    if bgm_looped is not None:
                        bgm_low = bgm_looped.volumex(0.2)
                        current_audio = base_video.audio
                        new_mixed = CompositeAudioClip([bgm_low, current_audio])
                        base_video = base_video.set_audio(new_mixed)
                        mixed_audio = new_mixed
            except Exception:
                pass
            finally:
                if bgm_audio is not None:
                    try:
                        bgm_audio.close()
                    except Exception:
                        pass

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
        try:
            if "mixed_audio" in locals() and mixed_audio is not None:
                mixed_audio.close()
        except Exception:
            pass

    return str(out_path)


__all__ = [
    "VideoAssembleError",
    "assemble_final_video",
]

