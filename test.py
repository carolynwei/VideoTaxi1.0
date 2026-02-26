from pathlib import Path
import os

from dotenv import load_dotenv

from moviepy.editor import TextClip, ColorClip, CompositeVideoClip  # type: ignore
from moviepy.config import change_settings  # type: ignore


def main() -> None:
    # 读 .env（拿 FFmpeg / ImageMagick / 字体路径）
    load_dotenv()

    ffmpeg_bin = os.getenv("FFMPEG_CMD", "").strip()
    im_bin = os.getenv("IMAGEMAGICK_BINARY", "").strip()
    font_path = os.getenv("SUBTITLE_FONT_PATH", r"E:\VideoTaxi2.0\font.ttf").strip()

    print(f"FFMPEG_BINARY = {ffmpeg_bin}")
    print(f"IMAGEMAGICK_BINARY = {im_bin}")
    print(f"SUBTITLE_FONT_PATH = {font_path}")

    settings = {}
    if ffmpeg_bin:
        settings["FFMPEG_BINARY"] = ffmpeg_bin
    if im_bin:
        settings["IMAGEMAGICK_BINARY"] = im_bin
    if settings:
        change_settings(settings)

    if not Path(font_path).is_file():
        raise FileNotFoundError(f"字体文件不存在: {font_path}")

    # 背景一张纯色图（1280x720，3 秒）
    size = (1280, 720)
    bg = ColorClip(size=size, color=(0, 0, 0), duration=3)

    text = "这是一个字幕测试\n使用 font.ttf + ImageMagick\n你好，抖音短视频。"
    try:
        txt_clip = TextClip(
            text,
            fontsize=60,
            font=font_path,        # 强制用你的 font.ttf
            color="yellow",
            stroke_color="black",
            stroke_width=2,
            method="caption",
            size=(int(size[0] * 0.9), None),
        ).set_position(("center", "bottom")).set_duration(3)
    except Exception as exc:
        print("TextClip 创建失败：", exc)
        return

    final = CompositeVideoClip([bg, txt_clip])

    out_png = Path("subtitle_test.png")
    out_mp4 = Path("subtitle_test.mp4")

    print("正在写出 PNG 预览...")
    final.save_frame(str(out_png), t=1.0)  # 取第 1 秒的帧

    print("正在写出 MP4 预览...")
    final.write_videofile(
        str(out_mp4),
        codec="libx264",
        audio=False,
        fps=25,
    )

    print("完成。请检查：")
    print(f"- {out_png}")
    print(f"- {out_mp4}")


if __name__ == "__main__":
    main()