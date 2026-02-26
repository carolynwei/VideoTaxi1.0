from pathlib import Path

from PIL import Image, ImageDraw, ImageFont  # 确保已安装：pip install pillow


def main() -> None:
    font_path = r"E:\VideoTaxi2.0\LXGWWenKai-Light.ttf"  # 或你现在 .env 里的 SUBTITLE_FONT_PATH
    if not Path(font_path).is_file():
        print("字体文件不存在：", font_path)
        return

    print("使用字体：", font_path)

    # 创建一张图片
    img = Image.new("RGB", (800, 240), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(font_path, 60)
    except Exception as e:
        print("ImageFont.truetype 加载失败：", e)
        return

    text = "测试中文 123 ABC\n这是第二行：旁白字幕测试"
    # 在 (40, 60) 位置画字
    draw.multiline_text(
        (40, 60),
        text,
        font=font,
        fill=(255, 255, 0),
        align="left",
    )

    out = Path("pil_font_test.png")
    img.save(out)
    print("已生成：", out)


if __name__ == "__main__":
    main()