from pathlib import Path

from dotenv import load_dotenv

from utils.media_generators import generate_tts_audio, VolcTTSError, TTS_SPEAKER_FUNNY
from utils.video_assembler import assemble_final_video, VideoAssembleError


def main() -> None:
    load_dotenv()

    root = Path(__file__).resolve().parent
    temp_dir = root / "temp"
    temp_dir.mkdir(exist_ok=True)

    video_path = temp_dir / "kling_raw.mp4"
    if not video_path.is_file():
        print(f"视频文件不存在：{video_path}")
        return

    # TODO：这里粘贴你想要的完整旁白文案（豆包脚本里的 narration）
    narration_text = """
 2026年3月12日上午九点。 古巴北部海域突发事件。 美籍快艇试图非法入境。 遭边防巡逻艇拦截检查。 快艇拒捕率先开火。 警卫队被迫还击自卫。 最终造成四人死亡。 六人受伤紧急送医。 事件细节仍在调查。 美古回应引全网热议。
""".strip()

    audio_path = temp_dir / "tts_audio_local.mp3"

    try:
        # 1. 豆包语音 2.0 生成配音（带时间戳）
        tts_meta = generate_tts_audio(
            narration_text,
            str(audio_path),
            speaker=TTS_SPEAKER_FUNNY,  # 或直接写成你在界面里选的音色 ID
            enable_timestamp=True,
        )
        print(f"已生成配音：{audio_path}")
    except (VolcTTSError, KeyError) as e:
        print(f"生成 TTS 失败：{e}")
        return

    # 2. 可灵视频 + TTS + 字幕 合成最终视频
    final_video_path = temp_dir / "final_video_local.mp4"
    try:
        out = assemble_final_video(
            str(video_path),
            str(audio_path),
            script_text=narration_text,
            output_path=str(final_video_path),
            timestamps=None,       # 先用按句平均方式打字幕，保证能看见
            bgm_path=None,
        )
        print(f"✅ 成片已生成：{out}")
    except VideoAssembleError as e:
        print(f"合成失败：{e}")


if __name__ == "__main__":
    main()