import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from utils.api_clients import (  # type: ignore
    ArkAPIError,
    DeepSeekAPIError,
    TianAPIError,
    generate_video_script,
    get_douyin_hot_trends,
    optimize_visual_prompt,
)
from utils.media_generators import (  # type: ignore
    KlingAPIError,
    VolcTTSError,
    TTS_SPEAKER_FUNNY,
    TTS_SPEAKER_PRESETS,
    generate_kling_multishot,
    generate_tts_audio,
)
from utils.video_assembler import (  # type: ignore
    VideoAssembleError,
    assemble_final_video,
)


def load_env_defaults() -> Dict[str, str]:
    """
    Load default API keys from Streamlit secrets or environment variables / .env.
    Users can still override them in the sidebar.
    """
    load_dotenv()

    # 优先从 Streamlit secrets 读取，其次从环境变量读取
    def _get_default(name: str) -> str:
        if name in st.secrets:
            value = str(st.secrets[name])
            if value:
                return value
        return os.getenv(name, "")

    return {
        # Doubao / Ark 大模型
        "doubao": _get_default("ARK_API_KEY"),
        # DeepSeek 提示词优化
        "deepseek": _get_default("DEEPSEEK_API_KEY"),
        # Kling 视频生成（AccessKey + SecretKey，用于 JWT 鉴权）
        "kling": _get_default("KLING_ACCESS_KEY"),
        "kling_secret": _get_default("KLING_SECRET_KEY"),
        # TianAPI 抖音热榜
        "tianxing": _get_default("TIANAPI_KEY"),
    }


def _ensure_session_state() -> None:
    if "hot_trends" not in st.session_state:
        st.session_state["hot_trends"] = []
    if "selected_topic" not in st.session_state:
        st.session_state["selected_topic"] = ""
    if "final_video_path" not in st.session_state:
        st.session_state["final_video_path"] = ""
    if "last_script" not in st.session_state:
        st.session_state["last_script"] = None


def _download_file(url: str, dest: Path, *, timeout: int = 300) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except Exception as exc:  # noqa: BLE001
        raise KlingAPIError(f"下载可灵视频失败: {exc}") from exc


def main() -> None:
    st.set_page_config(
        page_title="VideoTaxi 一键热点短视频",
        layout="wide",
    )

    _ensure_session_state()
    defaults = load_env_defaults()

    # Sidebar: API Keys + 系统状态
    with st.sidebar:
        st.header("API Keys 配置")
        st.caption("在这里配置用于后续步骤的各类大模型 / 数据接口密钥。")

        doubao_key = st.text_input(
            "Doubao (Ark) API Key",
            type="password",
            value=defaults["doubao"],
        )
        deepseek_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=defaults["deepseek"],
        )
        kling_key = st.text_input(
            "Kling Access Key",
            type="password",
            value=defaults["kling"],
            help="可灵控制台获取的 Access Key，与 Secret Key 一起用于 JWT 鉴权。",
        )
        kling_secret = st.text_input(
            "Kling Secret Key",
            type="password",
            value=defaults["kling_secret"],
            help="可灵控制台获取的 Secret Key。",
        )
        tianxing_key = st.text_input(
            "TianAPI Key（用于抖音热榜）",
            type="password",
            value=defaults["tianxing"],
            help="在天行数据 / 天聚数行平台注册并申请 douyinhot 接口后获得。",
        )

        # 将侧边栏配置写回环境变量，供 utils 模块读取
        os.environ["ARK_API_KEY"] = doubao_key or ""
        os.environ["LAS_API_KEY"] = doubao_key or ""
        os.environ["DEEPSEEK_API_KEY"] = deepseek_key or ""
        os.environ["KLING_ACCESS_KEY"] = kling_key or ""
        os.environ["KLING_SECRET_KEY"] = kling_secret or ""
        os.environ["TIANAPI_KEY"] = tianxing_key or ""

        st.markdown("---")
        st.caption(
            "提示：可以将以上密钥写入项目根目录的 `.env` 文件，"
            "变量名分别为 `ARK_API_KEY`、`DEEPSEEK_API_KEY`、"
            "`KLING_ACCESS_KEY`、`KLING_SECRET_KEY`、`TIANAPI_KEY`，应用会自动读取；"
            "或在 `.streamlit/secrets.toml` 中配置同名字段。"
        )

        st.markdown("---")
        st.subheader("系统状态")
        st.write(
            f"豆包 / Ark：{'✅ 已配置' if doubao_key else '⚠️ 未配置'}"
        )
        st.write(
            f"DeepSeek：{'✅ 已配置' if deepseek_key else '⚠️ 未配置'}"
        )
        st.write(
            f"Kling 视频：{'✅ 已配置' if (kling_key and kling_secret) else '⚠️ 未配置（需 Access Key + Secret Key）'}"
        )
        st.write(
            f"抖音热榜（TianAPI）：{'✅ 已配置' if tianxing_key else '⚠️ 未配置'}"
        )

    # ---------- 主页面：两步流程 ----------
    st.title("VideoTaxi 2.0 · 热点短视频一键成片")
    st.caption(
        "① 选热点话题 → ② 一键生成（豆包写脚本 → DeepSeek 做分镜 → 可灵出多镜头视频+配音）"
    )
    st.markdown("---")

    # ---------- 步骤 1：选择热点话题 ----------
    st.subheader("① 选择热点话题")
    st.caption("先获取今日抖音热榜，在表格中选一个话题作为本支视频的主题。")
    col_controls, col_table = st.columns([1, 2], gap="large")

    with col_controls:
        top_n = st.slider("热榜条数", min_value=5, max_value=30, value=10, step=5)
        if st.button("获取今日抖音热榜", type="primary"):
            if not tianxing_key:
                st.warning("请先在左侧配置 TianAPI Key。")
            else:
                try:
                    with st.spinner("拉取热榜中..."):
                        trends = get_douyin_hot_trends(limit=top_n)
                except TianAPIError as exc:
                    st.error(f"抖音热榜失败：{exc}")
                    trends = []
                st.session_state["hot_trends"] = trends

    hot_trends: List[Dict[str, Any]] = st.session_state.get("hot_trends") or []

    with col_table:
        if not hot_trends:
            st.info("👆 点击「获取今日抖音热榜」，再在下方选择一条话题。")
        else:
            df = pd.DataFrame(hot_trends)
            st.dataframe(df, width="stretch", hide_index=True)
            titles = [item.get("title", "") for item in hot_trends]
            if titles:
                selected_title = st.selectbox(
                    "选择本支视频的话题：",
                    options=titles,
                    index=0,
                )
                st.session_state["selected_topic"] = selected_title
            else:
                st.session_state["selected_topic"] = ""

    selected_topic: str = st.session_state.get("selected_topic", "") or ""
    st.markdown("---")

    # ---------- 步骤 2：一键生成视频 ----------
    st.subheader("② 一键生成视频")
    disabled_generate = not (
        selected_topic and doubao_key and deepseek_key and kling_key and kling_secret
    )
    if disabled_generate:
        st.caption(
            "请先完成 ① 选择话题，并在左侧配置：豆包、DeepSeek、可灵（Access Key + Secret Key）、TianAPI。"
        )
    else:
        st.caption(
            "将自动完成：豆包联网写脚本（5 个有逻辑衔接的分镜）→ DeepSeek 转英文 prompt → 可灵生成画面+音效 → 火山 TTS 旁白与可灵音效叠加 + 显眼字幕合成。"
            "豆包+DeepSeek 约 1～2 分钟，可灵约 2～10 分钟，TTS+合成约 30 秒，总耗时约 4～13 分钟。"
        )
        st.caption(
            "**可选 BGM**：在 `.env` 中设置 `BGM_PATH`（本地 MP3 路径）或 `BGM_URL`（直链），可自动混入成片作背景。"
            " 推荐来源：Pixabay Music、Free Music Archive、爱给网等免费可商用音乐。"
        )

        # 配音音色选择（豆包语音 2.0）
        st.markdown("**配音音色（豆包语音 2.0）**")
        st.caption(
            "说明：上面脚本里的旁白会用这里选择的音色来朗读。\n"
            "- 通用音色：小天 2.0（默认）、Vivi 2.0、小何 2.0、云舟 2.0\n"
            "- 角色配音：儒雅逸辰、可爱女生、调皮公主、爽朗少年、天才同桌、知性灿灿"
        )
        voice_labels = list(TTS_SPEAKER_PRESETS.keys())
        # 当前选中的 speaker id（默认用小天 2.0）
        current_speaker_id = st.session_state.get("tts_speaker") or TTS_SPEAKER_FUNNY
        default_label = next(
            (name for name, vid in TTS_SPEAKER_PRESETS.items() if vid == current_speaker_id),
            voice_labels[0],
        )
        selected_label = st.selectbox(
            "选择一个你想要的配音音色：",
            options=voice_labels,
            index=voice_labels.index(default_label),
            help="选择后：豆包生成的旁白会由该音色朗读，声音风格会影响整体观感。",
        )
        selected_speaker_id = TTS_SPEAKER_PRESETS[selected_label]
        st.session_state["tts_speaker"] = selected_speaker_id

    if st.button("开始一键生成", type="primary", disabled=disabled_generate):
        temp_dir = Path("temp")
        temp_dir.mkdir(parents=True, exist_ok=True)

        raw_video_path = temp_dir / "kling_raw.mp4"
        audio_path = temp_dir / "tts_audio.mp3"
        final_video_path = temp_dir / "final_video.mp4"

        # 可选 BGM：本地路径或从 URL 下载后混入成片
        bgm_path_to_use: Optional[str] = None
        bgm_path_env = os.getenv("BGM_PATH", "").strip()
        bgm_url_env = os.getenv("BGM_URL", "").strip()
        if bgm_path_env and Path(bgm_path_env).is_file():
            bgm_path_to_use = bgm_path_env
        elif bgm_url_env:
            temp_bgm = temp_dir / "bgm.mp3"
            try:
                _download_file(bgm_url_env, temp_bgm)
                bgm_path_to_use = str(temp_bgm)
            except Exception:
                bgm_path_to_use = None

        use_status = hasattr(st, "status")

        try:
            if use_status:
                with st.status(
                    "开始一键生成...", state="running", expanded=True
                ) as status:
                    # 1. 生成脚本
                    status.update(label="步骤 1/4：正在生成文案（Doubao / Ark）...", state="running")
                    script = generate_video_script(selected_topic)
                    st.session_state["last_script"] = script
                    status.write(f"脚本标题：{script.get('title', '（未返回）')}")

                    # 2. 优化分镜为英文 Prompt
                    status.update(label="步骤 2/4：正在优化分镜（DeepSeek 多镜头 prompt）...", state="running")
                    scenes = script.get("visual_scenes") or []
                    if not isinstance(scenes, list) or not scenes:
                        raise ArkAPIError("模型未返回 visual_scenes，用于视频分镜。")
                    optimized_prompts = optimize_visual_prompt(scenes)
                    st.session_state["optimized_prompts"] = optimized_prompts
                    status.write("分镜已优化为英文 Prompt（多镜头）。")

                    # 等待可灵期间展示脚本与分镜，避免干等
                    status.write("---")
                    status.write("**📄 脚本预览**（可灵生成约 2～10 分钟，可先看下方内容）")
                    status.write(f"**标题：** {script.get('title', '')}")
                    narration_preview = (script.get("narration") or "")[:300]
                    if len(script.get("narration") or "") > 300:
                        narration_preview += "…"
                    status.write(f"**旁白：** {narration_preview}")
                    status.write("**中文分镜：**")
                    for idx, s in enumerate(script.get("visual_scenes") or [], start=1):
                        status.write(f"  {idx}. {s}")
                    status.write("**英文分镜（送可灵）：**")
                    for idx, p in enumerate(optimized_prompts, start=1):
                        status.write(f"  {idx}. {p[:100]}{'…' if len(p) > 100 else ''}")

                    # 3. 可灵多镜头视频（画面+音效/环境音，与 TTS 旁白叠加为背景）
                    status.update(
                        label="步骤 3/4：正在生成多镜头视频（Kling-v3，约 2～10 分钟）…",
                        state="running",
                    )
                    video_url = generate_kling_multishot(
                        optimized_prompts,
                        total_duration=15,
                        aspect_ratio="9:16",
                        mode="pro",
                        sound="on",
                        timeout=900,
                    )
                    _download_file(video_url, raw_video_path)

                    # 4. TTS 旁白 + 与可灵音效叠加 + 显眼字幕合成（豆包 narration 同时用于配音与字幕，保持一致）
                    status.update(label="步骤 4/4：正在生成旁白、混音与字幕（火山 TTS + MoviePy）...", state="running")
                    narration_text = script.get("narration", "")
                    tts_meta = generate_tts_audio(
                        narration_text,
                        str(audio_path),
                        speaker=st.session_state.get("tts_speaker", TTS_SPEAKER_FUNNY),
                        enable_timestamp=True,
                    )
                    assemble_final_video(
                        str(raw_video_path),
                        str(audio_path),
                        script_text=narration_text,
                        output_path=str(final_video_path),
                        timestamps=tts_meta,
                        bgm_path=bgm_path_to_use,
                    )
                    status.update(label="生成完成！可以在下方预览与下载成片（含旁白、字幕与可选 BGM）。", state="complete")
            else:
                with st.spinner("正在一键生成完整视频，请稍候..."):
                    script = generate_video_script(selected_topic)
                    st.session_state["last_script"] = script

                    scenes = script.get("visual_scenes") or []
                    if not isinstance(scenes, list) or not scenes:
                        raise ArkAPIError("模型未返回 visual_scenes，用于视频分镜。")
                    optimized_prompts = optimize_visual_prompt(scenes)
                    st.session_state["optimized_prompts"] = optimized_prompts

                    video_url = generate_kling_multishot(
                        optimized_prompts,
                        total_duration=15,
                        aspect_ratio="9:16",
                        mode="pro",
                        sound="on",
                        timeout=900,
                    )
                    _download_file(video_url, raw_video_path)
                    # 豆包 narration 原样用于 TTS 配音与字幕，保证声画一致
                    narration_text = script.get("narration", "")
                    tts_meta = generate_tts_audio(
                        narration_text,
                        str(audio_path),
                        speaker=st.session_state.get("tts_speaker", TTS_SPEAKER_FUNNY),
                        enable_timestamp=True,
                    )
                    assemble_final_video(
                        str(raw_video_path),
                        str(audio_path),
                        script_text=narration_text,
                        output_path=str(final_video_path),
                        timestamps=tts_meta,
                        bgm_path=bgm_path_to_use,
                    )

            st.session_state["final_video_path"] = str(final_video_path)
            st.success("短视频生成完成！可以在下方预览与下载（含旁白、字幕与可选 BGM）。")
        except (ArkAPIError, DeepSeekAPIError, KlingAPIError, VolcTTSError, VideoAssembleError) as exc:
            st.error(f"生成流程失败：{exc}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"发生未知错误：{exc}")

    # ---------- 生成结果：脚本 + 成片 ----------
    st.markdown("---")
    st.subheader("生成结果")
    last_script: Dict[str, Any] = st.session_state.get("last_script") or {}
    optimized_prompts: List[str] = st.session_state.get("optimized_prompts") or []
    final_video_path_str: str = st.session_state.get("final_video_path") or ""

    col_script, col_video = st.columns(2, gap="large")
    with col_script:
        st.markdown("**豆包脚本**")
        if not last_script:
            st.caption("完成 ② 一键生成后，此处显示标题、旁白与中文分镜。")
        else:
            st.markdown(f"**标题：** {last_script.get('title', '')}")
            st.markdown("**旁白：**")
            st.write(last_script.get("narration", ""))
            scenes = last_script.get("visual_scenes") or []
            if isinstance(scenes, list) and scenes:
                st.markdown("**中文分镜：**")
                for idx, s in enumerate(scenes, start=1):
                    st.write(f"{idx}. {s}")

        if optimized_prompts:
            with st.expander("DeepSeek 英文分镜（喂给可灵）", expanded=False):
                for idx, p in enumerate(optimized_prompts, start=1):
                    st.markdown(f"**镜头 {idx}**")
                    st.write(p)

    with col_video:
        st.markdown("**成片预览与下载**")
        if final_video_path_str and Path(final_video_path_str).is_file():
            video_path_obj = Path(final_video_path_str)
            with video_path_obj.open("rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.download_button(
                "下载 MP4",
                data=video_bytes,
                file_name=video_path_obj.name,
                mime="video/mp4",
            )
        else:
            st.caption("完成 ② 一键生成后，此处为可灵多镜头视频（含配音）。")


if __name__ == "__main__":
    main()


