import os
from pathlib import Path
from typing import Any, Dict, List

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
    generate_kling_video,
    generate_tts_audio,
)
from utils.video_assembler import assemble_final_video  # type: ignore


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
        # Kling 视频生成
        "kling": _get_default("KLING_ACCESS_KEY"),
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
            "Kling API Key",
            type="password",
            value=defaults["kling"],
        )
        tianxing_key = st.text_input(
            "TianAPI Key（用于抖音热榜）",
            type="password",
            value=defaults["tianxing"],
            help="在天行数据 / 天聚数行平台注册并申请 douyinhot 接口后获得。",
        )

        # 将侧边栏配置写回环境变量，供 utils 模块读取
        os.environ["ARK_API_KEY"] = doubao_key or ""
        os.environ["DEEPSEEK_API_KEY"] = deepseek_key or ""
        os.environ["KLING_ACCESS_KEY"] = kling_key or ""
        os.environ["TIANAPI_KEY"] = tianxing_key or ""

        st.markdown("---")
        st.caption(
            "提示：可以将以上密钥写入项目根目录的 `.env` 文件，"
            "变量名分别为 `ARK_API_KEY`、`DEEPSEEK_API_KEY`、"
            "`KLING_ACCESS_KEY`、`TIANAPI_KEY`，应用会自动读取；"
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
            f"Kling 视频：{'✅ 已配置' if kling_key else '⚠️ 未配置'}"
        )
        st.write(
            f"抖音热榜（TianAPI）：{'✅ 已配置' if tianxing_key else '⚠️ 未配置'}"
        )

    # Main page: 热榜 -> 脚本 -> 媒体生成 -> 合成
    st.title("VideoTaxi 2.0 - 一键热点短视频生成")
    st.write(
        "流程：**获取今日抖音热榜 -> 选择话题 -> 生成脚本与分镜 -> 可灵视频 + 火山 TTS -> MoviePy 自动剪辑合成。**"
    )

    col_controls, col_table = st.columns([1, 2], gap="large")

    with col_controls:
        top_n = st.slider("展示热榜条数", min_value=5, max_value=30, value=10, step=5)

        if st.button("获取今日抖音热榜"):
            if not tianxing_key:
                st.warning("请先在左侧配置 TianAPI / Tianxing Key。")
            else:
                try:
                    with st.spinner("正在获取抖音热榜..."):
                        trends = get_douyin_hot_trends(limit=top_n)
                except TianAPIError as exc:
                    st.error(f"抖音热榜 API 调用失败：{exc}")
                    trends = []
                st.session_state["hot_trends"] = trends

    hot_trends: List[Dict[str, Any]] = st.session_state.get("hot_trends") or []

    with col_table:
        if not hot_trends:
            st.info("点击左侧按钮“获取今日抖音热榜”开始。")
        else:
            df = pd.DataFrame(hot_trends)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )
            titles = [item.get("title", "") for item in hot_trends]
            if titles:
                default_idx = 0
                selected_title = st.selectbox(
                    "选择一个话题用于一键生成：",
                    options=titles,
                    index=default_idx,
                )
                st.session_state["selected_topic"] = selected_title
            else:
                st.session_state["selected_topic"] = ""

    selected_topic: str = st.session_state.get("selected_topic", "") or ""

    st.markdown("---")
    st.subheader("第五阶段：一键生成完整短视频")

    disabled_generate = not (
        selected_topic and doubao_key and deepseek_key and kling_key
    )
    if disabled_generate:
        st.caption(
            "提示：需要选择一个话题，并在左侧配置 Doubao / DeepSeek / Kling API Key 才能开始一键生成。"
        )

    if st.button("开始一键生成", disabled=disabled_generate):
        temp_dir = Path("temp")
        temp_dir.mkdir(parents=True, exist_ok=True)

        raw_video_path = temp_dir / "kling_raw.mp4"
        audio_path = temp_dir / "tts_audio.mp3"
        final_video_path = temp_dir / "final_video.mp4"

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
                    status.update(label="步骤 2/4：正在优化分镜（DeepSeek 提示词工程）...", state="running")
                    scenes = script.get("visual_scenes") or []
                    if not isinstance(scenes, list) or not scenes:
                        raise ArkAPIError("模型未返回 visual_scenes，用于视频分镜。")
                    optimized_prompts = optimize_visual_prompt(scenes)
                    # 简单策略：将所有英文 prompt 串联为一个长 prompt
                    final_prompt = " ".join(optimized_prompts)
                    status.write("分镜已优化为英文 Prompt。")

                    # 3. 可灵视频 + 火山 TTS 音频
                    status.update(label="步骤 3/4：正在生成视频与语音（Kling + 火山 TTS）...", state="running")
                    video_url = generate_kling_video(final_prompt)
                    _download_file(video_url, raw_video_path)

                    tts_meta = generate_tts_audio(
                        script.get("narration", ""),
                        str(audio_path),
                        enable_timestamp=True,
                    )

                    # 4. MoviePy 合成最终成片
                    status.update(label="步骤 4/4：正在合成最终视频（MoviePy）...", state="running")
                    assemble_final_video(
                        str(raw_video_path),
                        str(audio_path),
                        script_text=str(script.get("narration", "")),
                        output_path=str(final_video_path),
                        timestamps=tts_meta,
                    )

                    status.update(label="生成完成！可以在下方预览与下载成片。", state="complete")
            else:
                with st.spinner("正在一键生成完整视频，请稍候..."):
                    script = generate_video_script(selected_topic)
                    st.session_state["last_script"] = script

                    scenes = script.get("visual_scenes") or []
                    if not isinstance(scenes, list) or not scenes:
                        raise ArkAPIError("模型未返回 visual_scenes，用于视频分镜。")
                    optimized_prompts = optimize_visual_prompt(scenes)
                    final_prompt = " ".join(optimized_prompts)

                    video_url = generate_kling_video(final_prompt)
                    _download_file(video_url, raw_video_path)

                    tts_meta = generate_tts_audio(
                        script.get("narration", ""),
                        str(audio_path),
                        enable_timestamp=True,
                    )

                    assemble_final_video(
                        str(raw_video_path),
                        str(audio_path),
                        script_text=str(script.get("narration", "")),
                        output_path=str(final_video_path),
                        timestamps=tts_meta,
                    )

            st.session_state["final_video_path"] = str(final_video_path)
            st.success("短视频生成完成！可以在下方预览与下载。")
        except (ArkAPIError, DeepSeekAPIError, KlingAPIError, VolcTTSError) as exc:
            st.error(f"生成流程失败：{exc}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"发生未知错误：{exc}")

    # 展示脚本与最终视频
    last_script: Dict[str, Any] = st.session_state.get("last_script") or {}
    final_video_path_str: str = st.session_state.get("final_video_path") or ""

    col_script, col_video = st.columns(2, gap="large")
    with col_script:
        st.subheader("生成的脚本与分镜")
        if not last_script:
            st.info("一键生成后，这里会展示豆包返回的标题、旁白和分镜。")
        else:
            st.markdown(f"**标题：** {last_script.get('title', '')}")
            st.markdown("**旁白文案：**")
            st.write(last_script.get("narration", ""))
            scenes = last_script.get("visual_scenes") or []
            if isinstance(scenes, list) and scenes:
                st.markdown("**分镜：**")
                for idx, s in enumerate(scenes, start=1):
                    st.write(f"{idx}. {s}")

    with col_video:
        st.subheader("最终合成视频预览")
        if final_video_path_str and Path(final_video_path_str).is_file():
            video_path_obj = Path(final_video_path_str)
            with video_path_obj.open("rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.download_button(
                "下载最终视频 (MP4)",
                data=video_bytes,
                file_name=video_path_obj.name,
                mime="video/mp4",
            )
        else:
            st.info("一键生成完成后，这里会展示最终合成的 MP4 视频。")


if __name__ == "__main__":
    main()


