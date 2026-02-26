import os
import time
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
    MinimaxAPIError,
    VolcTTSError,
    TTS_SPEAKER_FUNNY,
    TTS_SPEAKER_PRESETS,
    generate_kling_multishot,
    generate_minimax_video,
    generate_tts_audio,
)
from utils.video_assembler import (  # type: ignore
    VideoAssembleError,
    assemble_final_video,
)
from utils.user_store import (  # type: ignore
    append_history_item,
    ensure_user,
    load_user_history,
    persist_video_for_user,
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
        # MiniMax 文生视频（Anthropic 兼容 Key）
        "minimax": _get_default("ANTHROPIC_API_KEY"),
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


def load_css() -> None:
    """Inject global CSS for a modern, glassmorphism-style UI."""
    st.markdown(
        """
        <style>
        /* ---------- 全局基础样式 ---------- */
        :root {
            --vt-accent: #00A3FF;
            --vt-accent-soft: rgba(0, 163, 255, 0.15);
            --vt-bg-deep: #020617;
            --vt-bg-panel: rgba(15, 23, 42, 0.82);
            --vt-border-subtle: rgba(148, 163, 184, 0.5);
        }

        /* 隐藏原生顶部区域与菜单，去掉红线感 */
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        #MainMenu,
        footer {
            visibility: hidden;
            height: 0;
        }

        /* 背景与主容器：深色渐变 + 轻微光晕 */
        [data-testid="stAppViewContainer"] > .main {
            background:
                radial-gradient(circle at 10% 0%, #1a1a2e 0, #050816 55%, #020617 100%);
            font-family: "Inter", -apple-system, BlinkMacSystemFont, system-ui,
                         "PingFang SC", "Segoe UI", sans-serif;
            color: #e5e7eb;
        }

        main.block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.4rem;
        }

        /* 调整所有文字默认颜色（避免过灰） */
        .stMarkdown, .stCaption, .stText, .stSubheader, .stHeader, .stDataFrame {
            color: #e5e7eb !important;
        }

        /* ---------- 顶部 Hero：videoTaxi 横幅（磨砂玻璃 + 渐变） ---------- */
        .videotaxi-hero {
            position: relative;
            display: flex;
            align-items: center;
            gap: 1.5rem;
            padding: 1.2rem 1.6rem;
            margin-bottom: 1.1rem;
            border-radius: 22px;
            background:
                linear-gradient(135deg, rgba(255, 75, 75, 0.9), rgba(26, 26, 46, 0.85) 55%, rgba(5, 8, 22, 0.95));
            border: 1px solid rgba(248, 250, 252, 0.08);
            box-shadow:
                0 24px 55px rgba(0, 0, 0, 0.65),
                0 0 64px rgba(56, 189, 248, 0.28);
            backdrop-filter: blur(14px) saturate(160%);
            -webkit-backdrop-filter: blur(14px) saturate(160%);
            overflow: hidden;
        }

        .videotaxi-hero::before {
            content: "";
            position: absolute;
            inset: -40%;
            background:
                radial-gradient(circle at 0% 0%, rgba(96, 165, 250, 0.16) 0, transparent 55%),
                radial-gradient(circle at 90% 20%, rgba(250, 204, 21, 0.18) 0, transparent 52%),
                radial-gradient(circle at 50% 120%, rgba(251, 113, 133, 0.16) 0, transparent 60%);
            opacity: 0.75;
            pointer-events: none;
        }

        .videotaxi-hero > * {
            position: relative;
            z-index: 1;
        }

        .videotaxi-logo-circle {
            width: 68px;
            height: 68px;
            border-radius: 999px;
            background:
                radial-gradient(circle at 30% 20%, #ffe66b, #f59e0b 55%, #b45309 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 34px;
            box-shadow:
                0 0 24px rgba(248, 250, 252, 0.35),
                0 0 32px rgba(252, 211, 77, 0.75);
        }

        .videotaxi-logo-circle span {
            filter: drop-shadow(0 0 6px rgba(0,0,0,0.75));
        }

        .videotaxi-text {
            display: flex;
            flex-direction: column;
            gap: 0.1rem;
        }

        .videotaxi-name {
            font-size: 32px;
            font-weight: 900;
            letter-spacing: 0.03em;
            color: #fef2f2;
            text-shadow:
                0 0 18px rgba(248, 113, 113, 0.88),
                0 0 32px rgba(248, 113, 113, 0.68);
        }

        .videotaxi-name span {
            color: #fee2e2;
            font-weight: 700;
        }

        .videotaxi-tagline {
            font-size: 18px;
            font-weight: 600;
            color: #e5e7eb;
            text-shadow: 0 0 10px rgba(15, 23, 42, 0.95);
        }

        .videotaxi-sub {
            margin-top: 0.2rem;
            font-size: 13px;
            color: #9ca3af;
        }

        /* ---------- 新手 3 步走：卡片化 Step Cards ---------- */
        .vt-step-wrapper {
            margin-bottom: 1.0rem;
        }

        .vt-step-title {
            font-size: 15px;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: rgba(148, 163, 184, 0.95);
            margin-bottom: 0.35rem;
        }

        .step-cards {
            display: flex;
            gap: 0.9rem;
            flex-wrap: wrap;
        }

        .step-card {
            flex: 1 1 0;
            min-width: 0;
            padding: 0.9rem 1.0rem;
            border-radius: 16px;
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(15, 23, 42, 0.78));
            border: 1px solid rgba(148, 163, 184, 0.35);
            box-shadow:
                0 18px 35px rgba(15, 23, 42, 0.78),
                0 0 24px rgba(0, 163, 255, 0.22);
            backdrop-filter: blur(14px);
        }

        .step-header {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.2rem;
        }

        .step-number {
            width: 28px;
            height: 28px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: 700;
            color: #0b1220;
            background: radial-gradient(circle at 30% 0%, #7dd3fc, #0ea5e9 60%, #0369a1 100%);
            box-shadow:
                0 0 12px rgba(56, 189, 248, 0.8),
                0 0 24px rgba(59, 130, 246, 0.7);
        }

        .step-title {
            font-size: 14px;
            font-weight: 600;
            color: #e5e7eb;
        }

        .step-body {
            font-size: 13px;
            color: #9ca3af;
        }

        /* ---------- 用户名展示：在线状态指示灯 ---------- */
        .user-status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.55rem 0.9rem;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.6);
            box-shadow: 0 10px 32px rgba(15, 23, 42, 0.85);
        }

        .user-avatar {
            width: 26px;
            height: 26px;
            border-radius: 999px;
            background: radial-gradient(circle at 30% 0%, #4ade80, #22c55e 60%, #15803d 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
            font-weight: 700;
            color: #052e16;
        }

        .user-meta {
            display: flex;
            flex-direction: column;
            gap: 0.05rem;
        }

        .user-name {
            font-size: 13px;
            font-weight: 600;
            color: #e5e7eb;
        }

        .user-sub {
            display: flex;
            align-items: center;
            gap: 0.3rem;
            font-size: 11px;
            color: #9ca3af;
        }

        .user-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: #22c55e;
            box-shadow: 0 0 8px rgba(34, 197, 94, 0.9);
        }

        .user-dot.online {
            animation: pulse-online 1.7s ease-in-out infinite;
        }

        /* ---------- 按钮样式：主按钮 + Ghost 按钮 ---------- */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #ff416c, #ff4b2b, #ff9a3c);
            color: #f9fafb;
            border-radius: 999px;
            border: none;
            padding: 0.5rem 1.6rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            box-shadow:
                0 16px 40px rgba(255, 75, 43, 0.42),
                0 0 22px rgba(248, 113, 113, 0.85);
            transition: transform 0.25s ease, box-shadow 0.25s ease, filter 0.25s ease;
            animation: pulse-glow 2.4s ease-in-out infinite;
        }

        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) scale(1.01);
            filter: brightness(1.05);
            box-shadow:
                0 22px 55px rgba(255, 75, 43, 0.55),
                0 0 32px rgba(248, 113, 113, 0.95);
        }

        /* Ghost 按钮：用于退出登录等操作 */
        div.stButton > button[kind="secondary"],
        div.stButton > button:not([kind]) {
            background: transparent;
            color: #e5e7eb;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.7);
            padding: 0.38rem 1.0rem;
            font-weight: 500;
            box-shadow: none;
            transition: background 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
        }

        div.stButton > button[kind="secondary"]:hover,
        div.stButton > button:not([kind]):hover {
            background: rgba(15, 23, 42, 0.85);
            border-color: rgba(148, 163, 184, 0.95);
            transform: translateY(-1px);
        }

        /* 输入框与表格：细节微调，贴合深色玻璃感 */
        .stTextInput > div > div > input {
            background-color: rgba(15, 23, 42, 0.95);
            border-radius: 12px;
            border: 1px solid rgba(55, 65, 81, 0.85);
            color: #e5e7eb;
        }

        .stDataFrame {
            background: rgba(15, 23, 42, 0.9);
            border-radius: 12px;
            border: 1px solid rgba(55, 65, 81, 0.85);
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.9);
        }

        /* ---------- 动画 ---------- */
        @keyframes pulse-glow {
            0% {
                box-shadow:
                    0 14px 34px rgba(255, 75, 43, 0.45),
                    0 0 26px rgba(248, 113, 113, 0.85);
            }
            50% {
                box-shadow:
                    0 18px 42px rgba(255, 75, 43, 0.7),
                    0 0 34px rgba(248, 113, 113, 1);
            }
            100% {
                box-shadow:
                    0 14px 34px rgba(255, 75, 43, 0.45),
                    0 0 26px rgba(248, 113, 113, 0.85);
            }
        }

        @keyframes pulse-online {
            0% {
                transform: scale(1);
                box-shadow: 0 0 5px rgba(34, 197, 94, 0.7);
            }
            50% {
                transform: scale(1.2);
                box-shadow: 0 0 10px rgba(34, 197, 94, 1);
            }
            100% {
                transform: scale(1);
                box-shadow: 0 0 5px rgba(34, 197, 94, 0.7);
            }
        }

        /* ---------- 响应式优化：窄屏堆叠布局 ---------- */
        @media (max-width: 768px) {
            .videotaxi-hero {
                padding: 0.9rem 1.0rem;
                gap: 1.0rem;
            }
            .videotaxi-name {
                font-size: 24px;
            }
            .videotaxi-tagline {
                font-size: 15px;
            }
            .videotaxi-logo-circle {
                width: 56px;
                height: 56px;
                font-size: 28px;
            }
            .step-cards {
                flex-direction: column;
            }
            main.block-container {
                padding-left: 0.9rem;
                padding-right: 0.9rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="VideoTaxi 一键热点短视频",
        layout="wide",
    )

    _ensure_session_state()
    load_css()
    defaults = load_env_defaults()

    # 顶部品牌区：VideoTaxi 出租车 Logo + 标语（磨砂玻璃风格）
    st.markdown(
        """
        <div class="videotaxi-hero">
          <div class="videotaxi-logo-circle">
            <span>🚕</span>
          </div>
          <div class="videotaxi-text">
            <div class="videotaxi-name">videoTaxi<span> · 热点短视频引擎</span></div>
            <div class="videotaxi-tagline">让流量 7×24 小时为你跑单！</div>
            <div class="videotaxi-sub">一键热点采集 · 文案 + 分镜 + 画面 + 配音 + 字幕全流程自动化</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 新手 3 步走：卡片式布局
    st.markdown(
        """
        <div class="vt-step-wrapper">
          <div class="vt-step-title">新手 3 步走</div>
          <div class="step-cards">
            <div class="step-card">
              <div class="step-header">
                <div class="step-number">1</div>
                <div class="step-title">填写必需的 API Keys</div>
              </div>
              <div class="step-body">
                在左侧填入 Doubao（Ark）、DeepSeek，再任选一个视频生成 Key（MiniMax 或 Kling）。
              </div>
            </div>
            <div class="step-card">
              <div class="step-header">
                <div class="step-number">2</div>
                <div class="step-title">选择今日抖音热榜话题</div>
              </div>
              <div class="step-body">
                点击「获取今日抖音热榜」，在表格里点选你感兴趣的一条话题。
              </div>
            </div>
            <div class="step-card">
              <div class="step-header">
                <div class="step-number">3</div>
                <div class="step-title">一键生成完整成片</div>
              </div>
              <div class="step-body">
                在下方选择视频模型和配音声音，点击「开始一键生成」，等待系统自动完成全流程。
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 简单的账号登录（用户名 + 密码），首次输入会自动创建账号
    if "current_user" not in st.session_state:
        st.session_state["current_user"] = ""
    if "auth_message" not in st.session_state:
        st.session_state["auth_message"] = ""

    with st.container():
        col_auth, col_msg = st.columns([1.2, 2.0])
        with col_auth:
            if st.session_state["current_user"]:
                current_name = st.session_state["current_user"]
                avatar_char = (current_name or "U").strip()[0]
                st.markdown(
                    f"""
                    <div class="user-status-pill">
                      <div class="user-avatar">{avatar_char}</div>
                      <div class="user-meta">
                        <div class="user-name">{current_name}</div>
                        <div class="user-sub">
                          <span class="user-dot online"></span>
                          <span class="user-sub-text">在线</span>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("退出登录", key="logout_btn"):
                    st.session_state["current_user"] = ""
                    st.session_state["auth_message"] = ""
            else:
                st.caption("（可选）登录后会自动保存你的生成记录，只想体验功能可以先不用登录。")
                with st.form("login_form", clear_on_submit=False):
                    username = st.text_input("账号（支持中文 / 英文）", key="login_username")
                    password = st.text_input("密码（首次输入即为注册密码）", type="password", key="login_password")
                    submitted = st.form_submit_button("登录 / 注册", type="primary", use_container_width=True)
                if submitted:
                    ok, msg = ensure_user(username, password)
                    st.session_state["auth_message"] = msg
                    if ok:
                        st.session_state["current_user"] = username.strip()

        with col_msg:
            if st.session_state.get("auth_message"):
                st.info(st.session_state["auth_message"])

    # 未登录时仍允许体验，但不会为其持久化历史记录
    current_user = (st.session_state.get("current_user") or "").strip()

    # Sidebar: API Keys + 系统状态
    with st.sidebar:
        st.header("① 填写 API Keys（必填）")
        st.caption(
            "至少要填 Doubao + DeepSeek + 一个视频生成 Key，系统才能完整跑完一键生成流程。"
        )

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
        minimax_key = st.text_input(
            "MiniMax API Key（视频生成）",
            type="password",
            value=defaults["minimax"],
            help="MiniMax 控制台获取的 API Key（用于 Hailuo 文生视频与 Anthropic 兼容接口）。",
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
        os.environ["ANTHROPIC_API_KEY"] = minimax_key or ""
        os.environ["KLING_ACCESS_KEY"] = kling_key or ""
        os.environ["KLING_SECRET_KEY"] = kling_secret or ""
        os.environ["TIANAPI_KEY"] = tianxing_key or ""

        st.markdown("---")
        st.subheader("系统状态")
        st.write(
            f"豆包 / Ark：{'✅ 已配置' if doubao_key else '⚠️ 未配置'}"
        )
        st.write(
            f"DeepSeek：{'✅ 已配置' if deepseek_key else '⚠️ 未配置'}"
        )
        st.write(
            f"MiniMax 视频：{'✅ 已配置' if minimax_key else '⚠️ 未配置'}"
        )
        st.write(
            f"Kling 视频：{'✅ 已配置' if (kling_key and kling_secret) else '⚠️ 未配置（需 Access Key + Secret Key）'}"
        )
        st.write(
            f"抖音热榜（TianAPI）：{'✅ 已配置' if tianxing_key else '⚠️ 未配置'}"
        )

    # ---------- 主页面：两步流程 ----------
    st.caption(
        "整套流程只有两步：① 先从抖音热榜里选一个话题；② 再点下面的「开始一键生成」，中间所有步骤系统会自动完成。"
    )
    st.markdown("---")

    # ---------- 步骤 1：选择热点话题 ----------
    st.subheader("① 选择热点话题（必做）")
    st.caption(
        "先点左侧按钮「获取今日抖音热榜」，再在表格中点选一条你感兴趣的话题，不会选就用排在最前面的一条。"
    )
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
    st.subheader("② 一键生成视频（只需点一次按钮）")

    # 让用户在生成前选择文生视频模型：MiniMax（推荐）或 Kling 多镜头
    st.markdown("**选择视频生成模型**")
    video_model_options = [
        "MiniMax Hailuo 2.3（推荐，单段 6 秒）",
        "Kling-v3 多镜头（原多镜头流程）",
    ]
    default_model = st.session_state.get("video_model") or video_model_options[0]
    video_model = st.selectbox(
        "画面由谁来生成？",
        options=video_model_options,
        index=video_model_options.index(default_model),
        help="MiniMax：一条 6 秒短视频，更快更简单；Kling：多镜头 15 秒，效果更复杂，等待时间更长。",
    )
    st.session_state["video_model"] = video_model

    # 成片画面比例：9:16 / 16:9 / 1:1 / 4:3 / 3:2
    st.markdown("**选择成片画面比例**")
    aspect_options = {
        "竖屏 9:16 · 抖音 / TikTok / 小红书": "9:16",
        "横屏 16:9 · 电影感 / 电脑端": "16:9",
        "方形 1:1 · 通用广告位": "1:1",
        "4:3 · 传统视频比例": "4:3",
        "3:2 · 摄影常见比例": "3:2",
    }
    ar_labels = list(aspect_options.keys())
    default_ar_value = st.session_state.get("aspect_ratio_value", "9:16")
    default_ar_label = next(
        (label for label, value in aspect_options.items() if value == default_ar_value),
        ar_labels[0],
    )
    aspect_label = st.selectbox(
        "希望成片是竖屏还是横屏？",
        options=ar_labels,
        index=ar_labels.index(default_ar_label),
        help="9:16：适合抖音、TikTok、小红书等竖屏短视频；16:9：适合电脑端、横屏投放；其他比例用于信息流广告位等特殊场景。",
    )
    aspect_ratio_value = aspect_options[aspect_label]
    st.session_state["aspect_ratio_value"] = aspect_ratio_value

    missing_reasons: List[str] = []
    if not selected_topic:
        missing_reasons.append("步骤 ① 还没从抖音热榜表格里选中一个话题。")
    if not doubao_key:
        missing_reasons.append("左侧未填 Doubao (Ark) API Key（必填，用来写脚本）。")
    if not deepseek_key:
        missing_reasons.append("左侧未填 DeepSeek API Key（必填，用来做分镜）。")
    if video_model.startswith("MiniMax"):
        if not minimax_key:
            missing_reasons.append("左侧未填 MiniMax API Key（必填，用来生成画面）。")
    else:
        if not (kling_key and kling_secret):
            missing_reasons.append("左侧未填完整的 Kling Access Key + Secret Key（必填，用来生成画面）。")
    if not tianxing_key:
        missing_reasons.append("左侧未填 TianAPI Key（用于拉取抖音热榜，建议填写）。")

    disabled_generate = not (
        selected_topic
        and doubao_key
        and deepseek_key
        and (
            (video_model.startswith("MiniMax") and minimax_key)
            or (video_model.startswith("Kling") and kling_key and kling_secret)
        )
    )
    if disabled_generate:
        st.warning("当前还不能点击「开始一键生成」，请先完成下面这些：")
        for reason in missing_reasons:
            st.write(f"- {reason}")
        if not missing_reasons:
            st.caption(
                "请先完成 ① 选择话题，并在左侧填好豆包、DeepSeek、MiniMax/Kling、TianAPI 的密钥。"
            )
    else:
        st.caption(
            "点击按钮后，系统会自动：写脚本 → 设计镜头 → 生成画面 → 配好旁白和中文字幕。"
            "整个过程大约 4～10 分钟，请耐心等待。"
        )
        st.caption(
            "生成过程中请不要关闭本页面或浏览器标签，否则任务会被中断。"
            "生成成功后，成片和记录会出现在下方「生成结果」和「我的历史记录」，刷新或重新登录都能再找回。"
        )
        # 配音音色选择（豆包语音 2.0）
        st.markdown("**选择配音声音**")
        st.caption(
            "上面生成的旁白，会由你选的声音读出来。\n"
            "- 常规声音：小天 2.0（默认）、Vivi 2.0、小何 2.0、云舟 2.0\n"
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
            "你想用哪种声音来配音？",
            options=voice_labels,
            index=voice_labels.index(default_label),
            help="可以多试几种声音风格，看哪种最适合这个话题。",
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
        # 进度条：按 4 个大步骤粗略更新 0 → 25 → 50 → 75 → 100
        progress_bar = st.progress(0)
        # Kling 目前官方支持的画面比例（其他比例通过后期裁剪成片）
        kling_supported_aspects = {"9:16", "16:9", "1:1"}
        ar_value = st.session_state.get("aspect_ratio_value", "9:16")
        kling_aspect = ar_value if ar_value in kling_supported_aspects else "9:16"

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
                    progress_bar.progress(25)

                    # 2. 优化分镜为英文 Prompt
                    status.update(label="步骤 2/4：正在优化分镜（DeepSeek 多镜头 prompt）...", state="running")
                    scenes = script.get("visual_scenes") or []
                    if not isinstance(scenes, list) or not scenes:
                        raise ArkAPIError("模型未返回 visual_scenes，用于视频分镜。")
                    optimized_prompts = optimize_visual_prompt(scenes)
                    st.session_state["optimized_prompts"] = optimized_prompts
                    status.write("分镜已优化为英文 Prompt（多镜头）。")
                    progress_bar.progress(50)

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
                    status.write("**英文分镜（用于生成画面）：**")
                    for idx, p in enumerate(optimized_prompts, start=1):
                        status.write(f"  {idx}. {p[:100]}{'…' if len(p) > 100 else ''}")

                    # 3. 可灵多镜头视频（画面+音效/环境音，与 TTS 旁白叠加为背景）
                    status.update(
                        label=(
                            "步骤 3/4：正在生成视频（MiniMax Hailuo 2.3，约 1～3 分钟）…"
                            if video_model.startswith("MiniMax")
                            else "步骤 3/4：正在生成多镜头视频（Kling-v3，约 2～10 分钟）…"
                        ),
                        state="running",
                    )
                    if video_model.startswith("MiniMax"):
                        # 将多镜头英文 prompt 合并为一段描述，供 MiniMax 使用
                        combined_prompt = " ".join(optimized_prompts)
                        video_url = generate_minimax_video(
                            combined_prompt,
                            model="MiniMax-Hailuo-2.3",
                            duration=6,
                            resolution="768P",
                            timeout=600,
                        )
                    else:
                        video_url = generate_kling_multishot(
                            optimized_prompts,
                            total_duration=15,
                            aspect_ratio=kling_aspect,
                            mode="pro",
                            sound="on",
                            timeout=900,
                        )
                    _download_file(video_url, raw_video_path)
                    progress_bar.progress(75)

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
                        target_aspect=ar_value,
                    )
                    status.update(label="生成完成！可以在下方预览与下载成片（含旁白、字幕与可选 BGM）。", state="complete")
                    progress_bar.progress(100)
            else:
                with st.spinner("正在一键生成完整视频，请稍候..."):
                    script = generate_video_script(selected_topic)
                    st.session_state["last_script"] = script
                    progress_bar.progress(25)

                    scenes = script.get("visual_scenes") or []
                    if not isinstance(scenes, list) or not scenes:
                        raise ArkAPIError("模型未返回 visual_scenes，用于视频分镜。")
                    optimized_prompts = optimize_visual_prompt(scenes)
                    st.session_state["optimized_prompts"] = optimized_prompts
                    progress_bar.progress(50)

                    video_url = generate_kling_multishot(
                        optimized_prompts,
                        total_duration=15,
                        aspect_ratio=kling_aspect,
                        mode="pro",
                        sound="on",
                        timeout=900,
                    )
                    _download_file(video_url, raw_video_path)
                    progress_bar.progress(75)
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
                        target_aspect=ar_value,
                    )
                    progress_bar.progress(100)

            final_path_str = str(final_video_path)
            st.session_state["final_video_path"] = final_path_str
            # 为当前登录用户持久化一条历史记录（含成片路径）
            if current_user:
                persisted_path = persist_video_for_user(current_user, final_path_str)
                history_item = {
                    "created_at": time.time(),
                    "topic": selected_topic,
                    "title": script.get("title", ""),
                    "video_path": persisted_path,
                    "model": video_model,
                    "narration_preview": (script.get("narration") or "")[:120],
                    "script": script,
                    "optimized_prompts": optimized_prompts,
                }
                append_history_item(current_user, history_item)
            st.balloons()
            st.success("🎉 短视频生成完成！可以在下方预览与下载（含旁白、字幕与可选 BGM）。")
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
            with st.expander("DeepSeek 英文分镜（用于生成画面）", expanded=False):
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

    # ---------- 登录用户的历史记录 ----------
    if current_user:
        st.markdown("---")
        st.subheader("我的历史记录")
        history = load_user_history(current_user)
        if not history:
            st.caption("登录状态下完成一键生成后，你最近的成片会出现在这里。")
        else:
            for idx, item in enumerate(history, start=1):
                title = item.get("title") or item.get("topic") or f"历史记录 {idx}"
                created_ts = item.get("created_at") or 0
                try:
                    created_str = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(float(created_ts))
                    )
                except Exception:
                    created_str = ""
                subtitle = item.get("topic") or ""
                model_used = item.get("model") or ""
                with st.expander(f"{idx}. {title}", expanded=(idx == 1)):
                    if created_str:
                        st.caption(f"生成时间：{created_str}")
                    if subtitle:
                        st.markdown(f"**话题：** {subtitle}")
                    if model_used:
                        st.markdown(f"**模型：** {model_used}")
                    preview = item.get("narration_preview") or ""
                    if preview:
                        st.markdown("**旁白片段预览：**")
                        st.write(preview + ("…" if len(preview) >= 120 else ""))

                    video_path_hist = item.get("video_path") or ""
                    if video_path_hist and Path(video_path_hist).is_file():
                        if st.button(
                            "加载到上方预览",
                            key=f"load_history_{idx}",
                        ):
                            st.session_state["final_video_path"] = video_path_hist
                            # 若存有脚本和分镜，也一起恢复，方便查看
                            if isinstance(item.get("script"), dict):
                                st.session_state["last_script"] = item["script"]
                            if isinstance(item.get("optimized_prompts"), list):
                                st.session_state["optimized_prompts"] = item["optimized_prompts"]
                            st.experimental_rerun()


if __name__ == "__main__":
    main()


