import os
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv


TIANAPI_DOUYIN_HOT_URL = "https://apis.tianapi.com/douyinhot/index"


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


def fetch_douyin_hot(api_key: str, top_n: int = 10) -> pd.DataFrame:
    """
    Fetch Douyin/TikTok trending topics from Tianxing (TianAPI).

    Docs: https://www.tianapi.com/apiview/155
    """
    if not api_key:
        return pd.DataFrame()

    try:
        resp = requests.get(
            TIANAPI_DOUYIN_HOT_URL,
            params={"key": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        # Tianxing standard response format
        if data.get("code") != 200:
            st.error(f"抖音热榜 API 调用失败：{data.get('msg', '未知错误')}")
            return pd.DataFrame()

        items: List[Dict[str, Any]] = data.get("result", {}).get("list", [])[:top_n]
        if not items:
            return pd.DataFrame()

        records = [
            {
                "排名": idx + 1,
                "话题": item.get("word"),
                "标签": item.get("label"),
                "热度指数": item.get("hotindex"),
            }
            for idx, item in enumerate(items)
        ]
        return pd.DataFrame(records)
    except Exception as e:  # noqa: BLE001
        st.error(f"请求抖音热榜失败：{e}")
        return pd.DataFrame()


def main() -> None:
    st.set_page_config(
        page_title="VideoTaxi 热点脚本助手",
        layout="wide",
    )

    defaults = load_env_defaults()

    # Sidebar: API Keys
    with st.sidebar:
        st.header("API Keys 配置")
        st.caption("在这里配置用于后续步骤的各类大模型 / 数据接口密钥。")

        doubao_key = st.text_input(
            "Doubao API Key",
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
            "Tianxing API Key（用于抖音热榜）",
            type="password",
            value=defaults["tianxing"],
            help="在天行数据 / 天聚数行平台注册并申请 douyinhot 接口后获得。",
        )

        st.markdown("---")
        st.caption(
            "提示：可以将以上密钥写入项目根目录的 `.env` 文件，"
            "变量名分别为 `ARK_API_KEY`、`DEEPSEEK_API_KEY`、"
            "`KLING_ACCESS_KEY`、`TIANAPI_KEY`，应用会自动读取；"
            "或在 `.streamlit/secrets.toml` 中配置同名字段。"
        )

    # Main page: Douyin/TikTok trending topics table
    st.title("当前抖音 / TikTok 热门话题")
    st.write("本页用于展示来自天行数据抖音热搜榜 API 的实时热门话题。")

    col_controls, col_table = st.columns([1, 3], gap="large")

    with col_controls:
        top_n = st.slider("展示条数", min_value=5, max_value=50, value=10, step=5)
        auto_refresh = st.checkbox("加载页面时自动获取", value=True)
        refresh_btn = st.button("手动刷新热榜")

    should_fetch = (auto_refresh and tianxing_key) or refresh_btn

    with col_table:
        if not tianxing_key:
            st.info("请先在左侧填写 Tianxing API Key，才能获取抖音热榜数据。")
        elif should_fetch:
            df = fetch_douyin_hot(tianxing_key, top_n=top_n)
            if df.empty:
                st.warning("暂时没有获取到热榜数据，请稍后重试或检查 API 配置。")
            else:
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("勾选“加载页面时自动获取”或点击“手动刷新热榜”开始加载数据。")


if __name__ == "__main__":
    main()

