# VideoTaxi 2.0 - Streamlit 原型（Step 1）

本步骤完成了一个基础的 Streamlit 应用骨架：

- **侧边栏**：用于填写各类 API Key（Doubao、DeepSeek、Kling、Tianxing）。
- **主页面**：调用 **天行数据 / 天聚数行的抖音热榜 API**，展示当前热门话题表格。

## 1. 环境准备

建议使用 Python 3.10+。

```bash
cd e:/VideoTaxi2.0
python -m venv .venv
.venv\Scripts\activate  # PowerShell
pip install -r requirements.txt
```

## 2. 配置 API Key（可选但推荐）

在项目根目录创建或编辑 `.env` 文件，写入：

```bash
DOUBAO_API_KEY=你的豆包key
DEEPSEEK_API_KEY=你的DeepSeekkey
KLING_API_KEY=你的Klingkey
TIANXING_API_KEY=你的天行数据key
```

应用启动时会自动读取这些变量，并在侧边栏预填对应输入框。

> 抖音热榜接口文档（天行数据 / 天聚数行）：`https://www.tianapi.com/apiview/155`

## 3. 运行 Streamlit 应用

在虚拟环境激活状态下执行：

```bash
streamlit run app.py
```

浏览器会自动打开本地地址（通常为 `http://localhost:8501`），即可看到：

- 左侧：API Key 配置面板。
- 右侧：抖音 / TikTok 热门话题表格（依赖 `TIANXING_API_KEY` 有效）。

接下来可以在此基础上继续实现：

- 热点内容的详情（标题、文案、热门评论、BGM 链接）。
- 调用 DeepSeek 进行剧本生成。
- 调用 TTS 与可灵生成音频 / 视频素材。
- 最终在前端整合展示与下载入口。

