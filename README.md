## VideoTaxi 2.0 代码说明文档

VideoTaxi 2.0 是一个「一键生成热点搞笑短视频」的小工具，从「抖音热榜话题」到「成片 MP4」的全流程自动化已经打通：

1. 获取抖音热榜（天行数据 TianAPI）
2. Exa 实时搜索补充话题的客观事实、新闻与数据
3. Ark 大模型（Kimi / 豆包）在事实基础上生成脚本（标题 + 旁白 + 5 条画面要点）
4. DeepSeek 优化画面要点为英文多镜头 Prompt
5. 文生视频（可选）  
   - MiniMax Hailuo 2.3：推荐，单段 6 秒视频  
   - 可灵 Kling-v3：多镜头 15 秒视频
6. 豆包语音 2.0（WebSocket）生成旁白语音（可选字级时间戳）
7. MoviePy + PIL 自动剪辑，对齐节奏并用自带 `font.ttf` 合成带中文字幕的最终 MP4

本说明文档主要介绍代码结构、关键模块和运行方式。

---

### 1. 环境准备

- **Python 版本**：建议 Python 3.10+
- 安装依赖：

```bash
cd e:/VideoTaxi2.0
python -m venv .venv
.venv\Scripts\activate  # PowerShell
pip install -r requirements.txt
```

`requirements.txt` 核心依赖：

- `streamlit`：前端界面
- `requests`：HTTP 请求
- `pandas`：表格展示
- `python-dotenv`：加载 `.env` 配置
- `moviepy`：视频剪辑与字幕合成

---

### 2. 配置说明（.env 与 Streamlit secrets）

项目支持两种配置方式：

- 根目录 `.env`
- `.streamlit/secrets.toml`

二者字段尽量对齐，方便本地脚本和 Streamlit 共享。

常用配置示例（`.env`）：

```bash
# --- 聚合数据与搜索 ---
TIANAPI_KEY=你的天行数据抖音热榜key

# --- 脚本生成（火山方舟 Ark：Kimi / 豆包）---
ARK_API_KEY=你的火山方舟 Ark API Key
# 推荐 Kimi 联网写脚本，减少编造；豆包可改为 doubao-1.5-pro-32k 等
ARK_MODEL_ID=kimi-k2-thinking-251104

# --- 事实检索（Exa Web Search）---
EXA_API_KEY=你的 Exa API Key

# --- DeepSeek 提示词优化 ---
DEEPSEEK_API_KEY=你的DeepSeek API Key

# --- 视频生成（MiniMax Hailuo / 可灵 Kling）---
# MiniMax（推荐）：使用 ANTHROPIC_API_KEY 或 MINIMAX_API_KEY
ANTHROPIC_API_KEY=你的 MiniMax API Key
KLING_ACCESS_KEY=你的Kling API Key
KLING_SECRET_KEY=（如平台要求可选）

# --- 语音合成（豆包语音 2.0 WebSocket）---
VOLC_APPID=火山控制台-豆包语音-APP ID
VOLC_ACCESS_TOKEN=火山控制台-豆包语音-Access Token

# --- 可选 BGM（洗脑神曲等，自动混入成片作背景）---
BGM_PATH=本地 MP3 文件路径（与 BGM_URL 二选一）
# 或
BGM_URL=直链 MP3 地址（如从 Pixabay Music、爱给网 等下载的免费可商用音乐）

# --- FFmpeg 路径（可选，本地使用；云端通常无需设置）---
FFMPEG_CMD=E:\ffmpeg\ffmpeg.exe
```

在 `.streamlit/secrets.toml` 中同名字段也可以配置一份，**推荐用于 Streamlit Cloud 部署**，示例：

```toml
TIANAPI_KEY = "你的天行数据 key"
ARK_API_KEY = "你的 Ark API Key"
ARK_MODEL_ID = "kimi-k2-thinking-251104"
EXA_API_KEY = "你的 Exa API Key"
DEEPSEEK_API_KEY = "你的 DeepSeek Key"
ANTHROPIC_API_KEY = "你的 MiniMax API Key"
KLING_ACCESS_KEY = "你的 Kling Access Key"
KLING_SECRET_KEY = "你的 Kling Secret Key"
VOLC_APPID = "你的豆包语音 AppID"
VOLC_ACCESS_TOKEN = "你的豆包语音 AccessToken"
```

---

### 3. 代码结构总览

- `app.py`  
  Streamlit 主入口，负责 UI 和整体流程编排：
  - 侧边栏：配置 Ark / DeepSeek / MiniMax / Kling / TianAPI / 豆包语音等 Key，并展示每项「✅ 已配置 / ⚠️ 未配置」状态。
  - 主区域：
    - 步骤 ①：获取抖音热榜、选择一个话题。
    - 步骤 ②：选择视频生成模型（MiniMax Hailuo 或 Kling）、配音音色，一键生成并预览/下载成片。

- `utils/api_clients.py`  
  - `get_douyin_hot_trends(limit)`：调用天行数据抖音热榜 API，拉取当前热门话题。  
  - `fetch_topic_facts_with_exa(topic)`：调用 Exa Web Search，围绕话题检索 5～6 条最新事实/新闻/数据，整理为「事实材料」文本。
  - `generate_video_script(topic)`：
    - 先根据 ARK_MODEL_ID 判断是否使用 Kimi；
    - 若是 Kimi，则先用 Exa 获取事实材料，并放入 system prompt，约束模型不要编造具体时间/事件/数字；
    - 调用 Ark 接口（优先 `/responses`，失败回退到 `/chat/completions`）生成 JSON 脚本：
      - `title`：视频标题
      - `narration`：旁白文案（搞笑风格、断句清晰，将原样用于 TTS + 字幕）
      - `visual_scenes`：5 条基于事实提炼的「画面要点」
      - `bgm_style`：BGM 风格建议。
  - `optimize_visual_prompt(chinese_scenes_list)`：调用 DeepSeek，将中文画面要点转为包含镜头运动、景别、光效等信息的英文 Prompt 列表。

- `utils/media_generators.py`  
  - `generate_kling_video(prompt)` / `generate_kling_multishot(prompts)`：  
    使用可灵 Kling 文生视频 API（单镜头 / 多镜头），基于 prompt 列表生成 9:16 短视频，返回下载 URL。
  - `generate_minimax_video(prompt, model="MiniMax-Hailuo-2.3", duration=6, resolution="768P")`：  
    使用 MiniMax 文生视频 API 创建任务并轮询结果，最终通过 `/v1/files/retrieve` 拿到 `download_url`，生成单段 6 秒视频。
  - `generate_tts_audio(text, output_path, enable_timestamp=True, speaker=...)`：  
    使用豆包语音 2.0 WebSocket API：
    - 使用用户在 UI 中选择的 2.0 音色（通用音色 + 多个「视频配音」角色音色）；
    - 若开启时间戳，则返回 `{"duration": ..., "words": [...]}` 结构，包含字级 `start_time` / `end_time`，用于字幕打轴；
    - 将音频写入 `output_path`（如 `temp/audio.mp3`）。

- `utils/video_assembler.py`  
  - `assemble_final_video(video_path, audio_path, script_text, output_path, timestamps=None, bgm_path=None)`  
    使用 MoviePy + PIL 自动合成最终成片：
    1. 加载 `video_path` 和 `audio_path`，计算视频与音频时长；
    2. 使用 `vfx.speedx` 按比例统一加速/减速整条视频，使视频总长与旁白总长一致（不重复某一段画面）；
    3. 设置音轨为 TTS 音频（可选额外 BGM：循环 + 降音量后混入）；
    4. 字幕逻辑：
       - 若提供 `timestamps`（豆包语音返回的 `words`）：  
         使用字级时间戳精确打轴；
       - 若没有时间戳：  
         将 `script_text` 按句号等符号粗略分句，按视频时长平均分配时间段；
       - 字幕渲染完全使用 PIL + 项目根目录的 `font.ttf`：
         白字 + 黑描边 + 半透明黑底，底部居中，确保中文在本地与云端环境下都能正确显示。
    5. 调用 `write_videofile` 输出 `output_path`（mp4），使用 `libx264 + aac`。

- `temp/`  
  中间与最终的媒体文件输出目录：
  - `kling_raw.mp4`：可灵生成的原始视频
  - `tts_audio.mp3`：豆包语音 2.0 生成的音频
  - `final_video.mp4`：MoviePy 合成后的最终成片

---

### 4. 主流程：`app.py`

`app.py` 是整个项目的「中控台」，大致分为三块：

- **侧边栏：Key 配置 + 系统状态**
  - 读取 `.env` 与 `st.secrets` 预填 Ark（Kimi/豆包）/ DeepSeek / Kling / TianAPI Key。
  - 显示每个服务是否「✅ 已配置」或「⚠️ 未配置」。

- **主区域上半部分：抖音热榜**
  1. 用户点击「获取今日抖音热榜」，调用 `get_douyin_hot_trends(limit)`。
  2. 使用 `st.dataframe` 展示话题标题与热度。
  3. 用户从下拉框中选定一个话题作为后续生成的主题。

- **主区域下半部分：一键生成短视频**
  点击「开始一键生成」后，流程如下：

1. **生成脚本（Ark：Kimi 或豆包）**  
   - 调用 `generate_video_script(selected_topic)`  
   - 得到标题、搞笑旁白与中文分镜。

2. **优化分镜（DeepSeek）**  
   - 调用 `optimize_visual_prompt(visual_scenes)`  
   - 每条中文分镜对应一个英文 Prompt，用于可灵视频生成。

3. **生成视频 + 语音（Kling + 豆包语音 2.0）**  
   - 用优化后 Prompt 拼成一个长 prompt：
     - 调用 `generate_kling_video(prompt)`，拿到视频 URL 并下载到 `temp/kling_raw.mp4`。
   - 用旁白文案调用 `generate_tts_audio(narration, "temp/tts_audio.mp3", enable_timestamp=True)`：
     - 保存音频到 `temp/tts_audio.mp3`
     - 返回（如支持）字级时间戳结构用于字幕。

4. **自动合成（MoviePy）**  
   - 调用 `assemble_final_video("temp/kling_raw.mp4", "temp/tts_audio.mp3", narration, "temp/final_video.mp4", timestamps=tts_meta)`：
     - 自动对齐视频时长与音频时长
     - 尝试生成底部居中的黄色描边大字幕（优先用时间戳；无时间戳则平均分句）
     - 输出最终 MP4 并在页面中预览 + 提供下载按钮。

整个过程在界面上通过 `st.status`（或回退到 `st.spinner`）展示为：

- 正在生成文案
- 正在优化分镜
- 正在生成视频 / 语音
- 正在合成

---

### 5. 运行应用

在虚拟环境激活、依赖安装完成后，运行：

```bash
streamlit run app.py
```

浏览器将打开（通常是 `http://localhost:8501`），你可以：

1. 在侧边栏填入各类 API Key（或依赖 `.env` / `secrets.toml` 的预填）。
2. 点击 **「获取今日抖音热榜」**，选择一个话题。
3. 点击 **「开始一键生成」**，等待流程跑完。
4. 在页面下半部分预览生成视频，并点击按钮下载最终 MP4。

至此，VideoTaxi 2.0 的「从热点到成片」自动化流水线就完整跑通了。
