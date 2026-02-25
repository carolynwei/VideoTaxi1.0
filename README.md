## VideoTaxi 2.0 代码说明文档

VideoTaxi 2.0 是一个「一键生成热点搞笑短视频」的小工具，完整打通了：

1. 获取抖音热榜
2. Doubao / Ark 生成脚本（标题 + 旁白 + 分镜）
3. DeepSeek 优化分镜为英文 Prompt
4. 可灵 Kling 生成文生视频
5. 火山引擎 TTS 生成搞怪语音（可选字级时间戳）
6. MoviePy 自动剪辑，合成带字幕的最终 MP4

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

# --- Doubao / Ark 大模型 ---
ARK_API_KEY=你的火山豆包Ark API Key
ARK_MODEL_ID=你的Ark模型ID（如 doubao-seed-1-8-xxxxxx）

# --- DeepSeek 提示词优化 ---
DEEPSEEK_API_KEY=你的DeepSeek API Key

# --- 视频生成（可灵 Kling）---
KLING_ACCESS_KEY=你的Kling API Key
KLING_SECRET_KEY=（如平台要求可选）

# --- 语音合成（火山引擎 TTS）---
VOLC_APPID=你的火山AppId（用作 appkey）
VOLC_ACCESS_TOKEN=你的火山访问 token
VOLC_CLUSTER_ID=volcano_tts

# --- FFmpeg 路径（可选）---
FFMPEG_CMD=E:\ffmpeg\ffmpeg.exe
```

在 `.streamlit/secrets.toml` 中同名字段也可以配置一份，优先级通常高于环境变量。

---

### 3. 代码结构总览

- `app.py`  
  Streamlit 主入口，负责 UI 和整体流程编排。

- `utils/api_clients.py`  
  - `generate_video_script(topic)`：调用 Doubao / Ark，生成 JSON 结构的短视频脚本：
    - `title`：视频标题
    - `narration`：旁白文案（搞笑风格）
    - `visual_scenes`：中文分镜列表
    - `bgm_style`：BGM 风格建议  
  - `get_douyin_hot_trends(limit)`：调用天行数据抖音热榜 API，拉取当前热门话题。  
  - `optimize_visual_prompt(chinese_scenes_list)`：调用 DeepSeek，将中文分镜转为高质量英文 Prompt 列表。

- `utils/media_generators.py`  
  - `generate_kling_video(prompt)`：  
    使用 `requests` 调用可灵 Kling 文生视频 API：
    1. `POST https://api.klingai.com/v1/videos/text2video` 创建任务，获取 `task_id`
    2. 每 10 秒轮询 `GET .../text2video/{task_id}`，直到状态成功或失败
    3. 成功时返回视频下载 URL，失败或超时抛出 `KlingAPIError`
  - `generate_tts_audio(text, output_path, enable_timestamp=True)`：  
    使用火山引擎 TTS HTTP API：
    - `speaker` 固定为适合搞笑短视频的音色（如 `zh_male_sunwukong_clone2`）
    - 请求体中开启 `audio_config.enable_timestamp = true`  
      若后端支持，将在 `payload` 中返回：
      - `duration`：音频总时长
      - `words`：字级时间戳
      - `phonemes`：音素级时间戳
    - 将 Base64 音频数据解码并保存为 `output_path`（如 `temp/audio.mp3`）
    - 返回解析后的时间戳结构（如可用），否则返回 `None`

- `utils/video_assembler.py`  
  - `assemble_final_video(video_path, audio_path, script_text, output_path, timestamps=None)`  
    使用 MoviePy 自动合成最终成片：
    1. 加载 `video_path` 和 `audio_path`，计算时长
    2. 若视频比音频短：循环拼接视频并裁剪到音频时长
    3. 若视频比音频长：裁剪视频到音频时长
    4. 设置音轨为 TTS 音频
    5. 字幕逻辑：
       - 若提供 `timestamps`（火山 TTS payload）：  
         按字级 `start_time` / `end_time` 生成黄色描边大字字幕，底部居中逐字滚动。
       - 若没有时间戳：  
         将 `script_text` 按句号等符号粗略分句，按视频时长平均分配时间段，作为分段文案字幕。
    6. 调用 `write_videofile` 输出 `output_path`（mp4），使用 `libx264 + aac`。

- `temp/`  
  中间与最终的媒体文件输出目录：
  - `kling_raw.mp4`：可灵生成的原始视频
  - `tts_audio.mp3`：火山 TTS 生成的音频
  - `final_video.mp4`：MoviePy 合成后的最终成片

---

### 4. 主流程：`app.py`

`app.py` 是整个项目的「中控台」，大致分为三块：

- **侧边栏：Key 配置 + 系统状态**
  - 读取 `.env` 与 `st.secrets` 预填 Doubao / DeepSeek / Kling / TianAPI Key。
  - 显示每个服务是否「✅ 已配置」或「⚠️ 未配置」。

- **主区域上半部分：抖音热榜**
  1. 用户点击「获取今日抖音热榜」，调用 `get_douyin_hot_trends(limit)`。
  2. 使用 `st.dataframe` 展示话题标题与热度。
  3. 用户从下拉框中选定一个话题作为后续生成的主题。

- **主区域下半部分：一键生成短视频**
  点击「开始一键生成」后，流程如下：

1. **生成脚本（Doubao / Ark）**  
   - 调用 `generate_video_script(selected_topic)`  
   - 得到标题、搞笑旁白与中文分镜。

2. **优化分镜（DeepSeek）**  
   - 调用 `optimize_visual_prompt(visual_scenes)`  
   - 每条中文分镜对应一个英文 Prompt，用于可灵视频生成。

3. **生成视频 + 语音（Kling + 火山 TTS）**  
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
