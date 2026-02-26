from __future__ import annotations

"""
非常轻量的用户与历史记录存储：

- 账号与密码哈希保存在项目根目录下的 `user_data/<username>/profile.json`
- 每个用户的生成历史存为 `history` 列表，并且视频文件会复制到
  `user_data/<username>/videos/` 下，避免 temp 目录被清理后找不到成片。

说明：这是面向单机/小规模内网使用的简单方案，并不适合严格的生产级账号体系。
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 所有用户相关数据都放在项目根目录下，便于备份与迁移
BASE_DIR = Path(__file__).resolve().parent.parent
USER_DATA_ROOT = BASE_DIR / "user_data"
USER_DATA_ROOT.mkdir(parents=True, exist_ok=True)


def _safe_username(raw: str) -> str:
    """
    将用户名转换成安全的文件夹名：去掉首尾空格、小写、只保留简单字符。
    """
    name = (raw or "").strip().lower()
    if not name:
        raise ValueError("用户名不能为空。")
    # 只保留字母、数字、下划线和中划线，其余替换成下划线
    safe_chars = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-"):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    safe = "".join(safe_chars)
    if not safe:
        raise ValueError("用户名格式非法。")
    return safe


def _user_dir(username: str) -> Path:
    return USER_DATA_ROOT / _safe_username(username)


def _profile_path(username: str) -> Path:
    return _user_dir(username) / "profile.json"


def _password_hash(username: str, password: str) -> str:
    """
    使用用户名 + 密码进行 SHA256 哈希（简单场景足够，不涉及高强度安全要求）。
    """
    base = f"{_safe_username(username)}::{password}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def ensure_user(username: str, password: str) -> Tuple[bool, str]:
    """
    登录 / 注册合一：
    - 如果用户不存在：用当前密码创建账号；
    - 如果已存在：校验密码是否一致。

    返回 (success, message)；success 为 False 时，message 为错误原因。
    """
    if not password:
        return False, "密码不能为空。"

    udir = _user_dir(username)
    profile_file = _profile_path(username)

    if not profile_file.is_file():
        udir.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {
            "username": _safe_username(username),
            "display_name": username.strip(),
            "password_hash": _password_hash(username, password),
            "created_at": time.time(),
            "history": [],
        }
        profile_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True, "已为你创建新账号。"

    try:
        data = json.loads(profile_file.read_text(encoding="utf-8") or "{}")
    except Exception:
        return False, "用户数据读取失败，请联系管理员修复 profile.json。"

    expected = data.get("password_hash") or ""
    if not expected:
        return False, "用户密码未正确保存，请联系管理员重置。"

    if expected != _password_hash(username, password):
        return False, "密码不正确，请重试。"

    return True, "登录成功。"


def load_user_history(username: str) -> List[Dict[str, Any]]:
    """
    读取用户历史记录列表（最新的在前）。
    """
    profile_file = _profile_path(username)
    if not profile_file.is_file():
        return []
    try:
        data = json.loads(profile_file.read_text(encoding="utf-8") or "{}")
    except Exception:
        return []
    history = data.get("history") or []
    if not isinstance(history, list):
        return []
    return history


def append_history_item(username: str, item: Dict[str, Any], *, max_items: int = 20) -> None:
    """
    为指定用户追加一条历史记录，自动限制数量。
    """
    profile_file = _profile_path(username)
    if profile_file.is_file():
        try:
            data = json.loads(profile_file.read_text(encoding="utf-8") or "{}")
        except Exception:
            data = {}
    else:
        data = {}

    history = data.get("history")
    if not isinstance(history, list):
        history = []

    # 最新记录插到前面
    history.insert(0, item)
    if len(history) > max_items:
        history = history[:max_items]

    data["username"] = data.get("username") or _safe_username(username)
    data.setdefault("display_name", username.strip())
    data["history"] = history

    profile_file.parent.mkdir(parents=True, exist_ok=True)
    profile_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def persist_video_for_user(username: str, src_path: str) -> str:
    """
    将生成好的 MP4 拷贝到该用户专属目录下，并返回新路径（字符串）。
    如果拷贝失败，则返回原路径。
    """
    from shutil import copy2

    src = Path(src_path)
    if not src.is_file():
        return src_path

    user_videos_dir = _user_dir(username) / "videos"
    user_videos_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    dst = user_videos_dir / f"{_safe_username(username)}_{ts}.mp4"
    try:
        copy2(src, dst)
        return str(dst)
    except Exception:
        return src_path


