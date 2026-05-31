"""
拓扑图路径解析（visualnav-transformer / ViNT / NOMAD 专用）

与 example/VINT/collect_rescue_topomap.py 的 --topomap-root 目录约定一致。
"""

import os
from typing import Optional


def is_topomap_multimap_root(path: Optional[str]) -> bool:
    """若目录下存在 UnrealRescue-* 子文件夹，则视为「多地图拓扑根目录」。"""
    if not path or not os.path.isdir(path):
        return False
    try:
        for name in os.listdir(path):
            if name.startswith('UnrealRescue-') and os.path.isdir(os.path.join(path, name)):
                return True
    except OSError:
        return False
    return False


def resolve_topomap_dir_for_env(base: Optional[str], env_id: str) -> Optional[str]:
    """
    多地图布局：base 下按 env_id 分子目录。
    单地图布局：base 即为该地图拓扑目录，返回 base。
    """
    if not base:
        return None
    if not env_id or not is_topomap_multimap_root(base):
        return base
    sub = os.path.join(base, env_id)
    if os.path.isdir(sub):
        return sub
    # dooropen 变体默认复用同地图拓扑目录（不强制单独采集一套）。
    if env_id.endswith('_dooropen'):
        fallback_env_id = env_id[:-len('_dooropen')]
        fallback_sub = os.path.join(base, fallback_env_id)
        if os.path.isdir(fallback_sub):
            print(
                f"[Topomap] 未找到 {sub}，回退使用 {fallback_sub} "
                f"（当前 env_id={env_id}）"
            )
            return fallback_sub
    print(f"[Topomap] 多地图根目录下缺少子目录: {sub}（当前 env_id={env_id}）")
    return None
