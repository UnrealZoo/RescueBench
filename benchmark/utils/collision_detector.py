"""碰撞检测器 — 支持 API / 位置检测两种方式"""

import numpy as np


class CollisionDetector:
    """碰撞检测器 (api: unrealcv get_hit, position: 位置变化判断)"""

    def __init__(self, env, method: str = 'api'):
        self.env = env
        self.method = method
        self.last_position = None
        self.collision_cooldown = 0

    def reset(self):
        self.last_position = None
        self.collision_cooldown = 0

    def check(self) -> bool:
        """检测是否发生碰撞，带 5 步冷却避免重复计数"""
        if self.collision_cooldown > 0:
            self.collision_cooldown -= 1
            return False

        try:
            env = self.env.unwrapped
            player = env.player_list[env.protagonist_id]

            if self.method == 'api':
                is_hit = env.unrealcv.get_hit(player)
                if is_hit:
                    self.collision_cooldown = 5
                return is_hit
            else:
                current_pos = env.obj_poses[env.protagonist_id][:3]
                if self.last_position is not None:
                    movement = np.linalg.norm(
                        np.array(current_pos) - np.array(self.last_position)
                    )
                    if movement < 1.0:
                        self.collision_cooldown = 5
                        self.last_position = current_pos
                        return True
                self.last_position = current_pos
                return False

        except Exception as e:
            print(f"[Warning] Collision detection failed: {e}")
            return False
