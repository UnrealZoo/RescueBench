#!/usr/bin/env python3
"""
独立进程实时帧查看器
====================
在独立进程中使用 cv2.imshow 显示 benchmark 渲染帧，
避免与 Unreal Engine 的 X11 事件循环冲突。

用法:
    # 默认路径
    python frame_viewer.py

    # 自定义路径
    python frame_viewer.py --path /path/to/latest_frame.jpg

    # 调整刷新间隔 (毫秒)
    python frame_viewer.py --interval 30

快捷键:
    q / ESC  - 退出
    SPACE    - 暂停/恢复刷新
    s        - 保存当前帧截图
"""

import cv2
import os
import time
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description='独立进程实时帧查看器 (避免与 UE X11 冲突)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python frame_viewer.py
    python frame_viewer.py --path benchmark_results/_render_frames/latest_frame.jpg
    python frame_viewer.py --interval 30 --scale 1.5
        """
    )
    parser.add_argument('--path', type=str,
                        default='benchmark_results/_render_frames/latest_frame.jpg',
                        help='帧图片路径 (默认: benchmark_results/_render_frames/latest_frame.jpg)')
    parser.add_argument('--interval', type=int, default=50,
                        help='刷新间隔 (毫秒, 默认: 50, 即 ~20 FPS)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='显示缩放比例 (默认: 1.0)')
    parser.add_argument('--title', type=str, default='First Person View',
                        help='窗口标题')

    args = parser.parse_args()

    frame_path = args.path
    interval = args.interval
    scale = args.scale
    title = args.title

    print(f"{'='*50}")
    print(f"  独立进程帧查看器")
    print(f"{'='*50}")
    print(f"  监视文件: {frame_path}")
    fps = (1000 // interval) if interval > 0 else 0
    print(f"  刷新间隔: {interval} ms (~{fps} FPS)")
    print(f"  缩放比例: {scale}x")
    print(f"{'='*50}")
    print(f"  快捷键:")
    print(f"    q / ESC  - 退出")
    print(f"    SPACE    - 暂停/恢复")
    print(f"    s        - 保存截图")
    print(f"{'='*50}")

    # 等待文件出现
    if not os.path.exists(frame_path):
        print(f"\n⏳ 等待文件出现: {frame_path}")
        print(f"   (请确保 benchmark 已带 --render 参数启动)")
        while not os.path.exists(frame_path):
            time.sleep(0.5)
        print(f"✅ 文件已出现，开始显示\n")

    last_mtime = 0
    paused = False
    frame_count = 0
    current_img = None
    wait_ms = max(1, interval)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    while True:
        try:
            # 窗口被手动关闭后自动恢复，避免后续 imshow/waitKey 异常
            try:
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    cv2.destroyAllWindows()
                    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            except cv2.error:
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)

            if not paused and os.path.exists(frame_path):
                mtime = os.path.getmtime(frame_path)
                if mtime != last_mtime:
                    img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                    if img is not None and img.size > 0:
                        current_img = img
                        frame_count += 1

                        # 缩放
                        if scale != 1.0:
                            h, w = img.shape[:2]
                            new_w, new_h = int(w * scale), int(h * scale)
                            img = cv2.resize(img, (new_w, new_h),
                                             interpolation=cv2.INTER_LINEAR)

                        cv2.imshow(title, img)
                        last_mtime = mtime

            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord('q') or key == 27:  # q 或 ESC
                print("\n👋 退出查看器")
                break
            elif key == ord(' '):  # 空格：暂停/恢复
                paused = not paused
                status = "⏸️  已暂停" if paused else "▶️  已恢复"
                print(status)
            elif key == ord('s'):  # s：保存截图
                if current_img is not None:
                    screenshot_dir = "viewer_screenshots"
                    os.makedirs(screenshot_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = os.path.join(screenshot_dir,
                                                   f"screenshot_{ts}.jpg")
                    cv2.imwrite(screenshot_path, current_img)
                    print(f"📸 截图已保存: {screenshot_path}")

        except KeyboardInterrupt:
            print("\n👋 Ctrl+C 退出")
            break
        except cv2.error:
            # 文件切换/窗口状态变更引起的 OpenCV 异常，短暂等待后继续
            time.sleep(0.05)
        except Exception as e:
            # 文件可能正在被写入，忽略读取错误
            time.sleep(0.1)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


# # 终端 1: 运行 benchmark (带 --render)
# cd /media/littlecave/T9/Offline_RL_Active_Tracking/gym-rescue/benchmark
# python run_r2zeroshot.py --levels 1 --episodes 1 --render

# # 终端 2: 启动独立查看器
# cd /media/littlecave/T9/Offline_RL_Active_Tracking/gym-rescue/benchmark
# python frame_viewer.py