# -*- coding: utf-8 -*-
"""自动续传守护：封禁则等待、解除则续传，直到全市场下载完成。

Baostock 免费账号有请求总量限制，全市场约 25 万次请求无法一次跑完，会反复
触发黑名单封禁（10001011）。01_download_a_share.py 已内置「黑名单熔断 +
断点续传」，本脚本在它外层再加一层自动重试，把「封了等、解除后续」自动化。

用法：
    python scripts/00_auto_resume_download.py

原理：
    每轮先轻量 login 探测封禁状态；
    - 仍封禁：等待 WAIT_SEC 后重试；
    - 已解除：运行 01_download_a_share.py，随后读日志判断
      * 含「下载完成」→ 全部下完，退出；
      * 否则（触发熔断）→ 等待 WAIT_SEC 后进入下一轮。
    已下载文件一律跳过，不会重复下载。
"""

import subprocess
import sys
import time
from pathlib import Path

import baostock as bs

CODE_DIR = Path(__file__).resolve().parent
DOWNLOAD_SCRIPT = CODE_DIR / "01_download_a_share.py"
LAST_RUN_LOG = CODE_DIR.parent / "data" / "_download_last_run.log"

WAIT_SEC = 900        # 封禁后每 15 分钟探测一次
MAX_ROUNDS = 200      # 保护上限，约 50 小时


def is_blacklisted() -> bool:
    """轻量探测：login 失败或返回非 0 即视为封禁中。"""
    try:
        login = bs.login()
        if login.error_code == "0":
            bs.logout()
            return False
        return True
    except Exception:
        return True


def main():
    for rnd in range(1, MAX_ROUNDS + 1):
        stamp = time.strftime("%H:%M:%S")
        if is_blacklisted():
            print(f"[{stamp}] 第 {rnd} 轮：封禁中，{WAIT_SEC // 60} 分钟后重试", flush=True)
            time.sleep(WAIT_SEC)
            continue

        print(f"[{stamp}] 第 {rnd} 轮：封禁已解除，启动下载", flush=True)
        with open(LAST_RUN_LOG, "w", encoding="utf-8") as log:
            subprocess.run([sys.executable, str(DOWNLOAD_SCRIPT)],
                           cwd=str(CODE_DIR), stdout=log, stderr=subprocess.STDOUT)

        out = LAST_RUN_LOG.read_text(encoding="utf-8", errors="ignore")
        if "下载完成" in out:
            print(f"[{time.strftime('%H:%M:%S')}] 全部下载完成，守护退出。", flush=True)
            return
        print(f"[{time.strftime('%H:%M:%S')}] 第 {rnd} 轮触发熔断，"
              f"{WAIT_SEC // 60} 分钟后重试", flush=True)
        time.sleep(WAIT_SEC)

    print("已达最大轮次仍未完成，请手动检查。", flush=True)


if __name__ == "__main__":
    main()
