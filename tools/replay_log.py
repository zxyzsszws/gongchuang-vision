"""复盘工具：python tools/replay_log.py <file.jsonl> —— 每行打印 `t state event` 摘要。

state 取 fsm_state 字段（缺省 "-"），event 取 msg。文件缺失或行不可解析时
打印明确错误并以非零码退出。
"""
import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="回放 JSONL 运行日志")
    ap.add_argument("log_file", help="setup_logging 产生的 .jsonl 文件")
    args = ap.parse_args(argv)

    try:
        f = open(args.log_file, encoding="utf-8")
    except OSError as e:
        print(f"error: cannot open log file {args.log_file}: {e}", file=sys.stderr)
        return 1

    with f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"error: unparseable line {i} in {args.log_file}: {e}",
                      file=sys.stderr)
                return 1
            t = rec.get("t", "-")
            state = rec.get("fsm_state", "-")
            event = rec.get("msg", "")
            print(f"{t} {state} {event}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
