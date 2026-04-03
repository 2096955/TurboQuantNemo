import json
import urllib.request
import sys
import argparse

base = "http://127.0.0.1:2024"


def post(path, payload):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as f:
            return json.load(f)
    except urllib.error.URLError as e:
        print(f"Failed to connect to LangGraph Server at {base}. Is it running?")
        print(f"Error: {e}")
        sys.exit(1)


parser = argparse.ArgumentParser(
    description="Verify DeerFlow TurboQuant tool-calling wiring."
)
parser.add_argument(
    "--model-name",
    default="nemotron-120b",
    help="Model name in deer-flow/config.yaml",
)
parser.add_argument(
    "--bootstrap",
    action="store_true",
    help="Use is_bootstrap=true (useful for smaller models like 1.5B)",
)
args = parser.parse_args()


def _normalize_readback(value):
    """Normalize read_file payloads into comparable plain text."""
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text.replace("\r\n", "\n")


print("Starting TurboQuant Wiring Verification...")
print("1. Creating a new thread on LangGraph server...")
thread = post("/threads", {})
tid = thread["thread_id"]
print(f"   Created thread: {tid}")

if args.bootstrap:
    prompt = (
        "You are a helpful assistant with access to tools. "
        "Please use the 'write_file' tool to create the file '/mnt/user-data/test_tq.txt' with the content 'hello\\n'. "
        "Then, use the 'read_file' tool to read '/mnt/user-data/test_tq.txt'. "
        "IMPORTANT: You must output a valid JSON tool call."
    )
else:
    # Default-path prompt for larger models with the full lead_agent system prompt.
    prompt = (
        "Use write_file to create /mnt/user-data/test_tq.txt with content exactly 'hello\\n'. "
        "Then use read_file to read the same path and return only the file content."
    )

req = {
    "assistant_id": "lead_agent",
    "input": {"messages": [{"role": "user", "content": prompt}]},
    "config": {
        "configurable": {
            "model_name": args.model_name,
            "thread_id": tid,
            "is_bootstrap": args.bootstrap,
        }
    },
}

print(
    f"2. Triggering run on thread {tid} using model={args.model_name}, bootstrap={args.bootstrap} ..."
)
res = post(f"/threads/{tid}/runs/wait", req)
msgs = res.get("messages", [])

tool_calls = False
executed_tools = []
read_outputs = []
print("\n--- Model Interaction Log ---")
for m in msgs:
    if m.get("type") == "ai":
        if "tool_calls" in m and m["tool_calls"]:
            tool_calls = True
            calls = [tc["name"] for tc in m["tool_calls"]]
            print(f"AI requested tools: {calls}")
    elif m.get("type") == "tool":
        executed_tools.append(m.get("name"))
        content = (m.get("content") or "").strip()
        print(f"TOOL {m.get('name')} -> {content}")
        if m.get("name") == "read_file":
            read_outputs.append(content)

print("\n--- Final Answer ---")
ai_msgs = [m for m in msgs if m.get("type") == "ai"]
if ai_msgs:
    print(ai_msgs[-1].get("content", "")[:500])

print("\n--- Verification Result ---")
has_write = "write_file" in executed_tools
has_read = "read_file" in executed_tools
readback_ok = any(_normalize_readback(output) == "hello" for output in read_outputs)

if tool_calls and has_write and has_read and readback_ok:
    print(
        "SUCCESS: Structured tool calls were emitted, write_file/read_file executed, and readback matched expected content."
    )
else:
    print("FAILURE: Verification criteria were not met.")
    print(
        f"  tool_calls={tool_calls}, has_write={has_write}, has_read={has_read}, readback_ok={readback_ok}"
    )
    if read_outputs:
        print(f"  read_outputs={read_outputs!r}")
    sys.exit(1)
