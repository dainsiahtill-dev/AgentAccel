import inspect

import accel.mcp_server as ms
from accel.mcp_server import create_server


def main() -> None:
    server = create_server()
    print("Testing accel_verify_start...")
    tool_fn = server.tool(name_or_fn="accel_verify_start")
    print(f"Tool function signature: {inspect.signature(tool_fn)}")

    try:
        result = tool_fn(".", None, False, False)
        print(f"Result with positional args: {result}")
    except Exception as exc:  # noqa: BLE001
        print(f"Positional args error: {exc}")

    try:
        result = tool_fn(project=".")
        print(f"Result with kwargs: {result}")
    except Exception as exc:  # noqa: BLE001
        print(f"Kwargs error: {exc}")

    print("\nChecking original tool function...")
    if hasattr(ms, "accel_verify_start"):
        fn = getattr(ms, "accel_verify_start")
        print(f"accel_verify_start type: {type(fn)}")
        print(f"accel_verify_start signature: {inspect.signature(fn)}")
    else:
        print("accel_verify_start is registered on FastMCP server, not module-level export.")


if __name__ == "__main__":
    main()
