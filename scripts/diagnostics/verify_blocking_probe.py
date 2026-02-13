import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

def test_tool():
    from accel.mcp_server import _tool_verify
    
    result = _tool_verify(
        project='.',
        changed_files=['accel/mcp_server.py'],
        fast_loop=True
    )
    return result

print("Testing _tool_verify (synchronous) with 15s timeout...")
start = time.time()

try:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(test_tool)
        result = future.result(timeout=15)
        elapsed = time.time() - start
        print(f"Result after {elapsed:.1f}s: {result}")

except FuturesTimeoutError:
    elapsed = time.time() - start
    print(f"TIMEOUT after {elapsed:.1f}s - Process was BLOCKED!")

except Exception as e:
    elapsed = time.time() - start
    print(f"Error after {elapsed:.1f}s: {type(e).__name__}: {e}")
