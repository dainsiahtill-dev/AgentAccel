import time
import inspect
from accel.mcp_server import create_server

server = create_server()

print("Testing accel_verify_start...")
tool_fn = server.tool('accel_verify_start')
print(f"Tool function signature: {inspect.signature(tool_fn)}")

# Try calling with positional args
try:
    result = tool_fn('.', None, False, False)
    print(f"Result with positional args: {result}")
except Exception as e:
    print(f"Positional args error: {e}")

# Try calling with kwargs
try:
    result = tool_fn(project='.')
    print(f"Result with kwargs: {result}")
except Exception as e:
    print(f"Kwargs error: {e}")

# Check the original tool function
print("\nChecking original tool function...")
from accel.mcp_server import create_server
import accel.mcp_server as ms

# Get the actual function
print(f"accel_verify_start type: {type(ms.accel_verify_start)}")
print(f"accel_verify_start signature: {inspect.signature(ms.accel_verify_start)}")
