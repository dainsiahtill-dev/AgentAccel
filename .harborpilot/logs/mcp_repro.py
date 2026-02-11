import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

log_path = Path('X:/Git/Harborpilot/agent-accel/.harborpilot/logs/mcp_repro_trace.log')
log_path.parent.mkdir(parents=True, exist_ok=True)

def log(msg: str) -> None:
    with log_path.open('a', encoding='utf-8') as f:
        f.write(msg + '\n')

log('start script')

def frame(payload):
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    return f'Content-Length: {len(body)}\r\n\r\n'.encode('ascii') + body

def start():
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    return subprocess.Popen(
        [sys.executable, '-m', 'accel.mcp_server'],
        cwd='X:/Git/Harborpilot/agent-accel',
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

def send(proc, payload):
    assert proc.stdin is not None
    proc.stdin.write(frame(payload))
    proc.stdin.flush()

def recv(proc):
    assert proc.stdout is not None
    headers = {}
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError('stdout closed')
        if line in (b'\r\n', b'\n'):
            break
        key, value = line.decode('utf-8').split(':', 1)
        headers[key.strip().lower()] = value.strip()
    body = proc.stdout.read(int(headers['content-length']))
    return json.loads(body.decode('utf-8'))

def call(proc, req_id, method, params=None):
    send(proc, {'jsonrpc': '2.0', 'id': req_id, 'method': method, 'params': params or {}})
    return recv(proc)

with tempfile.TemporaryDirectory() as td:
    project_dir = Path(td) / 'sample_project'
    (project_dir / 'src').mkdir(parents=True)
    (project_dir / 'src' / 'sample.py').write_text(
        'def add(a: int, b: int) -> int:\n    return a + b\n',
        encoding='utf-8',
    )
    cfg = {
        'verify': {'python': ['python -c "print(\'ok\')"'], 'node': []},
        'index': {'include': ['src/**'], 'exclude': [], 'max_file_mb': 2},
    }
    (project_dir / 'accel.yaml').write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    log(f'project={project_dir}')
    proc = start()
    try:
        log('before initialize')
        call(proc, 1, 'initialize', {
            'protocolVersion': '2025-06-18',
            'capabilities': {'resources': {}},
            'clientInfo': {'name': 'pytest', 'version': '1'},
        })
        log('after initialize')

        log('before index')
        call(proc, 20, 'tools/call', {
            'name': 'accel_index_build',
            'arguments': {'project': str(project_dir), 'full': True},
        })
        log('after index')

        log('before context')
        call(proc, 21, 'tools/call', {
            'name': 'accel_context',
            'arguments': {
                'project': str(project_dir),
                'task': 'Summarize sample add implementation',
                'changed_files': ['src/sample.py'],
            },
        })
        log('after context')

        log('before verify')
        call(proc, 22, 'tools/call', {
            'name': 'accel_verify',
            'arguments': {
                'project': str(project_dir),
                'changed_files': ['src/sample.py'],
                'evidence_run': True,
            },
        })
        log('after verify')
    finally:
        log('before shutdown')
        try:
            call(proc, 9998, 'shutdown', {})
            send(proc, {'jsonrpc': '2.0', 'method': 'exit', 'params': {}})
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception as exc:
            log(f'shutdown exception={exc!r}')
        try:
            _out, err = proc.communicate(timeout=5)
            log('after communicate')
            log('stderr=' + err.decode('utf-8', errors='replace')[-2000:])
        except subprocess.TimeoutExpired:
            log('communicate timeout; killing')
            proc.kill()
            _out, err = proc.communicate(timeout=5)
            log('killed; stderr=' + err.decode('utf-8', errors='replace')[-2000:])

log('script end')
