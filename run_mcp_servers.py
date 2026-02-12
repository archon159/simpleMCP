import os
import signal
import sys
import time
import multiprocessing as mp
import importlib
import pkgutil

SERVERS_PACKAGE = "mcp_servers"
PORTS = {
    "custom": 8001,
    "brave_search": 8002,
}

PARENT_PID = None
procs = []

def start_one(module_name: str, host: str, port: int):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    mod = importlib.import_module(f"{SERVERS_PACKAGE}.{module_name}")
    try:
        mod.run_server(host=host, port=port)
    except KeyboardInterrupt:
        pass

def discover_server_modules():
    pkg = importlib.import_module(SERVERS_PACKAGE)
    return [m.name for m in pkgutil.iter_modules(pkg.__path__) if not m.ispkg and not m.name.startswith("_")]

def shutdown(signum, frame):
    if os.getpid() != PARENT_PID:
        return

    print("\nShutting down all servers...")

    for p in procs:
        if p.is_alive():
            p.terminate()

    deadline = time.time() + 5
    for p in procs:
        p.join(timeout=max(0, deadline - time.time()))

    alive = [p for p in procs if p.is_alive()]
    if alive and hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except Exception:
            pass

    sys.exit(0)

def main():
    global PARENT_PID, procs
    PARENT_PID = os.getpid()

    host = "127.0.0.1"
    modules = discover_server_modules()

    if hasattr(os, "setpgrp"):
        os.setpgrp()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for i, name in enumerate(modules):
        port = PORTS.get(name, 8100 + i)
        p = mp.Process(target=start_one, args=(name, host, port))
        p.start()
        procs.append(p)
        print(f"Started {name} pid={p.pid} on {host}:{port}")

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()