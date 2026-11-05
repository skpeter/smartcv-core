import asyncio
import socket
import sys
import time

BROADCAST_PORT = 6500
BROADCAST_INTERVAL = 1  # seconds


def _log(*args, **kwargs):
    # core.py is often __main__; avoid importing it and creating a cycle.
    for name in ("__main__", "core", "core.core"):
        mod = sys.modules.get(name)
        fn = getattr(mod, "print_with_time", None) if mod else None
        if fn:
            fn(*args, **kwargs)
            return
    print(*args, **kwargs)


def broadcast_device_info(client_name: str):
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    message = f"{client_name}:{hostname}:{ip}".encode("utf-8")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind(('', 0))

    try:
        while True:
            sock.sendto(message, ("192.168.1.255", BROADCAST_PORT))
            time.sleep(BROADCAST_INTERVAL)
    except asyncio.CancelledError:
        _log("Broadcast cancelled.")
    finally:
        sock.close()


def broadcaster_thread():
    try:
        asyncio.run(broadcast_device_info())
    except Exception as e:
        _log(f"Broadcast thread error: {e}")


if __name__ == "__main__":
    asyncio.run(broadcast_device_info())
