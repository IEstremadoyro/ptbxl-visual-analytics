import socket, sys
desde = int(sys.argv[1]) if len(sys.argv) > 1 else 8010
for p in range(desde, desde + 200):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", p)); s.close(); print(p); break
    except OSError:
        s.close()
