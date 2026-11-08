"""ビューアを localhost で開く：python serve.py（ポートは引数で変えられる）。"""

import functools
import http.server
import pathlib
import sys
import threading
import webbrowser

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
root = pathlib.Path(__file__).resolve().parent
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
url = f"http://localhost:{port}/viewer/"
print(f"{url} で開いています（止めるときは Ctrl+C）")
threading.Timer(0.5, webbrowser.open, [url]).start()
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
