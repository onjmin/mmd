"""ビューアを localhost で開く：python serve.py（ポートは引数で変えられる）。

顔の調整ページ http://localhost:8765/viewer/tune.html も、このサーバーで動く（作り直しと保存をここで受ける）。
  GET  /api/face?m=キー          調整できる項目と、保存してある値（その値で一度作り直してから返す）
  POST /api/face/build {m, values}  その値でキャラを作り直す（models/<キー>/ を上書き。保存はしない）
  POST /api/face/save  {m, values}  値を mmdgen/characters/face/<キー>.json に保存する
"""

import functools
import http.server
import json
import pathlib
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser

port = next((int(a) for a in sys.argv[1:] if a.isdigit()), 8765)
root = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from mmdgen import face  # noqa: E402
from build import TARGETS  # noqa: E402

LOCK = threading.Lock()


def rebuild(key, values):
    """values の調整値で key のキャラを作り直す。かかった秒数を返す。"""
    with LOCK:
        t0 = time.time()
        out = root / "models" / key
        out.mkdir(parents=True, exist_ok=True)
        face.use(values)
        model = TARGETS[key].build(str(out))
        model.save(str(out / f"{key}.pmx"))
        return time.time() - t0


class Handler(http.server.SimpleHTTPRequestHandler):
    def _json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path != "/api/face":
            return super().do_GET()
        key = urllib.parse.parse_qs(u.query).get("m", ["tabane_roze"])[0]
        if key not in TARGETS:
            return self._json(404, {"error": f"キャラ {key} はない"})
        try:
            saved = face.load(key)
            sec = rebuild(key, saved)
            values = {**face.DEFAULTS, "eye_color": face.EFFECTIVE.get("eye_color"), **saved}
            spec = [dict(name=n, label=l, min=a, max=b, step=s, default=d, help=h) for n, l, a, b, s, d, h in face.SPEC]
            self._json(200, {"key": key, "spec": spec, "values": values, "saved": saved, "seconds": sec,
                             "chars": list(TARGETS), "original_eye_color": face.EFFECTIVE.get("orig_eye_color")})
        except Exception:
            traceback.print_exc()
            self._json(500, {"error": traceback.format_exc()})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path not in ("/api/face/build", "/api/face/save"):
            return self._json(404, {"error": "ない"})
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            key, values = data.get("m"), data.get("values", {})
            if key not in TARGETS:
                return self._json(404, {"error": f"キャラ {key} はない"})
            if u.path.endswith("/save"):
                face.save(key, values)
                return self._json(200, {"saved": str(face.path(key).relative_to(root))})
            sec = rebuild(key, values)
            self._json(200, {"seconds": sec})
        except Exception:
            traceback.print_exc()
            self._json(500, {"error": traceback.format_exc()})


handler = functools.partial(Handler, directory=str(root))
server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
url = f"http://localhost:{port}/viewer/"
print(f"{url} で開いています。顔の調整は {url}tune.html（止めるときは Ctrl+C）")
if "--no-browser" not in sys.argv:
    threading.Timer(0.5, webbrowser.open, [url]).start()
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
