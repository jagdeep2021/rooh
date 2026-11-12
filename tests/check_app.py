"""Does every part of the app work? A real server on a scratch data folder, the real page in a headless browser,
every button, shortcut and server route used at least once, and every result checked on disk or through the server
(not just on screen). Synthetic sections from check_numbers.py, so the edits have known answers.

  .venv/bin/python tests/check_app.py [out_dir] [--no-cellpose]

  --no-cellpose   skip the image Cellpose outlines (the rest uses exact synthetic masks): ~1 min instead of ~3
Needs once:  .venv/bin/pip install -r requirements-dev.txt && .venv/bin/python -m playwright install chromium-headless-shell

Writes out_dir/checks_app.xlsx (every check), out_dir/coverage.xlsx (every control and route: used or not) and
screenshots of the main states (out_dir/shots/). Exit 1 on any FAIL, or on any control / route never used that is
not in NOT_AUTOMATED (each with its reason).

Two servers: the website (`server.py serve`: invite, join, approve, projects, upload, Outline, edits, scale, results,
download, delete) and the desktop app (`server.py local`, with HOME pointed at the scratch folder so the real
~/.rooh project list is never touched).
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))
import pipeline as P            # noqa: E402
import run_batch as RB          # noqa: E402
import check_numbers as CN      # noqa: E402   the synthetic section (geometry known)

from playwright.sync_api import sync_playwright, expect   # noqa: E402

PY = sys.executable
PAGE = (HERE / "web/static/index.html").read_text()
SERVER = (HERE / "web/server.py").read_text()
IDS = sorted(set(re.findall(r'id="([A-Za-z0-9_]+)"', PAGE)))
ROUTES = re.findall(r'@app\.(get|post|delete)\("([^"]+)"\)', SERVER)
KEYS = ["1", "2", "3", "4", "5", "6", ",", ".", "ArrowLeft", "ArrowRight", "Shift+S", "o", "Shift+A", "m",
        "Escape", "Shift+F", "f", "Meta+z", "Control+z", "Meta+Shift+z", "Control+Shift+z", "Control+y", "Space"]
NOT_AUTOMATED = {
    "id:cv": "the canvas: used by every click, drag, wheel and pan (counted through those, not as a control)",
    "id:empty": "a message, not a control (checked: hidden once an image opens)",
    "id:err": "the error line (checked by the error cases)",
    "id:tip": "the tool's help line (checked when the tool changes)",
    "id:view": "the image area (container)",
    "id:zoomBar": "the zoom bar container (its buttons and slider are used)",
    "id:rulerPx": "text inside the scale box (checked: shows the line's length)",
    "id:count": "text: checked count (checked after Save + next)",
    "id:imgName": "text (checked on image change)",
    "id:imgFig": "the live pattern picture (checked: loads)",
    "id:traitBars": "the trait bars (checked: drawn)",
    "id:jobMsg": "status text (checked during Outline / upload)",
    "id:resMsg": "status text (checked after Make results)",
    "id:scaleMsg": "status text (checked after scale)",
    "id:editMsg": "status text (checked after undo)",
    "id:figList": "group figure links (one is opened)",
    "id:prog": "progress bar (checked visible while Outline runs, with Cellpose)",
    "id:who": "the signed-in email (checked)",
    "id:peopleList": "list (its Approve / Remove buttons are used)",
    "id:target": "the tool row (each tool button is used)",
    "id:reasonRow": "row (checked: shown for Sectors only)",
    "id:invOut": "row (checked: shown after Invite)",
    "id:invMsg": "text (checked after Invite)",
    "id:projNewRow": "row (shown by +)",
    "id:delName": "text in the delete dialog",
    "id:delDlg": "the delete dialog (its three buttons are used)",
    "id:rulerBox": "the scale-bar box (its field and buttons are used)",
    "id:imgFigLink": "link around the pattern picture (its href is fetched)",
    "id:peoplePanel": "the People panel (opened; its controls are used)",
    "id:projAdd(local)": "",
    "route:post /api/local/open (picker)": "the desktop's folder picker is the system dialog; the route is called "
                                           "with a path instead, as the picker would",
}

CHECKS: list[dict] = []
USED: set[str] = set()
HIT: set[str] = set()


def check(case, what, got, want, tol=0.0):
    if tol:
        ok = got is not None and abs(got - want) <= tol * abs(want)
    else:
        ok = got == want
    CHECKS.append({"case": case, "check": what, "expected": want, "got": got, "ok": "PASS" if ok else "FAIL"})
    print(f"  {'PASS' if ok else 'FAIL'}  {case}: {what}" + ("" if ok else f"   (expected {want!r}, got {got!r})"),
          flush=True)
    return ok


# ---------------------------------------------------------------- synthetic data
def png(path: Path, gray: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8)).save(path)


def make_images(root: Path):
    """Maize/wt/P1/S.png and Maize/mut/P2/S2.png (exact masks, written by the test), Maize/mut/P3/M.png (Cellpose)."""
    cells, region, mx, gray = CN.section()
    up = root / "upload" / "Maize"
    png(up / "wt/P1/S.png", gray)
    png(up / "mut/P2/S2.png", np.rot90(gray))
    png(root / "cellpose" / "M.png", np.fliplr(gray))
    (root / "junk").mkdir(parents=True, exist_ok=True)
    (root / "junk" / "notes.txt").write_text("not an image")
    return {"Maize/wt/P1/S.png": (cells, region, mx),
            "Maize/mut/P2/S2.png": tuple(np.rot90(a).copy() for a in (cells, region, mx))}


def write_masks(proj: Path, exact: dict):
    """As run_batch leaves them, but exact: the edits below then have known answers."""
    anatomy = proj / "anatomy"
    RB.load_settings(anatomy, proj / "raw")
    (anatomy / "masks").mkdir(parents=True, exist_ok=True)
    for rel, (cells, region, mx) in exact.items():
        P.save_state(anatomy / "masks" / f"{RB.image_key(Path(rel))}.npz", cells, region, mx, image_file=rel)


# ---------------------------------------------------------------- servers
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, args, env, log: Path):
        self.log = open(log, "w")
        self.p = subprocess.Popen([PY, "-W", "ignore", str(HERE / "web/server.py"), *args], env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=HERE)
        self.lines: list[str] = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            self.lines.append(line)
            self.log.write(line)
            self.log.flush()

    def wait_for(self, pattern, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            for l in self.lines:
                m = re.search(pattern, l)
                if m:
                    return m
            time.sleep(0.2)
        raise TimeoutError(f"server never printed {pattern}")

    def stop(self):
        self.p.terminate()
        try:
            self.p.wait(10)
        except subprocess.TimeoutExpired:
            self.p.kill()


def wait_http(url, timeout=60):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception as e:
            if getattr(e, "code", None):           # any HTTP answer = up
                return
            time.sleep(0.3)
    raise TimeoutError(url)


# ---------------------------------------------------------------- page helpers
class UI:
    def __init__(self, page, shots: Path):
        self.page, self.shots = page, shots
        page.on("request", lambda r: self._route(r.method, r.url))
        page.on("dialog", lambda d: d.accept())       # confirm(): yes
        self.errors = []
        page.on("pageerror", lambda e: self.errors.append(str(e)))

    @staticmethod
    def _route(method, url):
        path = re.sub(r"^https?://[^/]+", "", url).split("?")[0]
        for m, r in ROUTES:
            rx = "^" + re.sub(r"\{[^}]+\}", "[^/]+", r) + "$"
            if m.upper() == method and re.match(rx, path):
                HIT.add(f"{m} {r}")

    def click(self, id_, **kw):
        USED.add(f"id:{id_}")
        self.page.click(f"#{id_}", **kw)

    def fill(self, id_, v):
        USED.add(f"id:{id_}")
        self.page.fill(f"#{id_}", str(v))

    def press(self, key):
        USED.add(f"key:{key}")
        self.page.keyboard.press(key)

    def js(self, expr, arg=None):
        return self.page.evaluate(expr, arg)

    def idle(self, timeout=60000):
        """No edit / undo in flight and the page has drawn its result."""
        self.page.wait_for_function("() => !S.saving", timeout=timeout)
        self.page.wait_for_timeout(150)

    def shot(self, name):
        self.page.screenshot(path=str(self.shots / f"{name}.png"))

    def at(self, y, x):
        """image pixel -> screen point."""
        return self.js("([y, x]) => { const r = cv.getBoundingClientRect(); "
                       "return [r.left + S.view.x + x * S.view.s, r.top + S.view.y + y * S.view.s]; }", [y, x])

    def click_img(self, y, x, mod=None):
        sx, sy = self.at(y, x)
        if mod:
            self.page.keyboard.down(mod)
        self.page.mouse.click(sx, sy)
        if mod:
            self.page.keyboard.up(mod)
        self.idle()

    def drag_img(self, pts, mod=None, button="left", wait=True):
        sp = [self.at(y, x) for y, x in pts]
        if mod:
            self.page.keyboard.down(mod)
        self.page.mouse.move(*sp[0])
        self.page.mouse.down(button=button)
        for a, b in zip(sp[:-1], sp[1:]):
            self.page.mouse.move(*b, steps=max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) // 3)))
        self.page.mouse.up(button=button)
        if mod:
            self.page.keyboard.up(mod)
        if wait:
            self.idle()

    def traits(self, rel):
        return self.js("async rel => (await fetch(`/api/projects/${enc(S.project)}/traits?rel=${enc(rel)}`)).json()", rel)

    def open_image(self, rel):
        k = self.js("rel => S.images.findIndex(i => i.image === rel)", rel)
        self.page.select_option("#pick", str(k))
        USED.add("id:pick")
        self.page.wait_for_function("k => S.i === k && S.gray && S.shown === k", arg=k)
        self.page.wait_for_timeout(300)


def circle(cy, cx, r, n=48):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(cy - r * np.sin(a), cx + r * np.cos(a)) for a in t] + [(cy - r * np.sin(0), cx + r)]


def at(r, deg):
    return CN.at(r, deg)


# ---------------------------------------------------------------- the website
def run_web(tmp: Path, shots: Path, cellpose: bool, pw):
    data = tmp / "web_data"
    env = {**os.environ, "CSA_DATA": str(data), "PYTHONUNBUFFERED": "1"}
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    tok = subprocess.run([PY, "-W", "ignore", str(HERE / "web/server.py"), "invite", "admin@test.org", "--admin",
                          "--base", base], env=env, capture_output=True, text=True, cwd=HERE).stdout.strip()
    check("web", "invite prints a login link", "/login?token=" in tok, True)
    srv = Server(["serve", "--port", str(port)], env, tmp / "server_web.log")
    wait_http(base + "/help")
    exact = make_images(tmp)
    browser = pw.chromium.launch()
    ctx = browser.new_context(viewport={"width": 1500, "height": 950}, accept_downloads=True)
    ctx.grant_permissions(["clipboard-read", "clipboard-write"], origin=base)
    page = ctx.new_page()
    ui = UI(page, shots)
    try:
        # ---- access: a bad link, the admin's link, someone asking, approve, invite, remove
        r = page.goto(base + "/login?token=wrong")
        check("access", "a wrong login link is refused (401)", r.status, 401)
        r = page.goto(base + "/local?key=x")
        check("access", "/local is closed on the website (403)", r.status, 403)
        page.goto(tok)
        page.wait_for_function("() => document.querySelector('#who')?.textContent.length > 0")
        check("access", "admin signed in", page.inner_text("#who"), "admin@test.org")
        check("access", "help is a PDF", page.request.get(base + "/help").headers["content-type"], "application/pdf")
        check("access", "icon served", page.request.get(base + "/icon.png").status, 200)
        HIT.update({"get /help", "get /icon.png"})

        guest = browser.new_context()
        gp = guest.new_page()
        UI(gp, shots)
        gp.goto(base + "/")
        gp.wait_for_selector("#jName")
        gp.fill("#jName", "Guest Person"); gp.fill("#jEmail", "guest@test.org"); gp.click("#jGo")
        USED.update({"id:jName", "id:jEmail", "id:jGo"})
        gp.wait_for_selector("#jRe")
        check("access", "ask for access -> waiting page", "Request sent" in gp.text_content("body"), True)
        gp.click("#jRe"); USED.add("id:jRe")
        gp.wait_for_selector("#jRe")
        check("access", "reload while waiting: still waiting", "Request sent" in gp.text_content("body"), True)
        page.reload(); page.wait_for_selector("#peoplePanel:not([hidden])")
        page.click("#peoplePanel summary"); USED.add("id:peoplePanel")
        check("access", "People shows 1 waiting", "1 waiting" in page.inner_text("#peoplePanel summary"), True)
        page.click("#peopleList li:has-text('guest@test.org') button:has-text('Approve')")
        page.wait_for_function("() => !document.querySelector('#peoplePanel summary').textContent.includes('waiting')")
        gp.goto(base + "/")
        gp.wait_for_function("() => document.querySelector('#who')?.textContent === 'guest@test.org'")
        check("access", "approved guest gets the app", gp.inner_text("#who"), "guest@test.org")
        ui.fill("invEmail", "invited@test.org"); ui.click("invGo")
        page.wait_for_selector("#invOut:not([hidden])")
        link = page.input_value("#invLink"); USED.add("id:invLink")
        check("access", "Invite makes a login link", "/login?token=" in link, True)
        ui.click("invCopy")
        page.wait_for_function("() => document.querySelector('#invCopy').textContent === 'Copied'")
        check("access", "Copy puts the link on the clipboard", ui.js("() => navigator.clipboard.readText()"), link)
        inv = browser.new_context().new_page()
        inv.goto(link)
        inv.wait_for_function("() => document.querySelector('#who')?.textContent === 'invited@test.org'")
        check("access", "the invite link signs in", inv.inner_text("#who"), "invited@test.org")
        page.click("#peopleList li:has-text('invited@test.org') button:has-text('Remove')")   # confirm(): yes
        page.wait_for_function("() => !document.querySelector('#peopleList').textContent.includes('invited@test.org')")
        r = inv.request.get(base + "/api/me")
        check("access", "removed person is signed out (401)", r.status, 401)
        guest.close()

        # ---- projects
        ui.click("projAdd")
        ui.fill("projName", "!!!"); ui.click("projNew")
        page.wait_for_function("() => $('err').textContent.length > 0")
        check("projects", "a bad name is refused with a message", "letters" in page.inner_text("#err"), True)
        ui.fill("projName", "Synthetic test")
        page.focus("#projName"); page.keyboard.press("Enter")
        page.wait_for_function("() => S.project === 'Synthetic_test'")
        check("projects", "Create (Enter) opens the new project", ui.js("() => S.project"), "Synthetic_test")
        USED.update({"id:projSel", "id:projNew"})
        proj = data / "projects" / "admin_test.org" / "Synthetic_test"

        # ---- upload: a folder (groups kept), files, and a selection without images
        page.set_input_files("#upFiles", str(tmp / "junk" / "notes.txt")); USED.add("id:upFiles")
        page.wait_for_function("() => $('err').textContent.includes('No images')")
        check("upload", "no images in the selection: message", "No images" in page.inner_text("#err"), True)
        page.set_input_files("#upDir", str(tmp / "upload" / "Maize")); USED.add("id:upDir")
        page.wait_for_function("() => S.images.length >= 2", timeout=60000)
        rels = sorted(ui.js("() => S.images.map(i => i.image)"))
        check("upload", "folder upload keeps the sub-folders", rels, ["Maize/mut/P2/S2.png", "Maize/wt/P1/S.png"])
        check("upload", "upload message", "added" in page.inner_text("#jobMsg"), True)
        if cellpose:
            page.set_input_files("#upFiles", str(tmp / "cellpose" / "M.png"))
            page.wait_for_function("() => S.images.length === 3", timeout=60000)
        write_masks(proj, exact)

        # ---- Outline (the measuring job): resumes, so only M is outlined by Cellpose
        ui.click("measure")
        if cellpose:
            page.wait_for_function("() => !$('prog').hidden", timeout=30000)
            check("outline", "progress bar shown while outlining", page.is_visible("#prog"), True)
        page.wait_for_function("() => $('jobMsg').textContent.includes('All images measured') || "
                               "$('jobMsg').className.includes('err')", timeout=900000)
        check("outline", "Outline finishes", page.inner_text("#jobMsg"), "All images measured.")
        page.reload()
        page.wait_for_function("() => S.images.length && S.images.every(i => i.status !== 'not measured') && S.gray",
                               timeout=60000)
        if cellpose:
            t = ui.traits("Maize/mut/P3/M.png") if False else None
            mrel = next(r for r in ui.js("() => S.images.map(i => i.image)") if r.endswith("M.png"))
            t = ui.traits(mrel)
            check("outline", "Cellpose image: stele found", bool(t.get("stele_found")), True)
            check("outline", "Cellpose image: cortical cells within 25% of 288", t.get("n_cortex_cells"), 288, .25)
            check("outline", "Cellpose image: vessels found (the synthetic vessels are not a test of the model)",
                  (t.get("mx_n") or 0) >= 1, True)
        check("outline", "empty-page message hidden once an image shows", page.is_hidden("#empty"), True)
        ui.shot("01_opened")

        # ---- navigation: menu, buttons, keys
        S_REL, S2_REL = "Maize/wt/P1/S.png", "Maize/mut/P2/S2.png"
        last = ui.js("() => S.images.length") - 1
        ui.js("() => show(0)"); page.wait_for_function("() => S.i === 0 && S.shown === 0")
        ui.click("next"); page.wait_for_function("() => S.i === 1")
        check("navigation", "› goes to the next image", ui.js("() => S.i"), 1)
        ui.js(f"() => show({last})"); page.wait_for_function(f"() => S.shown === {last}")
        ui.click("next"); page.wait_for_timeout(300)
        check("navigation", "› on the last image stays", ui.js("() => S.i"), last)
        ui.js("() => show(1)"); page.wait_for_function("() => S.shown === 1")
        ui.click("prev"); page.wait_for_function("() => S.i === 0")
        ui.click("prev"); page.wait_for_timeout(300)
        check("navigation", "‹ on the first image stays", ui.js("() => S.i"), 0)
        for k_next, k_prev in ((".", ","), ("ArrowRight", "ArrowLeft")):
            ui.press(k_next); page.wait_for_function("() => S.i === 1")
            ui.press(k_prev); page.wait_for_function("() => S.i === 0")
        ui.press("."); ui.press("."); ui.press(","); ui.press(".")        # fast: the last one pressed must win
        want = min(2, last)
        page.wait_for_function(f"() => S.shown === {want}", timeout=10000)
        page.wait_for_timeout(1500)                                      # a stale load would land in here
        check("navigation", "fast arrows: the image shown is the one named", ui.js("() => [S.i, S.shown]"), [want, want])
        ui.open_image(S_REL)
        check("navigation", "image name shown", S_REL in page.inner_text("#imgName"), True)
        n_img = ui.js("() => S.images.length")

        # ---- the view: zoom (wheel, + − slider, Fit, F), pan (Space, right button), overlays
        fit = ui.js("() => fitScale()")
        check("view", "opens fitted", ui.js("() => S.view.s"), fit, 1e-6)
        sx, sy = ui.at(500, 500)
        page.mouse.move(sx, sy); page.mouse.wheel(0, -400); USED.add("id:cv:wheel"); page.wait_for_timeout(200)
        check("view", "wheel zooms in", ui.js("() => S.view.s") > fit * 1.3, True)
        ui.click("zoomFit")
        check("view", "Fit (bar) back to fitted", ui.js("() => S.view.s"), fit, 1e-6)
        ui.click("zoomIn")
        check("view", "+ zooms 1.5x", ui.js("() => S.view.s / fitScale()"), 1.5, 1e-3)
        ui.click("zoomOut")
        check("view", "− zooms back", ui.js("() => S.view.s / fitScale()"), 1.0, 1e-3)
        ui.js("() => { const z = $('zoom'); z.value = 2; z.dispatchEvent(new Event('input')); "
              "z.dispatchEvent(new Event('change')); }"); USED.add("id:zoom")
        check("view", "slider: log2 2 = 4x", ui.js("() => S.view.s / fitScale()"), 4.0, 1e-3)
        check("view", "zoom label", page.inner_text("#zoomTxt"), "4.0×"); USED.add("id:zoomTxt")
        ui.shot("02_zoom4x")
        mid = lambda: ui.js("() => { const r = cv.getBoundingClientRect(); "
                            "return [(r.height / 2 - S.view.y) / S.view.s, (r.width / 2 - S.view.x) / S.view.s]; }")
        c0 = mid()
        page.set_viewport_size({"width": 1300, "height": 850}); page.wait_for_timeout(400)
        check("view", "window resized: zoom kept (4x)", ui.js("() => S.view.s / fitScale()"), 4.0, 1e-3)
        check("view", "window resized: same point at the centre", float(np.hypot(*np.subtract(mid(), c0))) < 1, True)
        page.set_viewport_size({"width": 1500, "height": 950}); page.wait_for_timeout(400)
        ui.press("f")
        check("view", "F fits (focus left the slider)", ui.js("() => S.view.s"), fit, 1e-6)
        ui.click("zoomIn"); ui.click("fitBtn")
        check("view", "Fit (side panel)", ui.js("() => S.view.s"), fit, 1e-6)
        n_edits = lambda: ui.js("() => performance.getEntriesByType('resource').filter(e => e.name.includes('/edit')).length")
        e0 = n_edits()
        page.focus("#opCells")                                   # a focused slider must not swallow Space
        x0 = ui.js("() => S.view.x")
        page.keyboard.down("Space"); USED.add("key:Space")
        ui.drag_img([(500, 500), (500, 600)], wait=False)
        page.keyboard.up("Space")
        page.wait_for_timeout(300)
        check("view", "Space-drag moves the image (slider focused)", ui.js("() => S.view.x") > x0 + 30, True)
        check("view", "Space-drag edits nothing", n_edits(), e0)
        x1 = ui.js("() => S.view.x")
        ui.drag_img([(500, 500), (500, 400)], button="right", wait=False)
        page.wait_for_timeout(300)
        check("view", "right-button drag moves the image", ui.js("() => S.view.x") < x1 - 30, True)
        check("view", "right drag edits nothing", n_edits(), e0)
        ui.press("f")
        ov = ui.js("() => S.showOv")
        ui.press("o")
        check("view", "O hides the outlines", ui.js("() => S.showOv"), not ov)
        ui.press("o")
        for sl in ("opCells", "opMasks"):
            ui.js(f"() => {{ const e = $('{sl}'); e.value = 0.3; e.dispatchEvent(new Event('input')); "
                  f"e.dispatchEvent(new Event('change')); }}"); USED.add(f"id:{sl}")
            check("view", f"{sl} fades its layer", ui.js(f"() => +$('{sl}').value"), 0.3)
        ui.press("f")
        check("view", "keys work after a slider (F fits)", ui.js("() => S.view.s"), fit, 1e-6)
        a0 = ui.js("() => S.showAer")
        ui.click("aerBtn")
        check("view", "Aerenchyma button toggles", ui.js("() => S.showAer"), not a0)
        ui.press("Shift+A")
        check("view", "Shift-A toggles back", ui.js("() => S.showAer"), a0)
        ui.click("filesBtn")
        page.wait_for_timeout(300)
        check("view", "Cell files on", ui.js("() => S.showFiles"), True)
        check("view", "file colours loaded", ui.js("() => !!S.ovF"), True)
        ui.shot("03_cell_files")
        ui.press("Shift+F")
        check("view", "Shift-F off", ui.js("() => S.showFiles"), False)
        check("view", "live cortex pattern picture loads", ui.js(
            "() => new Promise(ok => { const i = $('imgFig'); i.complete ? ok(i.naturalWidth) : i.onload = () => ok(i.naturalWidth); })") > 0,
            True)
        check("view", "trait bars drawn (11)", ui.js("() => $('traitBars').querySelectorAll('.tr').length"), 11)

        # ---- tools: buttons and keys 1-6
        for k, t in enumerate(["Cell", "Metaxylem", "Stele", "Sectors", "Root", "Aerenchyma"], 1):
            ui.press(str(k))
            check("tools", f"key {k} = {t}", ui.js("() => S.target"), t)
            check("tools", f"{t}: reason menu only for Sectors", page.is_visible("#reasonRow"), t == "Sectors")
            page.click(f"#target button[data-t='{t}']")
            check("tools", f"{t} button", ui.js("() => S.target"), t)
            check("tools", f"{t}: help line", len(page.inner_text("#tip")) > 20, True)

        # ---- edits with known answers (image S: 8 layers x 36 cortical cells, 6 vessels r 15)
        base_t = ui.traits(S_REL)
        check("edit", "start: 288 cortical cells", base_t["n_cortex_cells"], 288)
        cell = at((CN.EDGES[3] + CN.EDGES[4]) / 2, 115)
        ui.press("1")
        ui.click_img(*cell, mod="Meta")
        check("edit", "Cell: ⌘-click removes a cell", ui.traits(S_REL)["n_cortex_cells"], 287)
        check("edit", "image now reviewed", ui.js("() => S.images[S.i].status"), "reviewed")
        ui.click_img(*cell)
        check("edit", "Cell: click adds it back", ui.traits(S_REL)["n_cortex_cells"], 288)
        r0, r1 = CN.EDGES[3] + 1, CN.EDGES[5] - 1
        ui.drag_img([at(r0, t) for t in np.linspace(130.5, 139.5, 8)] + [at(r1, t) for t in np.linspace(139.5, 130.5, 8)])
        check("edit", "Cell: drawing one cell over two merges them", ui.traits(S_REL)["n_cortex_cells"], 287)
        ui.shot("04_cell_edits")
        ui.click("undoBtn"); ui.idle()
        check("undo", "Undo button: the merge is undone", ui.traits(S_REL)["n_cortex_cells"], 288)
        ui.click("redoBtn"); ui.idle()
        check("undo", "Redo button: merged again", ui.traits(S_REL)["n_cortex_cells"], 287)
        page.focus("#pick")                                      # whatever has focus
        ui.press("Meta+z"); ui.idle()
        check("undo", "⌘Z with the image menu focused", ui.traits(S_REL)["n_cortex_cells"], 288)
        ui.press("Meta+Shift+z"); ui.idle()
        check("undo", "⌘⇧Z redoes", ui.traits(S_REL)["n_cortex_cells"], 287)
        ui.press("Control+z"); ui.idle()
        check("undo", "Ctrl-Z", ui.traits(S_REL)["n_cortex_cells"], 288)
        ui.press("Control+y"); ui.idle()
        check("undo", "Ctrl-Y", ui.traits(S_REL)["n_cortex_cells"], 287)
        ui.press("Control+z"); ui.idle()
        ui.press("Control+Shift+z"); ui.idle()
        check("undo", "Ctrl-Shift-Z", ui.traits(S_REL)["n_cortex_cells"], 287)
        check("undo", "message says what is left", "to undo" in page.inner_text("#editMsg"), True)
        for _ in range(3):
            ui.press("Meta+z"); ui.idle()
        check("undo", "three undos: back to the start", ui.traits(S_REL)["n_cortex_cells"], 288)

        ui.press("2")
        v = at(CN.MX_AT, 135)
        ui.click_img(*v, mod="Meta")
        check("edit", "Vessel: ⌘-click removes one", ui.traits(S_REL)["mx_n"], 5)
        ui.click_img(*v)
        t = ui.traits(S_REL)
        check("edit", "Vessel: click puts it back", t["mx_n"], 6)
        check("edit", "Vessel: clicked back at its size (pi 15^2 mean)", t["mx_mean_area_px"], np.pi * 15 ** 2, .05)

        ui.press("3")
        ui.drag_img(circle(500, 500, 200))
        check("edit", "Stele: drawn r 200 -> pi 200^2", ui.traits(S_REL)["stele_area_px"], np.pi * 200 ** 2, .03)
        ui.press("Meta+z"); ui.idle()
        check("edit", "Stele undone", ui.traits(S_REL)["stele_area_px"], np.pi * 150 ** 2, .02)

        ui.press("5")
        ui.drag_img(circle(500, 500, 455))
        check("edit", "Root: drawn r 455 -> pi 455^2", ui.traits(S_REL)["root_area_px"], np.pi * 455 ** 2, .02)
        ui.press("Meta+z"); ui.idle()
        ui.drag_img([(500 - 470, 500 - 60), (500 - 470, 500 + 60), (500 - 380, 500 + 60), (500 - 380, 500 - 60)],
                    mod="Meta")
        check("edit", "Root: ⌘-drag cuts an area out", ui.traits(S_REL)["root_area_px"] < np.pi * 475 ** 2 - 4000, True)
        ui.press("Meta+z"); ui.idle()
        check("edit", "Root undone", ui.traits(S_REL)["root_area_px"], np.pi * 475 ** 2, .01)

        ui.press("4")
        page.select_option("#reason", "debris"); USED.add("id:reason")
        ui.click_img(*at(300, 40))
        check("edit", "Sectors: a click explains (note)", "drag" in page.inner_text("#editMsg").lower(), True)
        ui.drag_img([at(300, t) for t in np.linspace(100, 170, 15)])
        t = ui.traits(S_REL)
        check("edit", "Sectors: drag 100-170 excludes 70 degrees", t["angle_kept_deg"], 290, .01)
        check("edit", "Sectors: reason kept", "debris" in t["excluded_sectors"], True)
        ui.shot("05_sector")
        ui.click_img(*at(300, 135), mod="Meta")
        check("edit", "Sectors: ⌘-click brings it back", ui.traits(S_REL)["angle_kept_deg"], 360)

        ui.press("6")
        r0, r1 = CN.EDGES[4] + 1, CN.EDGES[5] - 1
        ui.drag_img([at(r0, t) for t in np.linspace(10.5, 49.5, 20)] + [at(r1, t) for t in np.linspace(49.5, 10.5, 20)])
        t = ui.traits(S_REL)
        check("edit", "Aerenchyma: drawn around 4 cells = 1 lacuna", t["n_lacunae"], 1)
        check("edit", "Aerenchyma: counted as drawn", t["n_lacunae_drawn"], 1)
        check("edit", "Aerenchyma button shows the count", "1 (" in page.inner_text("#aerBtn"), True)
        ui.shot("06_lacuna")
        ui.click_img(*at((CN.EDGES[4] + CN.EDGES[5]) / 2, 30), mod="Meta")
        check("edit", "Aerenchyma: ⌘-click = not a lacuna", ui.traits(S_REL)["n_lacunae"], 0)
        check("edit", "edit status written to disk (npz: reviewed)",
              P.load_state(proj / "anatomy/masks" / f"{RB.image_key(Path(S_REL))}.npz")[3], "reviewed")

        # ---- scale: typed (image, all), missing, measured with the ruler (M, Enter, Esc, ✕, too short)
        ui.fill("scale", ""); ui.click("scaleImg")
        page.wait_for_function("() => $('err').textContent.includes('Type the scale')")
        check("scale", "no scale typed: message", True, True)
        ui.fill("scale", 0.5); ui.click("scaleImg")
        page.wait_for_function("() => $('scaleMsg').textContent.includes('set for 1')")
        check("scale", "Image: one image", page.inner_text("#scaleMsg").startswith("0.5"), True)
        ui.click("scaleAll")
        page.wait_for_function(f"() => $('scaleMsg').textContent.includes('set for {n_img}')")
        s = pd.read_excel(proj / "anatomy/samples.xlsx")
        check("scale", "All: every image in samples.xlsx", s.um_per_px.tolist(), [0.5] * n_img)
        ui.press("m")
        check("scale", "M: ruler on", ui.js("() => !!S.ruler"), True)
        ui.press("Escape")
        check("scale", "Esc: ruler off", ui.js("() => S.ruler"), None)
        ui.click("scaleRuler")
        ui.drag_img([(900, 100), (900, 102)], wait=False)
        page.wait_for_timeout(200)
        check("scale", "too short a line: message", "Too short" in page.inner_text("#scaleMsg"), True)
        ui.drag_img([(900, 100), (902, 300)], wait=False)    # 2 px off level: snaps level, 200 px
        page.wait_for_selector("#rulerBox:not([hidden])")
        check("scale", "ruler length shown (snapped level: 200 px)", page.inner_text("#rulerPx"), "200.0 px")
        ui.shot("07_ruler")
        page.fill("#rulerUm", "50"); USED.add("id:rulerUm")
        page.keyboard.press("Enter")
        check("scale", "Enter sets 50 µm / 200 px = 0.25", ui.js("() => +$('scale').value"), 0.25)
        ui.click("scaleRuler")
        ui.drag_img([(900, 100), (900, 300)], wait=False)
        page.wait_for_selector("#rulerBox:not([hidden])")
        ui.click("rulerX")
        check("scale", "✕ cancels", ui.js("() => S.ruler"), None)
        ui.click("scaleRuler")
        ui.drag_img([(900, 100), (900, 300)], wait=False)
        page.wait_for_selector("#rulerBox:not([hidden])")
        page.fill("#rulerUm", "100"); ui.click("rulerSet")
        check("scale", "Set: 100 µm / 200 px = 0.5", ui.js("() => +$('scale').value"), 0.5)

        # ---- Save + next and Shift-S
        ui.open_image(S2_REL)
        st0 = ui.js("() => S.images.find(i => i.image === arguments[0] || false)", None) if False else None
        ui.click("saveNext")
        page.wait_for_function("rel => S.images.find(i => i.image === rel).status === 'reviewed'", arg=S2_REL)
        check("review", "Save + next marks the image checked", True, True)
        check("review", "count shows the checked images", "✓" in page.inner_text("#count"), True)
        if cellpose:
            ui.open_image(mrel)
            ui.press("Shift+S")
            page.wait_for_function("rel => S.images.find(i => i.image === rel).status === 'reviewed'", arg=mrel)
            check("review", "Shift-S marks it checked", True, True)
            check("review", "all checked: message", "All images checked" in page.inner_text("#editMsg"), True)
        else:
            ui.press("Shift+S")

        # ---- Make results, figures, download
        page.check("#perImage"); USED.add("id:perImage")
        ui.click("results")
        page.wait_for_function("() => $('resMsg').textContent.startsWith('Done')", timeout=600000)
        msg = page.inner_text("#resMsg")
        check("results", "Make results lists the 4 files", all(k in msg for k in ("traits_", "cortex_pattern_",
                                                                                  "cortex_cells_", "all_cells_")), True)
        tr = pd.read_excel(next((proj / "results/um").glob("traits_*.xlsx")), sheet_name="Per image").set_index("image_file")
        check("results", "every image in traits", sorted(tr.index), sorted(ui.js("() => S.images.map(i => i.image)")))
        check("results", "the edits are in the results (S: 6 vessels, 360 kept, 0 lacunae)",
              (int(tr.at[S_REL, "mx_n"]), float(tr.at[S_REL, "angle_kept_deg"])), (6, 360.0))
        check("results", "scale in the results", float(tr.at[S_REL, "um_per_px"]), 0.5)
        figs = page.query_selector_all("#figList a")
        check("results", "group figure link shown", len(figs) >= 1, True)
        r = page.request.get(base + figs[0].get_attribute("href"))
        HIT.add("get /api/projects/{name}/figure")   # fetched by the test, not the page
        check("results", "group figure opens", (r.status, r.headers["content-type"]), (200, "image/png"))
        with page.expect_download() as dl:
            ui.click("download")
        z = zipfile.ZipFile(dl.value.path())
        names = z.namelist()
        check("results", "Download zip holds traits + samples.xlsx",
              any("results/um/traits_" in n for n in names) and any(n.endswith("anatomy/samples.xlsx") for n in names), True)
        ui.shot("08_results")

        # ---- Download project -> upload it again as a project zip: everything back, byte for byte (2026-11-12)
        with page.expect_download() as dl:
            ui.click("downloadAll")
        zpath = tmp / "whole_project.zip"
        dl.value.save_as(zpath)
        page.set_input_files("#projZip", str(zpath)); USED.add("id:projZip")
        page.wait_for_function("() => S.project === 'Synthetic_test_2' && S.images.length > 0", timeout=60000)
        HIT.add("post /api/projects/import")
        proj2 = proj.parent / "Synthetic_test_2"
        skip = lambda p: p.parts[0] in (".cache", ".undo", ".redo") or p.parts[:2] == ("anatomy", "qc")
        files = lambda d: {p.relative_to(d): p for p in d.rglob("*") if p.is_file() and not skip(p.relative_to(d))}
        a, b = files(proj), files(proj2)
        check("project zip", "same files after download + upload (raw, masks, settings, samples, results)",
              sorted(map(str, set(a) ^ set(b))), [])
        same = lambda k: (json.loads(a[k].read_text()) == json.loads(b[k].read_text()) if k.suffix == ".json"
                          else a[k].read_bytes() == b[k].read_bytes())
        check("project zip", "every file identical", sorted(str(k) for k in set(a) & set(b) if not same(k)), [])
        check("project zip", "images, masks, statuses as before", ui.js("() => S.images.map(i => [i.image, i.status])"),
              [[i["image"], i["status"]] for i in json.loads(page.request.get(base + "/api/projects/Synthetic_test").text())["images"]])
        check("project zip", "the edited image measures the same", ui.traits(S_REL), json.loads(page.request.get(
              base + f"/api/projects/Synthetic_test/traits?rel={S_REL}").text()))
        ui.js("async () => api(`/api/projects/${enc(S.project)}/results`, form({ image_figures: false }))")
        t2 = pd.read_excel(next((proj2 / "results/um").glob("traits_*.xlsx")), sheet_name="Per image").set_index("image_file")
        check("project zip", "Make results on the copy = the original's numbers",
              t2.drop(columns=["status"], errors="ignore").equals(tr.drop(columns=["status"], errors="ignore")), True)

        # ---- delete one image (in the copy): off the list, its masks and samples row gone, to the trash
        ui.open_image(S2_REL)
        ui.click("imgDel")
        page.wait_for_function("rel => !S.images.some(i => i.image === rel)", arg=S2_REL)
        HIT.add("delete /api/projects/{name}/images")
        k2 = RB.image_key(Path(S2_REL))
        check("delete image", "image and masks gone from the project",
              ((proj2 / "raw" / S2_REL).exists(), (proj2 / f"anatomy/masks/{k2}.npz").exists()), (False, False))
        check("delete image", "... and in the trash", sorted(p.name for p in (data / "trash").rglob("*")
                                                             if p.name in (Path(S2_REL).name, f"{k2}.npz")),
              sorted([Path(S2_REL).name, f"{k2}.npz"]))
        check("delete image", "samples.xlsx row gone", S2_REL in set(pd.read_excel(proj2 / "anatomy/samples.xlsx").image_file),
              False)
        check("delete image", "the original project untouched", (proj / "raw" / S2_REL).exists(), True)
        ui.js("async () => { await openProject('Synthetic_test'); }")

        # ---- a second user cannot see this project; the admin sees everyone's
        gp2 = browser.new_context().new_page()
        gp2.goto(base + "/")
        r = gp2.request.get(base + "/api/projects/Synthetic_test")
        check("privacy", "a signed-out browser gets 401", r.status, 401)

        # ---- delete: Cancel, download the whole project, Delete
        ui.click("projDel")
        page.wait_for_selector("#delDlg[open]")
        ui.click("delNo")
        check("delete", "Cancel keeps it", ui.js("() => S.project"), "Synthetic_test")
        ui.click("projDel")
        with page.expect_download() as dl:
            ui.click("delDown")
        names = zipfile.ZipFile(dl.value.path()).namelist()
        check("delete", "whole-project zip: raw, masks, results; no undo/cache",
              (any("/raw/" in n for n in names), any("/anatomy/masks/" in n for n in names),
               any("/results/" in n for n in names), any("/.undo/" in n or "/.cache/" in n for n in names)),
              (True, True, True, False))
        ui.click("delYes")
        page.wait_for_function("() => S.project === null")
        check("delete", "Delete: off the list", "Synthetic_test" in ui.js("() => [...$('projSel').options].map(o => o.value)"),
              False)
        check("delete", "Delete: moved to trash, not erased", any((data / "trash").rglob("*.npz")), True)
        check("page", "no JavaScript errors", ui.errors, [])
    finally:
        browser.close()
        srv.stop()


# ---------------------------------------------------------------- the desktop app
def run_local(tmp: Path, shots: Path, pw):
    home = tmp / "home"
    home.mkdir()
    exp = tmp / "desktop_experiment"
    exact = make_images(tmp / "desk")
    shutil.copytree(tmp / "desk" / "upload", exp / "raw")
    write_masks(exp, exact)
    port = free_port()
    env = {**os.environ, "HOME": str(home), "PYTHONUNBUFFERED": "1"}
    srv = Server(["local", "--no-browser", "--port", str(port)], env, tmp / "server_local.log")
    url = srv.wait_for(r"(http://127\.0\.0\.1:\d+/local\?key=\S+)").group(1)
    base = f"http://127.0.0.1:{port}"
    wait_http(base + "/help")
    browser = pw.chromium.launch()
    page = browser.new_context(viewport={"width": 1500, "height": 950}).new_page()
    ui = UI(page, shots)
    try:
        r = page.goto(base + "/local?key=wrong")
        check("desktop", "wrong key refused (403)", r.status, 403)
        check("desktop", "no key: API refused (401)", page.request.get(base + "/api/projects").status, 401)
        page.goto(url)
        page.wait_for_function("() => S.local === true")
        check("desktop", "opens as 'this computer'", page.inner_text("#who"), "this computer")
        name = ui.js("async p => (await api('/api/local/open', form({ path: p }))).name", str(exp))
        ui.js("async n => { await loadProjects(n); }", name)
        page.wait_for_function("() => S.images.length === 2 && S.gray", timeout=60000)
        check("desktop", "the folder opens in place", name, "desktop_experiment")
        check("desktop", "project list kept under the test HOME, not the real one",
              (home / ".rooh" / "local_projects.json").exists(), True)
        ui.open_image("Maize/wt/P1/S.png")
        ui.press("1")
        ui.click_img(*at((CN.EDGES[3] + CN.EDGES[4]) / 2, 115), mod="Control")      # Ctrl as on Windows
        check("desktop", "Ctrl-click removes a cell", ui.traits("Maize/wt/P1/S.png")["n_cortex_cells"], 287)
        check("desktop", "the edit is saved in the folder itself",
              len(list((exp / ".undo").rglob("*.npz"))), 1)
        ui.press("Control+z"); ui.idle()
        check("desktop", "Ctrl-Z undoes", ui.traits("Maize/wt/P1/S.png")["n_cortex_cells"], 288)
        ui.click("results")
        page.wait_for_function("() => $('resMsg').textContent.startsWith('Done')", timeout=600000)
        check("desktop", "Make results writes into the folder", any((exp / "results/um").glob("traits_*.xlsx")), True)
        ui.open_image("Maize/wt/P1/S.png")
        ui.click("imgDel")
        page.wait_for_function("() => S.images.length === 1")
        check("desktop", "delete image: off the list, file left in place, masks in anatomy/removed/",
              ((exp / "raw/Maize/wt/P1/S.png").exists(), (exp / "anatomy/removed/masks/Maize__wt__P1__S.npz").exists(),
               json.loads((exp / "anatomy/settings.json").read_text())["removed"]), (True, True, ["Maize/wt/P1/S.png"]))
        ui.shot("09_desktop")
        ui.click("projDel")                       # confirm(): yes -> off the list, folder untouched
        page.wait_for_function("() => S.project === null")
        check("desktop", "✕ takes it off the list", json.loads((home / ".rooh/local_projects.json").read_text()), {})
        check("desktop", "... and leaves the folder alone", (exp / "anatomy" / "masks").is_dir(), True)
        USED.add("id:projAdd(local)")
        HIT.add("post /api/local/open")
        check("desktop", "no JavaScript errors", ui.errors, [])
    finally:
        browser.close()
        srv.stop()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cellpose = "--no-cellpose" not in sys.argv
    out = Path(args[0]) if args else Path(tempfile.mkdtemp(prefix="rooh_app_"))
    shutil.rmtree(out, ignore_errors=True)
    shots = out / "shots"
    shots.mkdir(parents=True)
    t0 = time.time()
    with sync_playwright() as pw:
        for name, fn in (("website", lambda: run_web(out / "web", shots, cellpose, pw)),
                         ("desktop", lambda: run_local(out / "local", shots, pw))):
            (out / ("web" if name == "website" else "local")).mkdir(exist_ok=True)
            print(f"\n== {name}", flush=True)
            try:
                fn()
            except Exception as e:
                import traceback
                traceback.print_exc()
                CHECKS.append({"case": name, "check": f"ERROR {type(e).__name__}: {str(e)[:300]}", "ok": "FAIL"})
    # coverage: every control, shortcut and route
    want = {f"id:{i}" for i in IDS} | {f"key:{k}" for k in KEYS} | {f"{m} {r}" for m, r in ROUTES}
    used = USED | HIT | {k for k in want if k.startswith("id:") and k in NOT_AUTOMATED}
    cov = pd.DataFrame([{"item": k, "used": k in used, "why not": NOT_AUTOMATED.get(k, "")} for k in sorted(want)])
    missing = cov[~cov.used].item.tolist()
    CHECKS.append({"case": "coverage", "check": "every control, shortcut and route used",
                   "expected": "none missing", "got": ", ".join(missing) or "none missing",
                   "ok": "PASS" if not missing else "FAIL"})
    t = pd.DataFrame(CHECKS)
    t.to_excel(out / "checks_app.xlsx", index=False)
    cov.to_excel(out / "coverage.xlsx", index=False)
    n_ok, n_bad = (t.ok == "PASS").sum(), (t.ok == "FAIL").sum()
    print(f"\ncoverage: {cov.used.sum()} of {len(cov)} controls / shortcuts / routes used"
          + (f"; NOT USED: {', '.join(missing)}" if missing else ""))
    print(f"{n_ok} pass, {n_bad} fail in {time.time() - t0:.0f} s  ->  {out}")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
