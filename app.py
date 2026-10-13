#!/usr/bin/env python3
# ============================================================================
# detecteur_web.py — interface web pour le détecteur Opsiom
#
# À placer dans le même dossier que Opsiom-Detector.py et detecteur_app.py :
#
#   pip install flask
#   python detecteur_web.py                      # http://localhost:5055
#   python detecteur_web.py --port 8080 --checkpoint opsiom_detector_best.pt
#
# Routes :
#   GET  /                interface (coller un texte ou déposer un .txt)
#   POST /api/analyse     {"text": "..."} -> verdict + phrases + texte normalisé
#   POST /detect          API brute du détecteur (pour Classroom, Omnia...)
#   GET  /detect/health   état du modèle
#
# ⚠️ Indice, jamais une preuve : ne sanctionne personne sur ce seul résultat.
# ============================================================================
import argparse

from flask import Flask, jsonify, request

from detecteur_app import charger_detecteur

PAGE = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Opsiom Détecteur</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#e8ecf3; --sheet:#ffffff; --ink:#14213d; --muted:#5b6785; --line:#d3d9e6;
  --accent:#3347c9; --accent-ink:#ffffff; --ia:#e8506a; --ok:#1f9d8a; --warn:#d98a14;
  --ui:"Figtree",system-ui,sans-serif; --read:"Newsreader",Georgia,serif;
}
@media (prefers-color-scheme:dark){
  :root{--bg:#0f1524; --sheet:#182036; --ink:#e9edf8; --muted:#97a2c0; --line:#2b3552;
        --accent:#7e8dff; --accent-ink:#0f1524; --ia:#ff7a90; --ok:#46c7b3; --warn:#f0a93b;}
}
*{box-sizing:border-box}
html{height:100%}
body{margin:0;min-height:100%;background:var(--bg);color:var(--ink);font:16px/1.5 var(--ui)}
header{max-width:1120px;margin:0 auto;padding:28px 20px 8px}
header h1{margin:0;font-size:1.5rem;font-weight:700;letter-spacing:-.01em}
header p{margin:4px 0 0;color:var(--muted)}
main{max-width:1120px;margin:0 auto;padding:16px 20px 48px;display:grid;
     grid-template-columns:minmax(0,1fr) 340px;gap:20px;align-items:start}
.sheet{background:var(--sheet);border:1px solid var(--line);border-radius:14px;min-height:420px;
       display:flex;flex-direction:column}
.sheet-bar{display:flex;gap:8px;align-items:center;padding:10px 12px;border-bottom:1px solid var(--line)}
.sheet-bar .count{margin-left:auto;color:var(--muted);font-size:.85rem}
button,.btn{font:inherit;font-weight:500;border-radius:9px;padding:8px 14px;cursor:pointer;
            border:1px solid var(--line);background:transparent;color:var(--ink)}
button:hover,.btn:hover{border-color:var(--accent)}
button.primary{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
button.primary:disabled{opacity:.5;cursor:default}
:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
textarea,.reading{flex:1;width:100%;min-height:340px;border:0;background:transparent;color:var(--ink);
  padding:22px 26px;font:1.12rem/1.75 var(--read);resize:vertical}
textarea:focus{outline:none}
textarea::placeholder{color:var(--muted)}
.reading{white-space:pre-wrap;overflow-wrap:anywhere}
.seg{border-radius:3px;padding:1px 0;box-decoration-break:clone;-webkit-box-decoration-break:clone;
     transition:background-color .15s}
.seg.focus{outline:2px solid var(--ink);outline-offset:1px}
.drop textarea{outline:2px dashed var(--accent);outline-offset:-10px}
aside{position:sticky;top:16px;display:grid;gap:14px}
.card{background:var(--sheet);border:1px solid var(--line);border-radius:14px;padding:18px}
.card h2{margin:0 0 6px;font-size:.95rem;font-weight:700;color:var(--muted)}
#verdict{font-size:1.3rem;font-weight:700;line-height:1.25;margin:0}
#detail{margin:6px 0 0;color:var(--muted);font-size:.95rem}
.meter{height:10px;border-radius:99px;background:var(--line);margin:14px 0 4px;overflow:hidden}
.meter i{display:block;height:100%;width:0;background:var(--ia);border-radius:99px;transition:width .4s}
.meter-l{display:flex;justify-content:space-between;color:var(--muted);font-size:.8rem}
#strip{display:flex;gap:2px;height:30px;margin-top:10px}
#strip button{padding:0;min-width:3px;height:100%;border:0;border-radius:3px;cursor:pointer}
#strip button:hover{filter:brightness(1.15)}
#hint{min-height:2.6em;margin:10px 0 0;font-size:.9rem;color:var(--muted)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:.82rem;color:var(--muted);margin-top:6px}
.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:3px;
  margin-right:6px;background:var(--c)}
.note{font-size:.85rem;color:var(--muted);margin:0}
.empty{color:var(--muted);margin:0}
.err{color:var(--ia);font-weight:500}
@media (max-width:880px){main{grid-template-columns:1fr}aside{position:static;order:-1}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<header>
  <h1>Opsiom Détecteur</h1>
  <p>Repère les passages d'un texte français probablement écrits par une IA.</p>
</header>

<main>
  <section class="sheet" id="sheet" aria-label="Texte à analyser">
    <div class="sheet-bar">
      <button class="primary" id="go">Analyser</button>
      <button id="edit" hidden>Modifier le texte</button>
      <label class="btn" for="file">Ouvrir un .txt</label>
      <input id="file" type="file" accept=".txt,text/plain" hidden>
      <button id="clear">Vider</button>
      <span class="count" id="count">0 caractère</span>
    </div>
    <textarea id="t" spellcheck="false"
      placeholder="Colle un texte ici, ou dépose un fichier .txt. Compte au moins quelques phrases : en dessous de 24 tokens, le score n'est pas fiable."></textarea>
    <div class="reading" id="reading" hidden></div>
  </section>

  <aside>
    <div class="card" id="result">
      <h2>Résultat</h2>
      <p class="empty" id="empty">Lance une analyse pour voir le verdict.</p>
      <div id="out" hidden>
        <p id="verdict"></p>
        <p id="detail"></p>
        <div class="meter" aria-hidden="true"><i id="bar"></i></div>
        <div class="meter-l"><span>Humain</span><span id="pct"></span><span>IA</span></div>
        <div id="strip" role="list" aria-label="Phrases du texte, de la première à la dernière"></div>
        <p id="hint">Survole ou sélectionne une phrase pour voir son score.</p>
        <div class="legend">
          <span style="--c:var(--ok)">peu probable</span>
          <span style="--c:var(--warn)">douteux</span>
          <span style="--c:var(--ia)">signalé</span>
        </div>
      </div>
      <p class="err" id="err" role="alert" hidden></p>
    </div>
    <div class="card">
      <p class="note"><strong>À lire avant d'agir.</strong> Ce score est un indice, pas une preuve. Il se trompe plus
      souvent sur les textes courts, les traductions, les textes reformulés par une IA et les rédactions
      d'élèves peu à l'aise en français. Ne sanctionne personne sur ce seul résultat.</p>
    </div>
  </aside>
</main>

<script>
const $ = id => document.getElementById(id);
const MAX = 50000;
const LABEL = {
  ia: "Probablement écrit par une IA",
  mixte: "Passages probablement générés par une IA",
  humain: "Probablement écrit par un humain",
  incertain: "Résultat incertain",
  trop_court: "Texte trop court"
};
const pctf = x => Math.round(x * 100) + " %";

function hue(p){ return 165 - Math.min(1, Math.max(0, p)) * 157; }   // vert -> corail
function color(s){
  if (s.flagged) return "var(--ia)";
  return s.ai_probability > 0.35 ? "var(--warn)" : "var(--ok)";
}
function tint(s){
  if (s.flagged) return "color-mix(in srgb, var(--ia) 38%, transparent)";
  if (s.ai_probability > 0.35) return "color-mix(in srgb, var(--warn) 26%, transparent)";
  return "transparent";
}

function updateCount(){
  const n = $("t").value.length;
  $("count").textContent = n.toLocaleString("fr-FR") + " caractère" + (n > 1 ? "s" : "") +
    (n > MAX ? " (limite : 50 000)" : "");
  $("go").disabled = n === 0 || n > MAX;
}

function showError(msg){
  $("err").textContent = msg; $("err").hidden = false;
  $("out").hidden = true; $("empty").hidden = true;
}

function backToEditing(){
  $("reading").hidden = true; $("t").hidden = false; $("edit").hidden = true; $("go").hidden = false;
  $("t").focus();
}

function focusSeg(i, scroll){
  document.querySelectorAll(".seg.focus").forEach(e => e.classList.remove("focus"));
  const el = document.querySelector('.seg[data-i="' + i + '"]');
  if (el){ el.classList.add("focus"); if (scroll) el.scrollIntoView({block:"center", behavior:"smooth"}); }
}

function render(res){
  $("err").hidden = true; $("empty").hidden = true; $("out").hidden = false;
  $("verdict").textContent = LABEL[res.verdict] || res.verdict;
  $("strip").replaceChildren();
  $("reading").replaceChildren();

  if (res.ai_probability == null){
    $("detail").textContent = res.message || "";
    $("bar").style.width = "0"; $("pct").textContent = "";
    return;
  }
  $("detail").textContent = pctf(res.ai_fraction) + " du texte signalé · " + res.n_tokens + " tokens";
  $("bar").style.width = pctf(res.ai_probability);
  $("pct").textContent = pctf(res.ai_probability);

  const text = res.text, segs = res.segments || [];
  let pos = 0;
  segs.forEach((s, i) => {
    if (s.start > pos) $("reading").append(document.createTextNode(text.slice(pos, s.start)));
    const span = document.createElement("span");
    span.className = "seg"; span.dataset.i = i;
    span.textContent = text.slice(s.start, s.end);
    span.style.backgroundColor = tint(s);
    span.addEventListener("mouseenter", () => hint(i));
    $("reading").append(span);
    pos = s.end;

    const b = document.createElement("button");
    b.style.flex = String(Math.max(1, s.n_tokens)); b.style.background = color(s);
    b.setAttribute("role", "listitem");
    b.setAttribute("aria-label", "Phrase " + (i + 1) + " : " + pctf(s.ai_probability) + " IA");
    b.addEventListener("click", () => { hint(i); focusSeg(i, true); });
    b.addEventListener("mouseenter", () => hint(i));
    $("strip").append(b);
  });
  if (pos < text.length) $("reading").append(document.createTextNode(text.slice(pos)));

  window._segs = segs;
  $("t").hidden = true; $("reading").hidden = false; $("edit").hidden = false; $("go").hidden = true;
}

function hint(i){
  const s = window._segs && window._segs[i];
  if (!s) return;
  const short = s.text.length > 90 ? s.text.slice(0, 90) + "…" : s.text;
  $("hint").textContent = "Phrase " + (i + 1) + " : " + pctf(s.ai_probability) + " IA" +
    (s.flagged ? " (signalée)" : "") + " — « " + short + " »";
}

async function analyse(){
  const text = $("t").value;
  $("go").disabled = true; $("go").textContent = "Analyse…";
  try{
    const r = await fetch("/api/analyse", {method:"POST", headers:{"Content-Type":"application/json"},
                                            body: JSON.stringify({text})});
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Erreur du serveur");
    render(d);
  }catch(e){
    showError(e.message === "Failed to fetch"
      ? "Le serveur ne répond pas. Vérifie qu'il tourne toujours." : e.message);
  }finally{
    $("go").textContent = "Analyser"; updateCount();
  }
}

function loadFile(f){
  if (!f) return;
  const rd = new FileReader();
  rd.onload = () => { backToEditing(); $("t").value = String(rd.result).slice(0, MAX); updateCount(); };
  rd.readAsText(f, "utf-8");
}

$("go").onclick = analyse;
$("edit").onclick = backToEditing;
$("clear").onclick = () => { backToEditing(); $("t").value = ""; updateCount();
  $("out").hidden = true; $("empty").hidden = false; $("err").hidden = true; $("t").focus(); };
$("t").addEventListener("input", updateCount);
$("t").addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && !$("go").disabled) analyse(); });
$("file").onchange = e => loadFile(e.target.files[0]);
["dragover","dragenter"].forEach(ev => $("sheet").addEventListener(ev, e => { e.preventDefault(); $("sheet").classList.add("drop"); }));
["dragleave","drop"].forEach(ev => $("sheet").addEventListener(ev, e => { e.preventDefault(); $("sheet").classList.remove("drop"); }));
$("sheet").addEventListener("drop", e => loadFile(e.dataTransfer.files[0]));
updateCount();
</script>
</body>
</html>
"""


def create_app(args) -> Flask:
    mod, det = charger_detecteur(args)
    app = Flask(__name__)
    app.register_blueprint(mod.create_detector_blueprint(det))  # POST /detect, GET /detect/health

    @app.get("/")
    def index():
        return PAGE

    @app.post("/api/analyse")
    def analyse():
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Le texte est vide."}), 400
        if len(text) > 50000:
            return jsonify({"error": "Texte trop long (50 000 caractères maximum)."}), 413
        res = det.predict(text, segments=True)
        res["text"] = mod.normalize_text(text)  # les positions des phrases se rapportent à ce texte
        return jsonify(res)

    return app


def main():
    p = argparse.ArgumentParser(description="Interface web du détecteur Opsiom")
    p.add_argument("--script", default=None, help="chemin de Opsiom-Detector.py")
    p.add_argument("--checkpoint", default=None, help="chemin de opsiom_detector_best.pt")
    p.add_argument("--port", type=int, default=5055)
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 pour l'ouvrir au réseau local")
    args = p.parse_args()
    create_app(args).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()