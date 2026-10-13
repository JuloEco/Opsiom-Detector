#!/usr/bin/env python3
# ============================================================================
# detecteur_app.py — utilise le détecteur Opsiom entraîné (Opsiom-Detector.py)
#
#   python detecteur_app.py texte "Voici le texte à tester..."
#   python detecteur_app.py fichier devoir.txt
#   python detecteur_app.py dossier ./copies --csv rapport.csv
#   python detecteur_app.py serve --port 5055      # petite page web + API /detect
#
# Options communes :
#   --script Opsiom-Detector.py   chemin du script du détecteur (défaut : à côté de ce fichier)
#   --checkpoint chemin.pt        checkpoint local (défaut : opsiom_detector_best.pt,
#                                 téléchargé depuis le Hub si absent — HF_TOKEN requis, repo privé)
#
# ⚠️ Le résultat est un INDICE, jamais une preuve : ne sanctionne personne sur ce seul score.
# ============================================================================
import argparse
import csv
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROUGE, JAUNE, VERT, GRIS, RESET = "\033[91m", "\033[93m", "\033[92m", "\033[90m", "\033[0m"
COULEUR_VERDICT = {"ia": ROUGE, "mixte": JAUNE, "incertain": JAUNE, "humain": VERT, "trop_court": GRIS}


def charger_module(script: str | None):
    """Importe Opsiom-Detector.py (le tiret empêche un `import` classique)."""
    candidats = [script] if script else [os.path.join(HERE, n) for n in
                                         ("Opsiom-Detector.py", "opsiom_detector.py")]
    chemin = next((c for c in candidats if c and os.path.exists(c)), None)
    if not chemin:
        sys.exit("Script du détecteur introuvable : passe --script /chemin/Opsiom-Detector.py")
    spec = importlib.util.spec_from_file_location("opsiom_detector", chemin)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opsiom_detector"] = mod
    spec.loader.exec_module(mod)
    return mod


def charger_detecteur(args):
    mod = charger_module(args.script)
    det = mod.Detector.from_checkpoint(args.checkpoint)
    print(f"✅ Détecteur prêt (seq_len={det.seq_len}, seuil IA={det.calib['thr_ia']:.2f} en logit)")
    return mod, det


# ----------------------------------------------------------------------------
# Affichage terminal
# ----------------------------------------------------------------------------
def afficher(res: dict) -> None:
    v = res["verdict"]
    print(f"\n{COULEUR_VERDICT.get(v, '')}■ Verdict : {v.upper()}{RESET}")
    if res.get("ai_probability") is None:
        print(res.get("message", ""))
        return
    print(f"  Probabilité IA globale : {res['ai_probability']:.1%}")
    print(f"  Part du texte signalée : {res['ai_fraction']:.0%}  ({res['n_tokens']} tokens)\n")
    for s in res.get("segments", []):
        c = ROUGE if s["flagged"] else (JAUNE if s["ai_probability"] > 0.5 else "")
        print(f"  {c}[{s['ai_probability']:.0%}]{RESET} {c}{s['text']}{RESET}")
    print("\n⚠️  Indice seulement : aucun détecteur n'est fiable à 100 %.")


# ----------------------------------------------------------------------------
# Commandes
# ----------------------------------------------------------------------------
def cmd_texte(det, args):
    res = det.predict(args.texte)
    print(json.dumps(res, ensure_ascii=False, indent=2)) if args.json else afficher(res)


def cmd_fichier(det, args):
    with open(args.fichier, encoding="utf-8", errors="ignore") as f:
        res = det.predict(f.read())
    print(json.dumps(res, ensure_ascii=False, indent=2)) if args.json else afficher(res)


def cmd_dossier(det, args):
    fichiers = sorted(os.path.join(r, n) for r, _, ns in os.walk(args.dossier) for n in ns if n.endswith(".txt"))
    if not fichiers:
        sys.exit(f"Aucun .txt dans {args.dossier}")
    lignes = []
    for fp in fichiers:
        with open(fp, encoding="utf-8", errors="ignore") as f:
            r = det.predict(f.read(), segments=False)
        lignes.append({"fichier": os.path.relpath(fp, args.dossier), "verdict": r["verdict"],
                       "proba_ia": r.get("ai_probability"), "part_ia": r.get("ai_fraction"),
                       "n_tokens": r["n_tokens"]})
        c = COULEUR_VERDICT.get(r["verdict"], "")
        print(f"{c}{r['verdict']:<10}{RESET} {r.get('ai_probability')}  {lignes[-1]['fichier']}")
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)
    print(f"\n💾 Rapport : {args.csv}")


PAGE = """<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Opsiom Détecteur</title>
<style>
body{font-family:system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
textarea{width:100%;min-height:180px;padding:.7rem;font:inherit;border:1px solid #bbb;border-radius:8px;box-sizing:border-box}
button{margin-top:.6rem;padding:.6rem 1.2rem;border:0;border-radius:8px;background:#3b4cca;color:#fff;font:inherit;cursor:pointer}
#verdict{margin:1.2rem 0 .4rem;font-size:1.2rem;font-weight:600}
.seg{padding:1px 2px;border-radius:3px}.note{color:#666;font-size:.85rem;margin-top:1rem}
</style></head><body>
<h1>Opsiom Détecteur</h1>
<textarea id="t" placeholder="Colle ici le texte à analyser (français, 50 tokens minimum)…"></textarea>
<button id="go">Analyser</button>
<div id="verdict"></div><div id="out"></div>
<p class="note">⚠️ Indice seulement : ne jamais sanctionner quelqu'un sur ce seul résultat.</p>
<script>
const COL={ia:"#c0392b",mixte:"#d68910",incertain:"#d68910",humain:"#1e8449",trop_court:"#888"};
document.getElementById("go").onclick=async()=>{
  const v=document.getElementById("verdict"),o=document.getElementById("out");
  v.textContent="Analyse…";o.replaceChildren();
  const r=await fetch("/detect",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({text:document.getElementById("t").value})});
  const d=await r.json();
  if(!r.ok){v.textContent=d.error||"Erreur";return}
  v.style.color=COL[d.verdict]||"#000";
  v.textContent=d.verdict==="trop_court"?d.message:
    "Verdict : "+d.verdict+" — "+Math.round(d.ai_probability*100)+" % IA, "+Math.round(d.ai_fraction*100)+" % du texte signalé";
  (d.segments||[]).forEach(s=>{
    const e=document.createElement("span");e.className="seg";e.textContent=s.text+" ";
    e.title=Math.round(s.ai_probability*100)+" % IA";
    e.style.background="rgba(192,57,43,"+(s.flagged?0.35:s.ai_probability*0.2)+")";
    o.appendChild(e);});
};
</script></body></html>"""


def cmd_serve(mod, det, args):
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(mod.create_detector_blueprint(det))  # POST /detect, GET /detect/health
    app.add_url_rule("/", "index", lambda: PAGE)
    app.run(host="0.0.0.0", port=args.port)


def main():
    p = argparse.ArgumentParser(description="Utilise le détecteur Opsiom (texte IA vs humain)")
    p.add_argument("--script", default=None)
    p.add_argument("--checkpoint", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("texte"); a.add_argument("texte"); a.add_argument("--json", action="store_true")
    b = sub.add_parser("fichier"); b.add_argument("fichier"); b.add_argument("--json", action="store_true")
    c = sub.add_parser("dossier"); c.add_argument("dossier"); c.add_argument("--csv", default="rapport.csv")
    d = sub.add_parser("serve"); d.add_argument("--port", type=int, default=5055)
    args = p.parse_args()

    mod, det = charger_detecteur(args)
    if args.cmd == "serve":
        cmd_serve(mod, det, args)
    else:
        {"texte": cmd_texte, "fichier": cmd_fichier, "dossier": cmd_dossier}[args.cmd](det, args)


if __name__ == "__main__":
    main()