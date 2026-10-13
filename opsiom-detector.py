# ============================================================================
# Opsiom Détecteur — mini détecteur de texte généré par IA vs écrit par un humain
# Architecture LLaMA/Qwen-style (RMSNorm + RoPE + SwiGLU + GQA + SDPA), reprise
# de main-GPU-CPU-TPU.py (Opsiom 500M / Kaïro) et d'Opsiom Large.
# ============================================================================
#
# Ce que fait ce script (un seul fichier, notebook-friendly Colab/Kaggle) :
#
#   1. build     : construit un dataset FR étiqueté humain (0) / IA (1)
#   2. generate  : (optionnel) génère des « miroirs » IA avec un LLM instruct
#   3. train     : fine-tune un Transformer Opsiom en classifieur
#   4. eval      : métriques sur la validation (AUROC, TPR à 1 % de FPR, par source)
#   5. predict   : score un texte, avec surlignage phrase par phrase
#   6. mine      : cherche des faux positifs sur du texte humain frais (hard negatives)
#   7. serve     : petite API Flask (/detect) pour brancher le détecteur à
#                  l'écosystème Opsiom (Classroom, Omnia...)
#
# Méthode — inspirée de la description publique du classifieur de Pangram
# (Emi & Spero, 2024), à échelle « mini » :
#   - classifieur Transformer entraîné sur du texte humain vs du texte IA ;
#   - « miroirs synthétiques » : pour un texte humain, un LLM écrit un texte IA
#     sur le même sujet / de la même longueur → le détecteur ne peut pas
#     tricher en détectant le sujet ou le genre ;
#   - « hard negative mining » : on cherche les textes humains que le
#     détecteur croit IA, on les ajoute au dataset, on ré-entraîne ;
#   - priorité au FAIBLE taux de faux positifs : on calibre le seuil sur la
#     validation pour qu'au plus ~1 % des textes humains soient accusés à tort
#     (accuser un élève à tort coûte bien plus cher que rater un texte IA).
#
# Particularités de cette implémentation :
#   - Le modèle reste CAUSAL (comme le LM pré-entraîné : les poids d'Opsiom
#     Large / Kaïro se chargent tels quels). Une tête linéaire (dim -> 1) donne
#     un logit « IA » PAR TOKEN ; le score du document = moyenne de ces logits
#     (en ignorant les premiers tokens, qui n'ont presque pas de contexte).
#   - Entraînement sur des exemples MIXTES (segment humain + segment IA collés)
#     avec étiquettes par token → le détecteur sait localiser le passage IA.
#   - Augmentations anti-raccourcis : longueurs aléatoires, markdown retiré
#     aléatoirement sur les textes IA, textes humains « style assistant »
#     (OpenAssistant FR) pour que « ton de chatbot » ≠ « IA ».
#   - Texte humain web (FineWeb2-HQ) filtré sur la date du crawl : on ne garde
#     que les pages d'AVANT ChatGPT (30/11/2022), sinon des textes IA se
#     glissent dans la classe « humain ».
#
# ⚠️ Limites assumées (à lire) :
#   - Aucun détecteur n'est fiable à 100 % : à utiliser comme indice, jamais
#     comme preuve unique (surtout pour sanctionner quelqu'un).
#   - Il ne détectera bien que les générateurs qu'il a vus à l'entraînement
#     (GPT-3.5 via French-Alpaca + les modèles de l'étape `generate`). Plus tu
#     varies les générateurs, mieux il généralise.
#   - Textes courts (< ~50 tokens), traductions, textes paraphrasés par une IA
#     ou écrits par des non-natifs : taux d'erreur nettement plus élevés.
#   - Accélérateurs gérés : CUDA et CPU. Pas de TPU dans ce script (non testé).
#   - RoPE : Large utilise la convention « rotate_half », Kaïro-CUDA la
#     convention « complexe/entrelacée » — les deux ne sont PAS interchangeables
#     pour un checkpoint donné. Voir detect_rope_style() : il compare la loss
#     du LM sous les deux conventions et prend la bonne automatiquement.
#
# Utilisation rapide (Colab/Kaggle, GPU) :
#   !pip install -q tokenizers datasets huggingface_hub
#   !python opsiom_detector.py --stage build
#   !python opsiom_detector.py --stage train --init-from large
#   !python opsiom_detector.py --stage predict --text "Voici un texte à tester."
#
# Variables d'environnement utiles : HF_TOKEN (sauvegarde sur le Hub),
# OPSIOM_DET_DIR (dossier de travail), OPSIOM_HF_BACKUP=0 (désactive le Hub).
# ============================================================================

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import re
import json
import math
import time
import glob
import pickle
import types
import shutil
import random
import hashlib
import argparse
import builtins as _builtins
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# print() non-bufferisé (nécessaire en notebook Kaggle/Colab) — idempotent si
# le script est relancé plusieurs fois dans le même kernel.
if not getattr(_builtins.print, "_is_flushing_wrapper", False):
    _orig_print = _builtins.print

    def _flushing_print(*a, **k):
        k.setdefault("flush", True)
        _flushing_print._original(*a, **k)

    _flushing_print._original = _orig_print
    _flushing_print._is_flushing_wrapper = True
    _builtins.print = _flushing_print


# ============================================================================
# Accélérateur et environnement (CUDA / CPU ; Modal, RunPod, Colab, Kaggle, local)
# ============================================================================
ACCEL_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(ACCEL_TYPE)
if ACCEL_TYPE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _select_amp_dtype():
    """bf16 sur Ampere+ (Tensor Cores), fp16 sur Turing/T4 (loss-scaling géré
    plus bas par GradScaler), pas d'autocast sur CPU — même logique que
    _select_mixed_precision() de main-GPU-CPU-TPU.py."""
    if ACCEL_TYPE != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


AMP_DTYPE = _select_amp_dtype()


def _is_kaggle() -> bool:
    return os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def _detect_dirs():
    """(save_dir, data_dir). save_dir = petits fichiers précieux (checkpoints) ;
    data_dir = grosses données reconstructibles (dataset tokenisé). Sur Kaggle,
    data_dir part sur /kaggle/tmp (hors du quota de 20 Go de /kaggle/working),
    exactement comme pour le corpus d'Opsiom 500M."""
    override = os.environ.get("OPSIOM_DET_DIR")
    if override:
        save_dir = override
    elif "MODAL_TASK_ID" in os.environ or os.path.exists("/vol"):
        save_dir = "/vol/opsiom_detector"
    elif "RUNPOD_POD_ID" in os.environ or "RUNPOD_GPU_COUNT" in os.environ:
        save_dir = "/workspace/opsiom_detector"
    elif _is_kaggle():
        save_dir = "/kaggle/working/opsiom_detector"
    elif os.path.exists("/content"):  # Colab
        try:
            from google.colab import drive  # type: ignore
            drive.mount("/content/drive")
            save_dir = "/content/drive/MyDrive/opsiom_detector"
        except Exception:
            save_dir = "/content/opsiom_detector"
    else:
        save_dir = os.path.join(".", "opsiom_detector")
    data_dir = "/kaggle/tmp/opsiom_detector_data" if _is_kaggle() and not override else os.path.join(save_dir, "data")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    return save_dir, data_dir


_SAVE_DIR, _DATA_DIR = _detect_dirs()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Sauvegarde externe (Hugging Face Hub) — mêmes réflexes que le script 500M
# ============================================================================
# Préfixe distinct : on n'écrase JAMAIS best_model.pt (Large) ni les
# checkpoints opsiom_500m_32k_* qui vivent dans le même repo.
HF_BACKUP_ENABLED = os.environ.get("OPSIOM_HF_BACKUP", "1") != "0"
HF_BACKUP_REPO_ID = "JuloEco/opsiom-fr-checkpoints"
HF_BACKUP_PRIVATE = True
HF_LATEST_INTERVAL_SECONDS = 600
CKPT_PREFIX = "opsiom_detector_"
BEST_PATH = os.path.join(_SAVE_DIR, f"{CKPT_PREFIX}best.pt")
LATEST_PATH = os.path.join(_SAVE_DIR, f"{CKPT_PREFIX}latest.pt")

_hf_api = None


def _get_hf_api():
    global _hf_api
    if _hf_api is not None:
        return _hf_api
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=HF_BACKUP_REPO_ID, private=HF_BACKUP_PRIVATE, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"⚠️ Sauvegarde HF indisponible ({e}) — désactivée pour cette session.")
        return None
    _hf_api = api
    return api


def push_backup_to_hf(local_path: str, path_in_repo: str) -> None:
    if not HF_BACKUP_ENABLED or not os.path.exists(local_path):
        return
    api = _get_hf_api()
    if api is None:
        return
    try:
        api.upload_file(path_or_fileobj=local_path, path_in_repo=path_in_repo,
                        repo_id=HF_BACKUP_REPO_ID, repo_type="model")
        print(f"   ↳ 🛟 '{path_in_repo}' poussé vers {HF_BACKUP_REPO_ID} "
              f"({os.path.getsize(local_path) / 1e6:.1f} Mo)")
    except Exception as e:
        print(f"   ↳ ⚠️ Échec de la sauvegarde externe de '{path_in_repo}' ({e}) — non bloquant.")


def hf_download(repo_id: str, filename: str, dest: str = None) -> str | None:
    """Télécharge un fichier du Hub (cache HF), le copie vers `dest` si fourni.
    Renvoie le chemin local, ou None en cas d'échec (jamais d'exception)."""
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=repo_id, filename=filename)
        if dest:
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            shutil.copy2(p, dest)
            return dest
        return p
    except Exception as e:
        print(f"ℹ️ '{filename}' introuvable/inaccessible sur {repo_id} ({type(e).__name__}).")
        return None


# ============================================================================
# Presets Opsiom (checkpoint de départ + tokenizer + convention RoPE)
# ============================================================================
PRESETS = {
    # Opsiom Large (197M, vocab 16k) — checkpoint final, déjà sur le Hub.
    "large": dict(ckpt_repo="JuloEco/opsiom-fr-checkpoints", ckpt_file="best_model.pt",
                  tok_repo="JuloEco/opsiom-fr-checkpoints", tok_file="fr_bpe_tokenizer.json",
                  rope_style="half"),
    # Opsiom Kaïro (511M, vocab 32k) — dès que le pré-entraînement est abouti.
    "kairo": dict(ckpt_repo="JuloEco/opsiom-fr-checkpoints", ckpt_file="opsiom_500m_32k_best_model.pt",
                  tok_repo="JuloEco/opsiom-fr-tokenizer", tok_file="tokenizer_32k.json",
                  rope_style="interleaved"),
    # Petit modèle from scratch (pour tester / sans checkpoint) : tokenizer de Large.
    "scratch": dict(ckpt_repo=None, ckpt_file=None,
                    tok_repo="JuloEco/opsiom-fr-checkpoints", tok_file="fr_bpe_tokenizer.json",
                    rope_style="half"),
}


def load_tokenizer(spec: str):
    """`spec` = chemin d'un tokenizer.json local, ou nom de preset (large/kairo/scratch)."""
    from tokenizers import Tokenizer
    if os.path.exists(spec):
        tok = Tokenizer.from_file(spec)
    else:
        p = PRESETS[spec]
        local = os.path.join(_SAVE_DIR, f"tokenizer_{spec}.json")
        if not os.path.exists(local):
            if hf_download(p["tok_repo"], p["tok_file"], local) is None:
                raise RuntimeError(f"Tokenizer '{spec}' introuvable ({p['tok_repo']}/{p['tok_file']}). "
                                   f"Passe --tokenizer /chemin/vers/tokenizer.json.")
        tok = Tokenizer.from_file(local)
    print(f"✅ Tokenizer chargé — vocab_size={tok.get_vocab_size()}")
    return tok


def special_ids(tok) -> dict:
    pad = tok.token_to_id("<pad>")
    return {"pad": pad if pad is not None else 0}


# ============================================================================
# Architecture — identique (noms de modules compris) à Opsiom Large / Kaïro
# pour que load_state_dict() charge les poids pré-entraînés, + tête de détection
# ============================================================================
@dataclass
class ModelArgs:
    vocab_size: int = 16000
    dim: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int = 4
    ffn_dim: int = 2816
    max_seq_len: int = 512
    dropout: float = 0.1
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    rope_style: str = "half"          # "half" (rotate_half) | "interleaved" (complexe)
    grad_checkpointing: bool = True   # recalcul des activations : indispensable sur T4


SCRATCH_ARGS = dict(dim=384, n_layers=8, n_heads=6, n_kv_heads=2, ffn_dim=1024, max_seq_len=512)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        n = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return n.type_as(x) * self.weight


def precompute_rope(head_dim: int, max_len: int, theta: float, style: str):
    """Version RÉELLE (cos/sin) des deux conventions — pas de nombres complexes,
    donc pas le bug view_as_complex/bf16 rencontré sur TPU."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    fr = torch.outer(torch.arange(max_len).float(), inv)  # (L, head_dim/2)
    if style == "half":
        emb = torch.cat((fr, fr), dim=-1)
        return emb.cos(), emb.sin()                        # (L, head_dim)
    return fr.cos(), fr.sin()                              # (L, head_dim/2)


def apply_rope(x, cos, sin, style: str):
    # x : (B, T, H, head_dim) ; cos/sin : (T, head_dim) ou (T, head_dim/2)
    xf = x.float()
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    if style == "half":
        half = xf.shape[-1] // 2
        rot = torch.cat((-xf[..., half:], xf[..., :half]), dim=-1)
        out = xf * cos + rot * sin
    else:
        x1, x2 = xf[..., 0::2], xf[..., 1::2]
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.type_as(x)


def repeat_kv(x, n_rep: int):
    if n_rep == 1:
        return x
    b, s, h, d = x.shape
    return x[:, :, :, None, :].expand(b, s, h, n_rep, d).reshape(b, s, h * n_rep, d)


class Attention(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.n_heads, self.n_kv_heads = a.n_heads, a.n_kv_heads
        self.n_rep = a.n_heads // a.n_kv_heads
        self.head_dim = a.dim // a.n_heads
        self.dropout = a.dropout
        self.rope_style = a.rope_style
        self.wq = nn.Linear(a.dim, a.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(a.dim, a.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(a.dim, a.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(a.n_heads * self.head_dim, a.dim, bias=False)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim)
        q, k = apply_rope(q, cos, sin, self.rope_style), apply_rope(k, cos, sin, self.rope_style)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        # Causal : avec du padding À DROITE, les vrais tokens ne voient jamais
        # les tokens de padding → aucun masque de padding nécessaire ici.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.wo(out.transpose(1, 2).contiguous().view(b, t, -1))


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.attention_norm = RMSNorm(a.dim, a.norm_eps)
        self.attention = Attention(a)
        self.ffn_norm = RMSNorm(a.dim, a.norm_eps)
        self.feed_forward = FeedForward(a.dim, a.ffn_dim, a.dropout)

    def forward(self, x, cos, sin):
        x = x + self.attention(self.attention_norm(x), cos, sin)
        return x + self.feed_forward(self.ffn_norm(x))


class DetectorModel(nn.Module):
    """Backbone Opsiom (causal) + tête linéaire dim -> 1 appliquée à CHAQUE token.
    `self.output` (tête LM, tied aux embeddings) est conservée uniquement pour
    charger les checkpoints et pour detect_rope_style()."""

    def __init__(self, a: ModelArgs):
        super().__init__()
        self.args = a
        self.tok_embeddings = nn.Embedding(a.vocab_size, a.dim)
        self.layers = nn.ModuleList([TransformerBlock(a) for _ in range(a.n_layers)])
        self.norm = RMSNorm(a.dim, a.norm_eps)
        self.output = nn.Linear(a.dim, a.vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight
        self.cls_dropout = nn.Dropout(0.1)
        self.cls_head = nn.Linear(a.dim, 1)
        self._build_rope()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _build_rope(self):
        a = self.args
        cos, sin = precompute_rope(a.dim // a.n_heads, a.max_seq_len * 2, a.rope_theta, a.rope_style)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def set_rope_style(self, style: str):
        self.args.rope_style = style
        for blk in self.layers:
            blk.attention.rope_style = style
        dev = self.rope_cos.device
        self._build_rope()
        self.rope_cos, self.rope_sin = self.rope_cos.to(dev), self.rope_sin.to(dev)

    def hidden(self, tokens):
        t = tokens.shape[1]
        h = self.tok_embeddings(tokens)
        cos, sin = self.rope_cos[:t], self.rope_sin[:t]
        for layer in self.layers:
            if self.training and self.args.grad_checkpointing:
                h = torch.utils.checkpoint.checkpoint(layer, h, cos, sin, use_reentrant=False)
            else:
                h = layer(h, cos, sin)
        return self.norm(h)

    def forward(self, tokens):
        """-> logits « IA » par token, (B, T), float32."""
        h = self.cls_dropout(self.hidden(tokens))
        return self.cls_head(h).squeeze(-1).float()

    @torch.no_grad()
    def lm_loss(self, tokens):
        h = self.hidden(tokens[:, :-1])
        return F.cross_entropy(self.output(h).float().reshape(-1, self.args.vocab_size),
                               tokens[:, 1:].reshape(-1)).item()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# Chargement d'un checkpoint Opsiom pré-entraîné
# ----------------------------------------------------------------------------
# Noms des poids d'Opsiom Large (script main_GPU.py) -> noms de ce script (= Kaïro).
_LARGE_KEY_RULES = [
    (r"\.attn\.", ".attention."), (r"\.ffn\.", ".feed_forward."),
    (r"\.norm1\.", ".attention_norm."), (r"\.norm2\.", ".ffn_norm."),
    (r"(^|\.)norm_f\.", r"\1norm."), (r"(^|\.)lm_head\.", r"\1output."),
]


def _rename_large_key(k: str) -> str:
    for pat, rep in _LARGE_KEY_RULES:
        k = re.sub(pat, rep, k)
    return k


def _extract_state_dict(ckpt):
    sd = ckpt
    if isinstance(ckpt, dict):
        for k in ("model", "model_state_dict", "state_dict", "model_state", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict) and ckpt[k]:
                sd = ckpt[k]
                break
    sd = {re.sub(r"^(module\.|_orig_mod\.)+", "", k): v for k, v in sd.items() if hasattr(v, "shape")}
    if any(re.match(r"(.*\.)?layers\.\d+\.attn\.wq\.weight$", k) for k in sd):
        sd = {_rename_large_key(k): v for k, v in sd.items()}
    # Préfixe résiduel (ex. « model.layers.0... », « transformer.tok_embeddings... ») : on le retire.
    anchor = next((k for k in sd if k.endswith("tok_embeddings.weight")), None)
    if anchor and anchor != "tok_embeddings.weight":
        pre = anchor[: -len("tok_embeddings.weight")]
        sd = {(k[len(pre):] if k.startswith(pre) else k): v for k, v in sd.items()}
    return sd


def describe_checkpoint(ckpt, n: int = 14) -> str:
    """Résumé lisible d'un checkpoint (clés de premier niveau + premières/dernières clés de poids)."""
    lines = []
    if isinstance(ckpt, dict):
        lines.append("Clés de premier niveau : " + ", ".join(f"{k}({type(v).__name__})" for k, v in list(ckpt.items())[:20]))
    sd = _extract_state_dict(ckpt)
    items = [(k, tuple(v.shape)) for k, v in sd.items()]
    lines.append(f"{len(items)} tenseurs. Premiers :")
    lines += [f"   {k} {sh}" for k, sh in items[:n]]
    lines.append("   ...")
    lines += [f"   {k} {sh}" for k, sh in items[-6:]]
    return "\n".join(lines)


def infer_args_from_state_dict(sd: dict, n_heads: int | None = None) -> ModelArgs:
    """Reconstruit ModelArgs depuis les SHAPES des poids (utile si le checkpoint
    ne contient pas ses args). Seul n_heads n'est pas déductible : on suppose
    head_dim=64 (vrai pour Large/Kaïro : 1024/16) sauf si --n-heads est donné."""
    needed = ("tok_embeddings.weight", "layers.0.attention.wq.weight", "layers.0.attention.wk.weight",
              "layers.0.feed_forward.w1.weight")
    absent = [k for k in needed if k not in sd]
    if absent:
        raise RuntimeError(
            "Noms de poids inattendus dans ce checkpoint (attendus : " + ", ".join(absent) + ").\n"
            + describe_checkpoint(sd) + "\n→ Ce checkpoint n'utilise pas les noms d'Opsiom Kaïro. Colle-moi ce résumé "
            "(ou lance --stage inspect) pour que j'ajoute le mapping.")
    vocab, dim = sd["tok_embeddings.weight"].shape
    n_layers = 1 + max(int(m.group(1)) for k in sd if (m := re.match(r"layers\.(\d+)\.", k)))
    n_heads = n_heads or max(1, dim // 64)
    head_dim = dim // n_heads
    n_kv = sd["layers.0.attention.wk.weight"].shape[0] // head_dim
    ffn = sd["layers.0.feed_forward.w1.weight"].shape[0]
    return ModelArgs(vocab_size=vocab, dim=dim, n_layers=n_layers, n_heads=n_heads,
                     n_kv_heads=n_kv, ffn_dim=ffn)


class _LenientUnpickler(pickle.Unpickler):
    """Les checkpoints d'Opsiom contiennent un objet `ModelArgs` picklé depuis le
    `__main__` du script d'entraînement. Si cette classe n'existe pas dans le
    script courant, le chargement plante : on la remplace par un objet vide qui
    garde les attributs (n_heads, rope_theta...) lisibles."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ImportError):
            return type(name, (), {})


def _lenient_pickle_module():
    m = types.ModuleType("lenient_pickle")
    m.__dict__.update({k: v for k, v in vars(pickle).items() if not k.startswith("__")})
    m.Unpickler = _LenientUnpickler
    return m


def torch_load_safe(path: str):
    # weights_only=True d'abord (le plus sûr), puis repli pour les checkpoints
    # contenant des objets (ModelArgs), avec un Unpickler qui tolère les classes absentes.
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        pass
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False, pickle_module=_lenient_pickle_module())


def build_model_from_pretrained(init_from: str, n_heads: int | None, rope_style: str | None):
    """-> (model, ModelArgs). init_from : preset ('large'/'kairo'), 'scratch', ou chemin .pt."""
    if init_from == "scratch":
        a = ModelArgs(**SCRATCH_ARGS, rope_style=rope_style or "half")
        return a, None
    if os.path.exists(init_from):
        path, preset = init_from, {"rope_style": "half"}
    else:
        preset = PRESETS[init_from]
        path = os.path.join(_SAVE_DIR, f"pretrained_{init_from}.pt")
        if not os.path.exists(path) and hf_download(preset["ckpt_repo"], preset["ckpt_file"], path) is None:
            raise RuntimeError(f"Checkpoint '{init_from}' introuvable. Passe --init-from /chemin/model.pt "
                               f"ou --init-from scratch.")
    ckpt = torch_load_safe(path)
    sd = _extract_state_dict(ckpt)
    # `args` sauvegardés avec le checkpoint (objet ou dict) : source fiable pour n_heads,
    # rope_theta, norm_eps, max_seq_len — mieux que de les deviner depuis les shapes.
    ck_args = (ckpt.get("args") or ckpt.get("model_args")) if isinstance(ckpt, dict) else None

    def g(name):
        return ck_args.get(name) if isinstance(ck_args, dict) else getattr(ck_args, name, None)

    nh = n_heads or (int(g("n_heads")) if ck_args is not None and g("n_heads") else None)
    a = infer_args_from_state_dict(sd, nh)
    if ck_args is not None:
        for k in ("rope_theta", "norm_eps", "max_seq_len"):
            v = g(k)
            if v:
                setattr(a, k, type(getattr(a, k))(v))
    a.rope_style = rope_style or preset["rope_style"]
    return a, sd


DETECT_SENTENCES_FR = [
    "La capitale de la France est Paris, une ville célèbre pour sa tour Eiffel et ses musées.",
    "Le chat dort tranquillement sur le canapé pendant que la pluie tombe sur les toits de la ville.",
    "Pour préparer une bonne soupe, il faut d'abord éplucher les légumes puis les faire cuire doucement.",
    "Hier soir, nous avons regardé un film avec nos amis avant de rentrer à la maison.",
]


def detect_rope_style(model: DetectorModel, tok, device) -> str:
    """Compare la loss du LM pré-entraîné sous les deux conventions RoPE et
    garde la meilleure : la mauvaise convention donne une loss nettement plus
    haute (le modèle « ne reconnaît plus » les positions). Résout le piège
    Large (rotate_half) vs Kaïro-CUDA (complexe entrelacé)."""
    ids = [tok.encode(s).ids[: model.args.max_seq_len] for s in DETECT_SENTENCES_FR]
    losses = {}
    model.eval()
    for style in ("half", "interleaved"):
        model.set_rope_style(style)
        ls = [model.lm_loss(torch.tensor([i], device=device)) for i in ids if len(i) > 4]
        losses[style] = float(np.mean(ls))
    best = min(losses, key=losses.get)
    print(f"🔎 Convention RoPE — loss LM : half={losses['half']:.3f} | interleaved={losses['interleaved']:.3f} "
          f"-> '{best}'")
    model.set_rope_style(best)
    return best


# ============================================================================
# Données — sources, nettoyage, chunking, split sans fuite
# ============================================================================
SRC_NAMES: list = []  # rempli par build_dataset (index -> nom de source)


def normalize_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def strip_markdown(t: str) -> str:
    """Retire le markdown « typique chatbot » (**gras**, # titres, puces) :
    sans ça, le détecteur apprendrait « il y a du markdown = IA », un raccourci
    qui ne tient plus dès que l'élève copie-colle sans mise en forme."""
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"^\s*[-*•]\s+", "", t, flags=re.M)
    return t.strip()


def default_sources(args) -> list:
    """Liste des sources. label 0 = humain, 1 = IA. `max_chunks` = plafond de
    fenêtres de `seq_len` tokens (multiplié par --scale)."""
    src = [
        # --- Humain -----------------------------------------------------------
        dict(name="wikipedia_fr", kind="hf_stream", dataset="wikimedia/wikipedia", config="20231101.fr",
             field="text", label=0, max_chunks=15000),
        # Pages web d'AVANT ChatGPT uniquement (champ `date` = date du crawl).
        dict(name="fineweb2_hq_fr_pre2023", kind="hf_stream", dataset="epfml/FineWeb2-HQ", config="fra_Latn",
             field="text", label=0, max_chunks=25000, max_date="2022-11-30"),
        # Humain « style assistant » : réponses/questions de vrais bénévoles.
        dict(name="oasst1_fr_human", kind="hf_rows", dataset="OpenAssistant/oasst1", field="text", label=0,
             max_chunks=4000,
             row_filter=lambda ex: ex.get("lang") == "fr" and not ex.get("deleted") and not ex.get("synthetic")),
        # --- IA ---------------------------------------------------------------
        dict(name="french_alpaca_gpt35", kind="hf_rows", dataset="jpacifico/French-Alpaca-dataset-Instruct-110K",
             field="output", label=1, max_chunks=30000, min_chars=200, strip_md_prob=0.5),
    ]
    for label, d, nm in ((0, args.human_dir, "local_human_dir"), (1, args.ai_dir, "local_ai_dir")):
        if d:
            src.append(dict(name=nm, kind="dir", path=d, label=label, max_chunks=10 ** 9, local=True))
    for fname, label, nm in (("mirrors.jsonl", 1, "mirrors_ai"), ("hard_negatives.jsonl", 0, "hard_negatives_human")):
        p = os.path.join(_DATA_DIR, fname)
        if os.path.exists(p):
            src.append(dict(name=nm, kind="jsonl", path=p, label=label, max_chunks=10 ** 9, local=True,
                            strip_md_prob=0.5 if label == 1 else 0.0))
    return src


def iter_source(spec: dict, skip: int = 0):
    """Générateur (doc_key, texte). `skip` saute les N premiers docs (utilisé par
    `mine` pour ne pas retomber sur les documents d'entraînement)."""
    kind = spec["kind"]
    if kind in ("hf_stream", "hf_rows"):
        from datasets import load_dataset
        stream = kind == "hf_stream"
        ds = load_dataset(spec["dataset"], name=spec.get("config"), split=spec.get("split", "train"),
                          streaming=stream)
        if skip and stream:
            ds = ds.skip(skip)
        elif skip and skip >= len(ds):
            print(f"ℹ️ [{spec['name']}] source trop petite pour skip={skip} : ignorée pour ce passage.")
            return
        flt, max_date = spec.get("row_filter"), spec.get("max_date")
        for i, ex in enumerate(ds):
            if not stream and i < skip:
                continue
            if max_date and (ex.get("date") or "9999") >= max_date:
                continue
            if flt and not flt(ex):
                continue
            t = ex.get(spec["field"])
            if t:
                yield f"{i}", t
    elif kind == "jsonl":
        with open(spec["path"], encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < skip:
                    continue
                try:
                    t = json.loads(line).get("text")
                except Exception:
                    continue
                if t:
                    yield f"{i}", t
    elif kind == "dir":
        files = sorted(glob.glob(os.path.join(spec["path"], "**", "*.txt"), recursive=True))
        for fp in files[skip:]:
            with open(fp, encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            # un fichier long = plusieurs documents (blocs de paragraphes ~1500 caractères)
            buf = ""
            for para in re.split(r"\n\s*\n", txt):
                buf += para + "\n\n"
                if len(buf) > 1500:
                    yield f"{fp}:{len(buf)}", buf
                    buf = ""
            if len(buf) > 200:
                yield f"{fp}:end", buf
    else:
        raise ValueError(kind)


def _is_val_doc(src_name: str, doc_key: str, val_pct: int) -> bool:
    """Split PAR DOCUMENT (hash stable) : deux fenêtres du même document ne se
    retrouvent jamais l'une en train, l'autre en val — sinon la validation est
    optimiste (fuite)."""
    h = hashlib.md5(f"{src_name}|{doc_key}".encode()).hexdigest()
    return int(h[:8], 16) % 100 < val_pct


def chunk_ids(ids: list, seq_len: int, min_tokens: int, max_chunks: int) -> list:
    out = []
    for s in range(0, len(ids), seq_len):
        c = ids[s:s + seq_len]
        if len(c) >= min_tokens:
            out.append(c)
        if len(out) >= max_chunks:
            break
    return out


def data_path(tok, args) -> str:
    return os.path.join(_DATA_DIR, f"detector_data_v{tok.get_vocab_size()}_L{args.seq_len}.npz")


def build_dataset(tok, args) -> str:
    """Construit puis sauvegarde le dataset tokenisé :
    X (N, L) uint16 | lens | y (0 humain, 1 IA) | src (index de source) | split (0 train, 1 val)."""
    if tok.get_vocab_size() > 65535:
        raise ValueError("vocab > 65535 : X doit passer en uint32.")
    global SRC_NAMES
    L = args.seq_len
    rng = random.Random(args.seed)
    sources = default_sources(args)
    SRC_NAMES = [s["name"] for s in sources]
    rows_X, rows_len, rows_y, rows_src, rows_split = [], [], [], [], []
    seen = set()
    for si, spec in enumerate(sources):
        budget = int(spec["max_chunks"] * args.scale) if not spec.get("local") else spec["max_chunks"]
        print(f"📚 [{spec['name']}] label={'IA' if spec['label'] else 'humain'} — objectif {budget} fenêtres...")
        n, ndocs, t0 = 0, 0, time.time()
        batch = []

        def flush(batch):
            nonlocal n
            if not batch:
                return
            keys, texts = zip(*batch)
            for key, enc in zip(keys, tok.encode_batch(list(texts))):
                for c in chunk_ids(enc.ids, L, args.min_tokens, args.max_chunks_per_doc):
                    rows_X.append(c)
                    rows_len.append(len(c))
                    rows_y.append(spec["label"])
                    rows_src.append(si)
                    rows_split.append(1 if _is_val_doc(spec["name"], key, args.val_pct) else 0)
                    n += 1

        try:
            for key, text in iter_source(spec):
                text = normalize_text(text)
                if len(text) < spec.get("min_chars", 100):
                    continue
                if spec.get("strip_md_prob") and rng.random() < spec["strip_md_prob"]:
                    text = strip_markdown(text)
                hsh = hashlib.md5(text[:300].encode()).hexdigest()
                if hsh in seen:
                    continue
                seen.add(hsh)
                batch.append((key, text[: args.max_chunks_per_doc * L * 6]))
                ndocs += 1
                if len(batch) >= 256:
                    flush(batch)
                    batch = []
                    if n >= budget:
                        break
                    if ndocs % 2560 == 0:
                        print(f"   … {ndocs} docs, {n} fenêtres ({time.time() - t0:.0f}s)")
            flush(batch)
        except Exception as e:
            print(f"   ⚠️ Source '{spec['name']}' interrompue ({type(e).__name__}: {e}) — "
                  f"on garde les {n} fenêtres déjà collectées.")
        print(f"   ✅ {n} fenêtres ({ndocs} docs, {time.time() - t0:.0f}s)")
    if not rows_X:
        raise RuntimeError("Dataset vide : aucune source n'a produit de texte.")
    N = len(rows_X)
    X = np.zeros((N, L), dtype=np.uint16)
    for i, c in enumerate(rows_X):
        X[i, : len(c)] = c
    y = np.array(rows_y, dtype=np.uint8)
    lens = np.array(rows_len, dtype=np.int16)
    out = data_path(tok, args)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, X=X, lens=lens, y=y, src=np.array(rows_src, dtype=np.uint8),
             split=np.array(rows_split, dtype=np.uint8), src_names=np.array(SRC_NAMES))
    for lab, nm in ((0, "humain"), (1, "IA")):
        m = y == lab
        print(f"📊 {nm} : {int(m.sum())} fenêtres, longueur moyenne {lens[m].mean():.0f} tokens")
    if min(int((y == 0).sum()), int((y == 1).sum())) < 200:
        print("⚠️ Une des deux classes a moins de 200 fenêtres — résultats peu fiables.")
    print(f"💾 Dataset sauvegardé : {out} ({os.path.getsize(out) / 1e6:.1f} Mo)")
    return out


class DetectorData:
    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.X, self.lens, self.y = z["X"], z["lens"].astype(np.int64), z["y"].astype(np.int64)
        self.src, self.split = z["src"].astype(np.int64), z["split"]
        self.src_names = [str(s) for s in z["src_names"]]
        self.L = self.X.shape[1]

    def idx(self, split: int, label: int | None = None):
        m = self.split == split
        if label is not None:
            m &= self.y == label
        return np.nonzero(m)[0]


class BatchSampler:
    """Échantillonne des lots ÉQUILIBRÉS (50/50 humain/IA) avec :
       - mélange (prob. mix_prob) : segment humain + segment IA collés, étiquettes PAR TOKEN ;
       - troncature aléatoire (prob. trunc_prob) : la longueur ne doit pas trahir la classe."""

    def __init__(self, data: DetectorData, split: int, mix_prob: float, trunc_prob: float,
                 min_tokens: int, pad_id: int, seed: int):
        self.d, self.mix_prob, self.trunc_prob = data, mix_prob, trunc_prob
        self.min_tokens, self.pad_id = min_tokens, pad_id
        self.h_idx, self.a_idx = data.idx(split, 0), data.idx(split, 1)
        if len(self.h_idx) == 0 or len(self.a_idx) == 0:
            raise RuntimeError("Il faut des exemples humains ET IA dans le split d'entraînement.")
        self.rng = np.random.default_rng(seed)

    def _pure(self, i):
        n = int(self.d.lens[i])
        if n > self.min_tokens and self.rng.random() < self.trunc_prob:
            n = int(self.rng.integers(self.min_tokens, n + 1))
        return self.d.X[i, :n].astype(np.int64), np.full(n, float(self.d.y[i]), dtype=np.float32)

    def _mixed(self):
        ih, ia = self.rng.choice(self.h_idx), self.rng.choice(self.a_idx)
        seg = []
        for i, lab in ((ih, 0.0), (ia, 1.0)):
            n = int(self.d.lens[i])
            lo = min(16, n)
            cut = int(self.rng.integers(lo, n + 1))
            seg.append((self.d.X[i, :cut].astype(np.int64), np.full(cut, lab, dtype=np.float32)))
        if self.rng.random() < 0.5:
            seg.reverse()
        ids = np.concatenate([s[0] for s in seg])[: self.d.L]
        lab = np.concatenate([s[1] for s in seg])[: self.d.L]
        return ids, lab

    def batch(self, B: int):
        items = []
        for _ in range(B):
            if self.rng.random() < self.mix_prob:
                items.append(self._mixed())
            else:
                pool = self.h_idx if self.rng.random() < 0.5 else self.a_idx
                items.append(self._pure(int(self.rng.choice(pool))))
        T = max(len(it[0]) for it in items)
        tokens = np.full((B, T), self.pad_id, dtype=np.int64)
        labels = np.zeros((B, T), dtype=np.float32)
        mask = np.zeros((B, T), dtype=bool)
        for b, (ids, lab) in enumerate(items):
            tokens[b, : len(ids)], labels[b, : len(ids)], mask[b, : len(ids)] = ids, lab, True
        return torch.from_numpy(tokens), torch.from_numpy(labels), torch.from_numpy(mask)


# ============================================================================
# Perte, pooling et métriques
# ============================================================================
WARM_TOKENS = 8  # les premiers tokens n'ont quasi aucun contexte : ignorés du score


def pool_logits(tl, lens, warm: int = WARM_TOKENS):
    """Score du document = moyenne des logits par token (hors `warm` premiers
    tokens, sauf pour les très courts documents où l'on garde tout)."""
    T = tl.shape[1]
    pos = torch.arange(T, device=tl.device)[None, :]
    m = (pos < lens[:, None]).float()
    w = torch.where((lens > warm + 16)[:, None], m * (pos >= warm).float(), m)
    return (tl * w).sum(1) / w.sum(1).clamp(min=1)


def detector_loss(tl, labels, mask, aux_weight: float, warm: int = WARM_TOKENS):
    lens = mask.sum(1)
    m = mask.float()
    doc_logit = pool_logits(tl, lens, warm)
    doc_target = (labels * m).sum(1) / m.sum(1).clamp(min=1)  # soft pour les exemples mixtes
    doc_loss = F.binary_cross_entropy_with_logits(doc_logit, doc_target)
    pos = torch.arange(tl.shape[1], device=tl.device)[None, :]
    w = m * (pos >= warm).float()
    tok = F.binary_cross_entropy_with_logits(tl, labels, reduction="none")
    tok_loss = (tok * w).sum() / w.sum().clamp(min=1)
    return doc_loss + aux_weight * tok_loss, doc_loss.detach(), tok_loss.detach()


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]  # rangs moyens (ex æquo)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def tpr_at_fpr(scores: np.ndarray, y: np.ndarray, fpr: float = 0.01):
    h, a = scores[y == 0], scores[y == 1]
    if len(h) == 0 or len(a) == 0:
        return float("nan"), float("nan")
    thr = float(np.quantile(h, 1 - fpr))
    return float((a > thr).mean()), thr


def calibrate(logits: np.ndarray, y: np.ndarray, fpr: float = 0.01, fnr: float = 0.05) -> dict:
    """Deux seuils (en LOGIT) estimés sur la validation :
       thr_ia     : au plus `fpr` des textes humains le dépassent -> verdict « ia » ;
       thr_humain : au plus `fnr` des textes IA sont en dessous -> verdict « humain » ;
       entre les deux : « incertain »."""
    n_h = int((y == 0).sum())
    if n_h < 10 / fpr:  # < 1000 humains pour 1 % : moins de ~10 points dans la queue, quantile très bruité
        print(f"⚠️ Calibration sur seulement {n_h} textes humains de validation : le seuil à "
              f"{fpr:.0%} de faux positifs est approximatif (vise ≥ {int(10 / fpr)} pour un seuil fiable).")
    thr_ia = float(np.quantile(logits[y == 0], 1 - fpr))
    thr_h = min(float(np.quantile(logits[y == 1], fnr)), thr_ia)
    return {"thr_ia": thr_ia, "thr_humain": thr_h, "target_fpr": fpr, "target_fnr": fnr}


@torch.no_grad()
def score_val(model, data: DetectorData, split: int, pad_id: int, bs: int = 32):
    model.eval()
    idx = data.idx(split)
    out = np.zeros(len(idx), dtype=np.float64)
    for s in range(0, len(idx), bs):
        ii = idx[s:s + bs]
        T = int(data.lens[ii].max())
        tokens = torch.from_numpy(data.X[ii, :T].astype(np.int64)).to(DEVICE)
        lens = torch.from_numpy(data.lens[ii]).to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP_DTYPE is not None):
            tl = model(tokens)
        out[s:s + bs] = pool_logits(tl.float(), lens).cpu().numpy()
    model.train()
    return idx, out


def evaluate(model, data: DetectorData, pad_id: int, split: int = 1, verbose: bool = True) -> dict:
    idx, logits = score_val(model, data, split, pad_id)
    y = data.y[idx]
    if len(idx) == 0:
        return {}
    p = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    loss = float(-(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)).mean())
    tpr, thr = tpr_at_fpr(logits, y, 0.01)
    res = {"loss": loss, "acc": float(((logits > 0) == (y == 1)).mean()), "auroc": auroc(logits, y),
           "tpr_at_1pct_fpr": tpr, "n": int(len(idx))}
    if verbose:
        print(f"   📈 val : loss={loss:.4f} acc={res['acc']:.3f} AUROC={res['auroc']:.4f} "
              f"TPR@1%FPR={tpr:.3f} (n={len(idx)})")
        for si, nm in enumerate(data.src_names):
            m = data.src[idx] == si
            if m.sum() >= 20:
                acc = ((logits[m] > thr) == (y[m] == 1)).mean()
                print(f"      · {nm:<26} n={int(m.sum()):>5}  bien classés (seuil 1% FPR) = {acc:.3f}")
    return res


# ============================================================================
# Entraînement
# ============================================================================
def get_lr(step: int, total: int, warmup: int, lr: float, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    r = (step - warmup) / max(1, total - warmup)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, r))))


def configure_optimizer(model, lr: float, head_lr: float, wd: float):
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        head = n.startswith("cls_head")
        decay = p.dim() >= 2 and not head
        groups.setdefault((head, decay), []).append(p)
    pg = [{"params": ps, "weight_decay": wd if decay else 0.0, "lr": head_lr if head else lr,
           "base_lr": head_lr if head else lr} for (head, decay), ps in groups.items()]
    fused = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames and ACCEL_TYPE == "cuda"
    return torch.optim.AdamW(pg, betas=(0.9, 0.95), eps=1e-8, fused=fused)


def save_checkpoint(path: str, model, optimizer, step: int, margs: ModelArgs, extra: dict) -> None:
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
                "step": step, "model_args": asdict(margs), **extra}, tmp)
    os.replace(tmp, path)  # écriture atomique : jamais de checkpoint à moitié écrit


def run_training(args) -> None:
    set_seed(args.seed)
    tok = load_tokenizer(args.tokenizer or args.init_from)
    pad_id = special_ids(tok)["pad"]
    dpath = data_path(tok, args)
    if not os.path.exists(dpath):
        build_dataset(tok, args)
    data = DetectorData(dpath)
    print(f"📦 Dataset : {len(data.y)} fenêtres (train={int((data.split == 0).sum())}, "
          f"val={int((data.split == 1).sum())})")

    resume = None
    if not args.fresh:
        if not os.path.exists(LATEST_PATH) and HF_BACKUP_ENABLED:
            hf_download(HF_BACKUP_REPO_ID, f"{CKPT_PREFIX}latest.pt", LATEST_PATH)
        if os.path.exists(LATEST_PATH):
            resume = torch_load_safe(LATEST_PATH)
            print(f"♻️  Reprise depuis {LATEST_PATH} (step {resume['step']})")

    pre_sd = None
    if resume is not None:
        margs = ModelArgs(**resume["model_args"])
    else:
        margs, pre_sd = build_model_from_pretrained(args.init_from, args.n_heads, args.rope_style)
    margs.vocab_size = tok.get_vocab_size() if args.init_from == "scratch" else margs.vocab_size
    if margs.vocab_size != tok.get_vocab_size():
        raise RuntimeError(f"Tokenizer ({tok.get_vocab_size()}) ≠ modèle ({margs.vocab_size}) : "
                           f"mauvais couple tokenizer/checkpoint (large=16k, kairo=32k).")
    if args.seq_len > margs.max_seq_len:
        raise RuntimeError(f"--seq-len {args.seq_len} > max_seq_len du modèle ({margs.max_seq_len}).")
    margs.grad_checkpointing = not args.no_grad_ckpt
    model = DetectorModel(margs)

    if resume is not None:
        model.load_state_dict(resume["model"])
    elif pre_sd is not None:
        missing, unexpected = model.load_state_dict(pre_sd, strict=False)
        miss = [k for k in missing if not k.startswith("cls_head")]
        print(f"✅ Poids pré-entraînés chargés ({args.init_from}) — manquants hors tête: {miss or 'aucun'} "
              f"| inattendus: {unexpected or 'aucun'}")
        if miss:
            raise RuntimeError("Des poids du backbone manquent : checkpoint incompatible avec cette architecture.")
        if args.rope_style is None and args.init_from != "scratch":
            model.to(DEVICE)
            detect_rope_style(model, tok, DEVICE)
    model.to(DEVICE).train()
    print(f"🧠 {count_parameters(model) / 1e6:.1f}M paramètres — dim={margs.dim}, couches={margs.n_layers}, "
          f"RoPE={margs.rope_style}, device={ACCEL_TYPE}, amp={AMP_DTYPE}")

    lr = args.lr if args.lr else (3e-4 if args.init_from == "scratch" else 5e-5)
    opt = configure_optimizer(model, lr, lr * args.head_lr_mult, args.weight_decay)
    start_step, best_key, calib = 0, (-1.0, -1e9), None
    if resume is not None:
        start_step = int(resume["step"])
        best_key = tuple(resume.get("best_key", best_key))
        if resume.get("optimizer"):
            try:
                opt.load_state_dict(resume["optimizer"])
            except Exception as e:
                print(f"⚠️ État de l'optimiseur non restauré ({e}) — momentum Adam perdu, poids conservés.")
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16))
    sampler = BatchSampler(data, 0, args.mix_prob, args.trunc_prob, args.min_tokens, pad_id, args.seed + start_step)

    total = args.steps
    print(f"🚀 Entraînement : {total} steps × batch {args.batch_size} × accum {args.grad_accum} "
          f"(lr={lr:.1e}, tête={lr * args.head_lr_mult:.1e})")
    t0, last_push, run_loss, run_n = time.time(), time.time(), 0.0, 0
    try:
        for step in range(start_step, total):
            cur = get_lr(step, total, args.warmup, 1.0)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * cur
            for _ in range(args.grad_accum):
                tokens, labels, mask = sampler.batch(args.batch_size)
                tokens, labels, mask = tokens.to(DEVICE), labels.to(DEVICE), mask.to(DEVICE)
                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP_DTYPE is not None):
                    tl = model(tokens)
                loss, dl, tl_loss = detector_loss(tl.float(), labels, mask, args.aux_weight)
                scaler.scale(loss / args.grad_accum).backward()
                run_loss += float(loss.detach())
                run_n += 1
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            s1 = step + 1
            if s1 % args.log_every == 0:
                print(f"step {s1}/{total} | loss {run_loss / max(1, run_n):.4f} | "
                      f"lr {opt.param_groups[0]['lr']:.2e} | {(time.time() - t0) / (s1 - start_step):.2f}s/step")
                run_loss, run_n = 0.0, 0
            if s1 % args.eval_every == 0 or s1 == total:
                m = evaluate(model, data, pad_id)
                key = (m.get("tpr_at_1pct_fpr", 0.0), -m.get("loss", 1e9))
                extra = {"best_key": list(max(best_key, key)), "tokenizer": args.tokenizer or args.init_from,
                         "seq_len": args.seq_len, "metrics": m}
                if key > best_key:
                    best_key = key
                    idx, lg = score_val(model, data, 1, pad_id)
                    calib = calibrate(lg, data.y[idx])
                    extra["calib"] = calib
                    save_checkpoint(BEST_PATH, model, None, s1, margs, extra)
                    print(f"   ⭐ Nouveau meilleur modèle (seuils : {calib})")
                    push_backup_to_hf(BEST_PATH, f"{CKPT_PREFIX}best.pt")
                save_checkpoint(LATEST_PATH, model, opt, s1, margs, extra)
                if time.time() - last_push > HF_LATEST_INTERVAL_SECONDS:
                    push_backup_to_hf(LATEST_PATH, f"{CKPT_PREFIX}latest.pt")
                    last_push = time.time()
    except torch.cuda.OutOfMemoryError:
        print("💥 OOM CUDA : relance avec --batch-size plus petit (et --grad-accum plus grand), "
              "ou --seq-len 192. Le dernier checkpoint 'latest' est conservé, la reprise est automatique.")
        raise
    print(f"🏁 Terminé. Meilleur : {BEST_PATH}")


# ============================================================================
# Inférence — Detector (scores, verdict, surlignage phrase par phrase)
# ============================================================================
def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


class Detector:
    def __init__(self, model: DetectorModel, tok, calib: dict, seq_len: int, device=None):
        self.model, self.tok, self.calib, self.seq_len = model.eval(), tok, calib, seq_len
        self.device = device or next(model.parameters()).device
        self.pad_id = special_ids(tok)["pad"]

    @classmethod
    def from_checkpoint(cls, path: str | None = None, tokenizer: str | None = None, device=None):
        path = path or BEST_PATH
        if not os.path.exists(path) and HF_BACKUP_ENABLED:
            hf_download(HF_BACKUP_REPO_ID, f"{CKPT_PREFIX}best.pt", path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Aucun checkpoint de détecteur ({path}). Lance d'abord --stage train.")
        ck = torch_load_safe(path)
        margs = ModelArgs(**ck["model_args"])
        margs.grad_checkpointing = False
        margs.dropout = 0.0
        model = DetectorModel(margs)
        model.load_state_dict(ck["model"])
        model.to(device or DEVICE).eval()
        tok = load_tokenizer(tokenizer or ck.get("tokenizer", "large"))
        calib = ck.get("calib") or {"thr_ia": 0.0, "thr_humain": 0.0}
        return cls(model, tok, calib, int(ck.get("seq_len", 256)), device or DEVICE)

    @torch.no_grad()
    def token_logits(self, ids: list) -> np.ndarray:
        """Logit par token. Fenêtres glissantes à 50 % de recouvrement : chaque
        token (hors début du texte) est noté avec au moins une demi-fenêtre de contexte."""
        L, n = self.seq_len, len(ids)
        out = np.zeros(n, dtype=np.float64)
        stride = max(1, L // 2)
        written, s = 0, 0
        while written < n:
            w = ids[s:s + L]
            x = torch.tensor([w], dtype=torch.long, device=self.device)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP_DTYPE is not None):
                tl = self.model(x)[0].float().cpu().numpy()
            start = written - s  # on n'écrit que les tokens pas encore notés
            out[written:s + len(w)] = tl[start:]
            written = s + len(w)
            s += stride
        return out

    def predict(self, text: str, segments: bool = True) -> dict:
        text = normalize_text(text)
        enc = self.tok.encode(text)
        ids, offsets = enc.ids, enc.offsets
        n = len(ids)
        if n < 24:
            return {"verdict": "trop_court", "ai_probability": None, "n_tokens": n,
                    "message": "Texte trop court pour un score fiable (< 24 tokens)."}
        tl = self.token_logits(ids)
        warm = WARM_TOKENS if n > WARM_TOKENS + 16 else 0
        doc_logit = float(tl[warm:].mean())
        thr_ia, thr_h = self.calib["thr_ia"], self.calib["thr_humain"]
        segs = []
        if segments:
            starts = np.array([o[0] for o in offsets])
            for m in re.finditer(r"[^.!?…\n]+[.!?…]*\s*", text):
                a, b = m.start(), m.end()
                sel = np.nonzero((starts >= a) & (starts < b))[0]
                sel = sel[sel >= warm] if (sel >= warm).any() else sel
                if len(sel) == 0 or not text[a:b].strip():
                    continue
                lg = float(tl[sel].mean())
                segs.append({"text": text[a:b].strip(), "start": a, "end": b, "n_tokens": int(len(sel)),
                             "ai_probability": round(_sigmoid(lg), 4), "flagged": lg >= thr_ia})
        tot = sum(s["n_tokens"] for s in segs)
        frac = sum(s["n_tokens"] for s in segs if s["flagged"]) / tot if tot else float(doc_logit >= thr_ia)
        # Verdict (heuristique, à relire avec les segments) :
        #   ia        : score global au-dessus du seuil 1 % FPR ET >= 80 % du texte surligné ;
        #   mixte     : entre 20 % et 80 % du texte surligné (passages IA dans un texte humain) ;
        #   humain    : score global sous le seuil bas ET < 20 % surligné ;
        #   incertain : le reste (zone grise entre les deux seuils).
        if frac >= 0.8 and doc_logit >= thr_ia:
            verdict = "ia"
        elif 0.2 <= frac < 0.8:
            verdict = "mixte"
        elif frac < 0.2 and doc_logit <= thr_h:
            verdict = "humain"
        else:
            verdict = "incertain"
        res = {"verdict": verdict, "ai_probability": round(_sigmoid(doc_logit), 4),
               "ai_fraction": round(frac, 3), "n_tokens": n}
        if segments:
            res["segments"] = segs
        return res


def create_detector_blueprint(detector: Detector, url_prefix: str = "/detect"):
    """Blueprint Flask dans le même esprit que /rag /reason /tools de l'API
    Opsiom : POST /detect {"text": "..."} -> JSON. À monter avec
    app.register_blueprint(create_detector_blueprint(det))."""
    from flask import Blueprint, jsonify, request
    bp = Blueprint("detector", __name__, url_prefix=url_prefix)

    @bp.get("/health")
    def health():
        return jsonify({"status": "ok", "seq_len": detector.seq_len, "calib": detector.calib})

    @bp.post("")
    @bp.post("/")
    def detect():
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or "").strip()
        if not text:
            return jsonify({"error": "champ 'text' manquant"}), 400
        if len(text) > 50000:
            return jsonify({"error": "texte trop long (max 50 000 caractères)"}), 413
        return jsonify(detector.predict(text, segments=bool(payload.get("segments", True))))

    return bp


# ============================================================================
# Miroirs synthétiques (étape optionnelle `generate`) et hard negatives (`mine`)
# ============================================================================
MIRROR_TEMPLATES = {
    "wikipedia_fr": ["Rédige un article encyclopédique d'environ {n} mots sur le sujet suivant : « {topic} ». "
                     "Ne recopie pas la phrase d'amorce. Réponds uniquement avec le texte.",
                     "Écris un paragraphe de type Wikipédia (environ {n} mots) à propos de : « {topic} »."],
    "fineweb2_hq_fr_pre2023": ["Écris un article de blog d'environ {n} mots qui traite de : « {topic} ».",
                               "Rédige un texte informatif pour un site web, d'environ {n} mots, sur ce thème : « {topic} »."],
    "oasst1_fr_human": ["Réponds en français, comme sur un forum d'entraide, à ceci : « {topic} ». "
                        "Environ {n} mots."],
    "default": ["Écris un texte d'environ {n} mots inspiré de cet extrait : « {topic} »."],
}


def make_mirror_prompt(text: str, source: str, rng: random.Random):
    """Le « miroir » : sujet = amorce du texte humain, longueur ≈ celle du texte humain."""
    topic = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0][:200].replace("\n", " ")
    n_words = int(min(260, max(60, len(text.split()) * rng.uniform(0.6, 1.0))))
    tpl = rng.choice(MIRROR_TEMPLATES.get(source, MIRROR_TEMPLATES["default"]))
    return tpl.format(n=n_words, topic=topic)


def generate_mirrors(args) -> None:
    """Génère des textes IA « miroirs » de textes humains avec un ou plusieurs
    LLM instruct (--gen-models a,b,c — varier les générateurs améliore la
    généralisation). Écrit/complète mirrors.jsonl dans le dossier de données ;
    relancer le stage `build` ensuite pour l'intégrer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out_path = os.path.join(_DATA_DIR, "mirrors.jsonl")
    done = sum(1 for _ in open(out_path, encoding="utf-8")) if os.path.exists(out_path) else 0
    if done >= args.gen_n:
        print(f"✅ {done} miroirs déjà présents ({out_path}).")
        return
    rng = random.Random(args.seed + done)
    seeds = []
    human = [s for s in default_sources(args) if s["label"] == 0 and not s.get("local")]
    per = max(1, (args.gen_n - done) // max(1, len(human)) + 1)
    for spec in human:
        k = 0
        try:
            for _, t in iter_source(spec, skip=args.gen_skip):
                t = normalize_text(t)
                if len(t) >= 300:
                    seeds.append((spec["name"], t))
                    k += 1
                if k >= per:
                    break
        except Exception as e:
            print(f"⚠️ Graine '{spec['name']}' ignorée ({type(e).__name__}: {e})")
    rng.shuffle(seeds)
    models = [m.strip() for m in args.gen_models.split(",") if m.strip()]
    todo = seeds[: args.gen_n - done]
    print(f"✍️  Génération de {len(todo)} miroirs avec {models}...")
    with open(out_path, "a", encoding="utf-8") as fout:
        per_model = math.ceil(len(todo) / len(models))
        for mi, mname in enumerate(models):
            part = todo[mi * per_model:(mi + 1) * per_model]
            if not part:
                continue
            tk = AutoTokenizer.from_pretrained(mname)
            tk.padding_side = "left"
            if tk.pad_token is None:
                tk.pad_token = tk.eos_token
            dt = AMP_DTYPE or torch.float32
            try:  # `dtype` (transformers récents) ; `torch_dtype` pour les anciennes versions
                lm = AutoModelForCausalLM.from_pretrained(mname, dtype=dt)
            except TypeError:
                lm = AutoModelForCausalLM.from_pretrained(mname, torch_dtype=dt)
            lm = lm.to(DEVICE).eval()
            for s in range(0, len(part), args.gen_batch):
                chunk = part[s:s + args.gen_batch]
                prompts = [make_mirror_prompt(t, src, rng) for src, t in chunk]
                if getattr(tk, "chat_template", None):
                    inputs = [tk.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                     add_generation_prompt=True) for p in prompts]
                else:
                    inputs = [p + "\n\n" for p in prompts]
                enc = tk(inputs, return_tensors="pt", padding=True).to(DEVICE)
                temp = rng.uniform(0.6, 1.0)
                with torch.no_grad():
                    gen = lm.generate(**enc, max_new_tokens=args.gen_max_new, do_sample=True, temperature=temp,
                                      top_p=0.95, pad_token_id=tk.pad_token_id)
                texts = tk.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for (src, _), p, t in zip(chunk, prompts, texts):
                    t = normalize_text(t)
                    if len(t) >= 150:
                        fout.write(json.dumps({"text": t, "model": mname, "source_hint": src, "prompt": p,
                                               "temperature": round(temp, 2)}, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"   {mname}: {s + len(chunk)}/{len(part)}")
            del lm
            if ACCEL_TYPE == "cuda":
                torch.cuda.empty_cache()
    print(f"💾 Miroirs écrits dans {out_path} — relance `--stage build` (supprime l'ancien .npz) puis `--stage train`.")


def mine_false_positives(det: Detector, args) -> None:
    """Hard negative mining : score du texte humain FRAIS (documents non vus à
    l'entraînement) et garde ceux que le détecteur accuse à tort. Résultat dans
    hard_negatives.jsonl (label humain), intégré au prochain `build`.
    ⚠️ Ne mine que des sources dont l'origine humaine est garantie
    (Wikipédia, FineWeb2-HQ pré-2023, OpenAssistant) : sur du web récent, un
    « faux positif » est parfois un vrai texte IA — l'ajouter en « humain »
    empoisonnerait le dataset."""
    out_path = os.path.join(_DATA_DIR, "hard_negatives.jsonl")
    thr = det.calib["thr_ia"]
    kept, scanned = 0, 0
    with open(out_path, "a", encoding="utf-8") as f:
        for spec in [s for s in default_sources(args) if s["label"] == 0 and not s.get("local")]:
            for _, t in iter_source(spec, skip=args.mine_skip):
                t = normalize_text(t)
                if len(t) < 300:
                    continue
                r = det.predict(t[: det.seq_len * 5], segments=False)
                scanned += 1
                if r.get("ai_probability") is not None and math.log(
                        max(r["ai_probability"], 1e-9) / max(1 - r["ai_probability"], 1e-9)) >= thr:
                    f.write(json.dumps({"text": t, "source": spec["name"], "ai_probability": r["ai_probability"]},
                                       ensure_ascii=False) + "\n")
                    kept += 1
                if scanned >= args.mine_docs:
                    break
            print(f"   [{spec['name']}] scannés={scanned} faux positifs gardés={kept}")
            scanned = 0
    print(f"💾 {kept} hard negatives -> {out_path}. Relance `--stage build` puis `--stage train --fresh`.")


# ============================================================================
# CLI
# ============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Opsiom Détecteur — texte IA vs humain (FR)")
    p.add_argument("--stage", default="train",
                   choices=["build", "generate", "train", "all", "eval", "predict", "mine", "serve", "inspect"])
    p.add_argument("--init-from", default="large", help="large | kairo | scratch | /chemin/model.pt")
    p.add_argument("--tokenizer", default=None, help="chemin tokenizer.json (défaut : celui du preset)")
    p.add_argument("--n-heads", type=int, default=None, help="si ≠ dim/64")
    p.add_argument("--rope-style", default=None, choices=[None, "half", "interleaved"],
                   help="défaut : auto-détecté en comparant la loss LM")
    # données
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--min-tokens", type=int, default=32)
    p.add_argument("--max-chunks-per-doc", type=int, default=2)
    p.add_argument("--scale", type=float, default=1.0, help="multiplie la taille des sources")
    p.add_argument("--val-pct", type=int, default=5)
    p.add_argument("--human-dir", default=None, help="dossier de .txt humains (ta prose, etc.)")
    p.add_argument("--ai-dir", default=None, help="dossier de .txt générés par IA")
    # entraînement
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--head-lr-mult", type=float, default=20.0)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--mix-prob", type=float, default=0.15)
    p.add_argument("--trunc-prob", type=float, default=0.5)
    p.add_argument("--aux-weight", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--no-grad-ckpt", action="store_true")
    p.add_argument("--fresh", action="store_true", help="ignore le checkpoint 'latest'")
    p.add_argument("--seed", type=int, default=42)
    # inférence
    p.add_argument("--text", default=None)
    p.add_argument("--file", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--port", type=int, default=5055)
    # miroirs / mining
    p.add_argument("--gen-models", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--gen-n", type=int, default=3000)
    p.add_argument("--gen-batch", type=int, default=8)
    p.add_argument("--gen-max-new", type=int, default=300)
    p.add_argument("--gen-skip", type=int, default=0)
    p.add_argument("--mine-docs", type=int, default=5000)
    p.add_argument("--mine-skip", type=int, default=200000)
    # parse_known_args : tolère les arguments parasites injectés par Jupyter/Kaggle
    args, _ = p.parse_known_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    if args.stage in ("build", "all"):
        build_dataset(load_tokenizer(args.tokenizer or args.init_from), args)
    if args.stage == "inspect":
        path = args.init_from if os.path.exists(args.init_from) else os.path.join(_SAVE_DIR, f"pretrained_{args.init_from}.pt")
        if not os.path.exists(path):
            pr = PRESETS[args.init_from]
            hf_download(pr["ckpt_repo"], pr["ckpt_file"], path)
        print(describe_checkpoint(torch_load_safe(path)))
    if args.stage == "generate":
        generate_mirrors(args)
    if args.stage in ("train", "all"):
        run_training(args)
    if args.stage in ("eval", "predict", "mine", "serve"):
        det = Detector.from_checkpoint(args.checkpoint, args.tokenizer)
        if args.stage == "eval":
            data = DetectorData(data_path(det.tok, args))
            evaluate(det.model, data, det.pad_id)
        elif args.stage == "predict":
            text = args.text or (open(args.file, encoding="utf-8").read() if args.file else None)
            if not text:
                raise SystemExit("Donne --text \"...\" ou --file chemin.txt")
            print(json.dumps(det.predict(text), ensure_ascii=False, indent=2))
        elif args.stage == "mine":
            mine_false_positives(det, args)
        else:
            from flask import Flask
            app = Flask(__name__)
            app.register_blueprint(create_detector_blueprint(det))
            app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()