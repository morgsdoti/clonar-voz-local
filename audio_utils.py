"""Utilitários locais; nenhum áudio é enviado a serviços externos."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import soundfile as sf


def load_reference(path: str | Path) -> tuple[np.ndarray, int]:
    """Leia WAV/FLAC, misture canais e valide antes de carregar modelos."""
    info = sf.info(str(path))
    if not 0 < info.duration <= 20:
        raise ValueError("a referência precisa ter mais de zero e no máximo 20 segundos; recorte e ajuste a transcrição")
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if not np.isfinite(mono).all() or np.max(np.abs(mono)) < 1e-5:
        raise ValueError("a referência está vazia, silenciosa ou contém valores inválidos")
    return mono, rate


def split_text(text: str, max_chars: int = 240) -> list[str]:
    """Prefira frases; corte frases enormes em palavras, sem perder conteúdo."""
    if max_chars < 20:
        raise ValueError("--max-chars deve ser pelo menos 20")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("o texto para narrar está vazio")
    sentences = re.split(r"(?<=[.!?;:])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        for word in sentence.split():
            # Uma palavra/URL pode exceder o limite, mesmo sem espaços.
            while len(word) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(word[:max_chars])
                word = word[max_chars:]
            if not word:
                continue
            if current and len(current) + 1 + len(word) > max_chars:
                chunks.append(current)
                current = ""
            current = f"{current} {word}".strip()
        # Preserve limites de frases em vez de partir no meio sem necessidade.
        if current:
            chunks.append(current)
            current = ""
    return chunks


def as_mono(audio) -> np.ndarray:
    """Aceita a saída numpy atual e tensores de versões anteriores."""
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("o modelo retornou áudio vazio ou inválido")
    return np.clip(audio, -1.0, 1.0)


def check_output(path: Path, inputs: list[Path]) -> None:
    if path.suffix.lower() != ".wav":
        raise ValueError("use um arquivo de saída com extensão .wav")
    if any(path.resolve() == source.resolve() for source in inputs):
        raise ValueError("a saída não pode substituir a referência ou o texto de entrada")
    if path.exists():
        raise FileExistsError(f"a saída já existe: {path}; escolha outro nome")
