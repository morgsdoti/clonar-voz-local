#!/usr/bin/env python3
"""Transcreva uma referência com faster-whisper, localmente na CPU."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys
import time

from audio_utils import load_reference


def transcribe(reference: str | Path, language: str = "pt", model_name: str = "small") -> str:
    # O mesmo limite e arquivo usado na clonagem: nunca recortamos silenciosamente.
    load_reference(reference)
    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_name, device="cpu", compute_type="int8",
        cpu_threads=min(8, os.cpu_count() or 1),
        download_root=os.environ.get("HF_HUB_CACHE"),
        local_files_only=os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"},
    )
    try:
        segments, _ = model.transcribe(str(reference), language=language, beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        del model
        gc.collect()
    if not text:
        raise ValueError("não reconheci fala; confira o áudio ou use --ref-text")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, type=Path, help="WAV/FLAC de até 20 s")
    parser.add_argument("--lang", default="pt", help="idioma da referência, padrão pt")
    parser.add_argument("--model", default="small", help="modelo faster-whisper ou diretório local")
    parser.add_argument("--out", type=Path, help="salvar transcrição UTF-8; não substitui arquivo existente")
    args = parser.parse_args()
    start = time.perf_counter()
    try:
        if args.out and args.out.exists():
            raise FileExistsError("o arquivo de transcrição já existe; escolha outro nome")
        text = transcribe(args.ref, args.lang, args.model)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as target:
                target.write(text + "\n")
        print(text)
        print(f"transcrição local: {time.perf_counter() - start:.1f} s; revise o texto antes de clonar", file=sys.stderr)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
