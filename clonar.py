#!/usr/bin/env python3
"""Clone somente sua própria voz ou uma voz com autorização explícita."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf

from audio_utils import as_mono, check_output, load_reference, split_text


def select_device(requested: str) -> str:
    import torch
    if requested == "cpu":
        return "cpu"
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return "cuda:0"
    if requested in {"auto", "mps"} and torch.backends.mps.is_available():
        return "mps"
    if requested != "auto":
        print(f"{requested} indisponível; usando CPU", file=sys.stderr)
    return "cpu"


def release_accelerator() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def synthesize(args, reference, transcript: str, chunks: list[str], device: str):
    import torch
    from omnivoice.models.omnivoice import OmniVoice

    torch.set_num_threads(args.threads)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = None
    try:
        print(f"carregando {args.model} em {device} ({dtype})", file=sys.stderr)
        model = OmniVoice.from_pretrained(args.model, device_map=device, dtype=dtype)
        rate = int(model.sampling_rate)
        outputs = []
        with torch.inference_mode():
            # Só fica em memória: não salve tokens/embeddings da voz no repo.
            prompt = model.create_voice_clone_prompt(ref_audio=reference, ref_text=transcript)
            for index, chunk in enumerate(chunks, 1):
                print(f"gerando trecho {index}/{len(chunks)}", file=sys.stderr)
                if outputs and args.pause > 0:
                    outputs.append(np.zeros(round(args.pause * rate), dtype=np.float32))
                audio = model.generate(
                    text=chunk, language=args.lang, voice_clone_prompt=prompt,
                    num_step=args.steps,
                )
                outputs.append(as_mono(audio[0]))
        return np.concatenate(outputs), rate
    finally:
        del model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, type=Path, help="referência WAV/FLAC de até 20 s")
    refs = parser.add_mutually_exclusive_group()
    refs.add_argument("--ref-text", help="transcrição exata da referência; omita para transcrever automaticamente")
    refs.add_argument("--ref-text-file", type=Path, help="transcrição revisada em arquivo UTF-8")
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("--texto", help="texto a narrar")
    text.add_argument("--arquivo", type=Path, help="texto UTF-8 a narrar")
    parser.add_argument("--lang", default="pt", help="idioma da saída, padrão pt")
    parser.add_argument("--ref-lang", default=None, help="idioma da referência para ASR; padrão igual a --lang")
    parser.add_argument("--out", type=Path, default=Path("saida.wav"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--model", default="k2-fsa/OmniVoice", help="identificador HF ou diretório local")
    parser.add_argument("--asr-model", default="small")
    parser.add_argument("--max-chars", type=int, default=240)
    parser.add_argument("--pause", type=float, default=0.15, help="silêncio entre trechos, em segundos")
    parser.add_argument("--steps", type=int, default=32, help="passos de geração; 16 é mais rápido, pode perder qualidade")
    parser.add_argument("--threads", type=int, default=4, help="threads CPU; evite oversubscription")
    parser.add_argument("--consentimento", action="store_true", required=True,
                        help="confirmo que a voz é minha ou tenho autorização explícita")
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        if not 1 <= args.threads <= 128 or not 1 <= args.steps <= 128 or not 0 <= args.pause <= 5:
            raise ValueError("threads/passos devem estar entre 1 e 128 e pausa entre 0 e 5 s")
        inputs = [args.ref] + [p for p in (args.arquivo, args.ref_text_file) if p]
        check_output(args.out, inputs)
        reference = load_reference(args.ref)
        target = args.arquivo.read_text(encoding="utf-8-sig") if args.arquivo else args.texto
        chunks = split_text(target, args.max_chars)
        transcript = args.ref_text
        if args.ref_text_file:
            transcript = args.ref_text_file.read_text(encoding="utf-8-sig")
        if transcript is None:
            from transcrever import transcribe
            print("transcrevendo referência localmente; a transcrição automática pode errar", file=sys.stderr)
            transcript = transcribe(args.ref, args.ref_lang or args.lang, args.asr_model)
            print("para revisar antes, rode transcrever.py e depois --ref-text-file", file=sys.stderr)
        transcript = transcript.strip()
        if not transcript:
            raise ValueError("a transcrição da referência está vazia")
        device = select_device(args.device)
        try:
            audio, rate = synthesize(args, reference, transcript, chunks, device)
        except (RuntimeError, NotImplementedError) as exc:
            if device == "cpu":
                raise
            # Não tente contornar downloads/auth/licença: só incompatibilidades do backend.
            backend_error = any(term in str(exc).lower() for term in
                                ("cuda", "mps", "out of memory", "not implemented", "not supported", "float16"))
            if not backend_error:
                raise
            print("acelerador falhou; reiniciando a geração inteira em CPU", file=sys.stderr)
            exc.__traceback__ = None  # libere pesos mantidos por frames antes do retry
            release_accelerator()
            audio, rate = synthesize(args, reference, transcript, chunks, "cpu")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Criação exclusiva: nem a referência nem uma saída anterior é sobrescrita.
        with args.out.open("xb") as target_file:
            sf.write(target_file, audio, rate, format="WAV", subtype="PCM_16")
        elapsed = time.perf_counter() - started
        print(f"salvo: {args.out} | {rate} Hz mono | {len(audio) / rate:.2f} s | tempo total: {elapsed:.1f} s")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"erro: {exc}\nconsulte a seção 'se der erro' do README", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
