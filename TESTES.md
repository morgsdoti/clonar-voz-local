# validação técnica

validação em 6 de novembro de 2026, no Linux Debian 13 x86_64,
Python 3.12.15, 8 vCPUs, aproximadamente 15 GiB de RAM total, sem GPU.
inferência CPU/float32 com **4 threads**, modelo e tokenizer a 24 kHz.

## instalação realmente executada

ambiente virtual **novo**, separado do runtime que já existia. bootstrap com
`uv venv --seed` usando Python 3.12.15; nenhuma dependência veio do site-packages
do runtime antigo. a instalação de pacotes foi pelo **pip**:

```bash
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
python -c "from omnivoice.models.omnivoice import OmniVoice; print('omnivoice instalado')"
```

resultado: importação passou e `pip check` não encontrou dependências quebradas.
a primeira instalação limpa de torch + OmniVoice e dependências levou cerca
de **59 s** (rede/cache deste ambiente; não é promessa para outros PCs).

versões principais: OmniVoice 0.2.1 (PyPI oficial), torch/torchaudio 2.8.0+cpu,
transformers 5.22.2, faster-whisper 1.2.1, soundfile 0.13.1, PyAV 16.0.1.
o snapshot completo está em `requirements-tested-linux-cpu.txt`.

**correção encontrada no teste:** o resolvedor inicialmente instalou PyAV 19;
faster-whisper 1.2.1 falhou com `metadata_errors`. a versão PyAV 16.0.1 foi
fixada no manifesto, reinstalada e a transcrição passou.

## modelos e privacidade

a inferência reutilizou cache já presente, com `HF_HOME` e `HF_HUB_CACHE`
apontando para ele, e **`HF_HUB_OFFLINE=1`**. não houve redownload dos ~5 GB
de modelos. revisões observadas:

| componente | revisão no cache |
|---|---|
| k2-fsa/OmniVoice | `c5fdb5ccb189668d56333f77ba2629f4cd7535f4` |
| eustlb/higgs-audio-v2-tokenizer | `528e871c2a26c4f0f7773b9754e2e1acae20899d` |
| Systran/faster-whisper-small | `536b0662742c02347bc0e980a01041f333bce120` |

as revisões foram registradas para auditoria, não fixadas nos scripts:
novos downloads podem resolver revisões diferentes e precisam ser reavaliados.
não foram copiados código VoiceStudio, áudios, roteiros ou scripts privados.
nenhuma amostra real de pessoa foi usada: a referência foi **sintetizada pelo
próprio OmniVoice em modo auto, sem ref_audio/ref_text**.

## referência sintética reproduzível

em um ambiente preparado, a lógica usada para criar a referência foi:

```python
from pathlib import Path
import torch
import soundfile as sf
from omnivoice.models.omnivoice import OmniVoice

torch.set_num_threads(4)
torch.manual_seed(42)
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32
)
text = "Olá, este é um exemplo de voz sintética. Vamos testar a geração de áudio no computador."
audio = model.generate(text=text, language="pt", duration=8.0, num_step=32)[0]
Path("dados").mkdir(exist_ok=True)
sf.write("dados/ref_sintetica.wav", audio, model.sampling_rate)
```

**não execute se esse WAV já existir**, pois o exemplo mínimo usa sf.write
diretamente. os comandos clonar.py/transcrever.py se recusam a sobrescrever.
no teste, todos os artefatos ficaram em uma pasta temporária fora do repo.
a referência ficou com **7,26 s**, com geração em **39,12 s**; carregar os
modelos levou **2,03 s** (total interno 41,15 s). `duration=8` não é garantia
de duração exata depois do pós-processamento upstream.

## transcrição

```bash
python transcrever.py --ref dados/ref_sintetica.wav --lang pt --out dados/ref.txt
```

**passou: 2,4 s** de tempo interno, com faster-whisper small CPU/int8 offline.
a transcrição reconheceu a frase, mas omitiu a palavra “áudio” em relação ao
texto de síntese. isso reforça a necessidade de revisão e não constitui uma
avaliação humana do que efetivamente foi pronunciado.

## geração em trechos e transcrição integrada

texto genérico usado em `dados/roteiro.txt`:

> este é um teste de geração de fala no próprio computador. o texto foi dividido em trechos e unido em um único arquivo.

comandos (na execução real, os caminhos apontavam para a pasta temporária):

```bash
python clonar.py --ref dados/ref_sintetica.wav --ref-text-file dados/ref.txt --arquivo dados/roteiro.txt --lang pt --out outputs/clonado_trechos.wav --max-chars 240 --device cpu --threads 4 --consentimento
python clonar.py --ref dados/ref_sintetica.wav --texto "olá, este é outro teste local." --lang pt --out outputs/clonado_auto.wav --device auto --threads 4 --consentimento
```

| teste | resultado | tempo interno total |
|---|---|---|
| arquivo UTF-8, transcrição em arquivo, duas frases/trechos concatenados | WAV PCM16, 24 kHz mono, **9,20 s** | **90,1 s** |
| transcrição automática integrada, device auto detectou CPU | WAV PCM16, 24 kHz mono, **2,40 s** | **47,6 s** |

os dois comandos terminaram com código 0. uma transcrição adicional do WAV
curto reconheceu “Olá, este é outro teste local.”, igual ao texto solicitado
(desconsiderando maiúsculas), em 2,1 s; isso não substitui avaliação auditiva. os WAVs foram reabertos com
soundfile: duração, formato, valores finitos e sinal não silencioso confirmados
(RMS 0,057256 e 0,064808 respectivamente). os tempos incluem carregar modelos;
não incluem o download inicial. as pausas extras entre trechos fazem parte
da duração final. CPU foi aproximadamente 10–20 vezes mais lenta que o áudio
nesses exemplos curtos; a proporção muda com texto, referência e máquina.

**limitação encontrada:** uma tentativa com `--max-chars 45` criou quatro
fragmentos; o modelo retornou áudio vazio/inválido em um deles. o script
parou corretamente, com saída não-zero, sem gravar um WAV incompleto.
o README recomenda frases completas e o padrão 240, não fragmentos mínimos.

## testes automatizados e limites

`python -m unittest discover -s tests -v`: **23 testes passaram**. cobrem
divisão de texto, entradas vazias, referência silenciosa ou >20 s, estéreo,
saída inválida, não sobrescrever arquivos, consentimento obrigatório, gravação
PCM16 e escolha/fallback de backend **com mocks**. AST/compileall também passaram.

Windows, Mac/Apple Silicon, CUDA, MPS, falha real de GPU e instalação Git do
upstream **não foram testados**. não houve avaliação humana auditiva, benchmark
de semelhança de voz, teste de horas de narração ou auditoria jurídica completa.
o teste prova instalação e execução local no ambiente descrito, não qualidade
perfeita nem funcionamento em todo hardware. teste de download inicial dos
pesos não foi refeito: o cache existente foi usado deliberadamente.

nenhum WAV, referência, transcrição de teste, venv, cache ou peso entra no
commit. os testes não autorizam uso comercial: veja [NOTICE](NOTICE).
