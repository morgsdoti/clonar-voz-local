# clonar voz local 🎙️

um guia da [mariana morgs](https://x.com/morgsmariana), co-ceo da DOTI,
para gerar fala com a sua voz no próprio computador usando
[OmniVoice, do k2-fsa](https://github.com/k2-fsa/OmniVoice).

**sem elevenlabs, sem chave de API, sem mensalidade de geração.**
quem faz a síntese é o OmniVoice; este repo só facilita instalar, transcrever
uma referência, dividir o texto e salvar um WAV.

> **grátis não significa liberado para uso comercial.** o código deste guia é
> MIT, mas os **pesos do OmniVoice são CC-BY-NC (não comercial)**. o tokenizer
> Higgs tem termos adicionais e documentação de licença incompleta no espelho.
> leia [ética e licenças](#ética-e-licenças) e o [NOTICE](NOTICE) antes de usar.

## o que você precisa

- computador Windows, Linux ou Mac com Apple Silicon; **o teste real deste
  repo foi Linux x86_64, sem GPU**. os passos Windows/Mac são orientações ainda
  não executadas nesses sistemas.
- Python **3.12** e Git. outras versões não foram validadas neste guia.
- como ponto de partida, **16 GB de RAM**; o teste usou uma máquina com cerca
  de 15 GiB disponíveis no sistema, 8 vCPUs e 4 threads de inferência. menos
  memória pode não dar conta. feche outros programas pesados.
- reserve **10–15 GB livres** para ambiente, downloads e saídas. o cache dos
  modelos usado no teste ocupava aproximadamente **5 GB**, incluindo componentes
  de áudio e ASR. não é um limite máximo; instalações CUDA podem ocupar bem mais.
- internet para instalar e baixar os modelos na primeira vez. depois, com
  o cache completo, a geração pode funcionar offline. não é tempo real garantido:
  a CPU pode levar minutos para produzir segundos de áudio. no teste deste
  guia, 9,2 s de áudio em dois trechos levaram 90,1 s; um teste com
  transcrição integrada levou 47,6 s para 2,4 s de áudio.

não treinamos um modelo do zero: a voz é condicionada por uma referência curta.
o programa não precisa enviar sua referência para uma API. downloads iniciais
acessam PyPI/PyTorch/Hugging Face; isso não é a mesma coisa que gerar tudo offline
na primeira execução.

## 1. baixar e criar um ambiente

quando este repositório estiver disponível para você:

```bash
git clone https://github.com/morgsdoti/clonar-voz-local.git
cd clonar-voz-local
```

### linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

alternativa usada no ambiente de validação, caso você já tenha [uv](https://docs.astral.sh/uv/):
`uv venv --python 3.12 --seed .venv`, seguido de `source .venv/bin/activate`.
os próximos passos continuam sendo `python -m pip`, não `uv sync`.

se faltar `venv`, instale o pacote correspondente à sua distribuição. em
Debian/Ubuntu, normalmente `python3-venv`; se faltar libsndfile, `libsndfile1`.

### mac (apple silicon)

instale Python 3.12 e Git por uma fonte confiável; depois, no terminal:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

use um Python nativo arm64, não uma sessão Rosetta. a detecção de MPS é
experimental neste guia; se houver incompatibilidade, force `--device cpu`.

### windows (powershell)

instale Python 3.12 pelo python.org e Git for Windows. depois:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

se a política do PowerShell impedir a ativação, não é necessário alterar a
política: substitua `python` nos próximos comandos por
`.\.venv\Scripts\python.exe`. não use `sudo` nem desative proteções globais.

## 2. instalar o pytorch e o pacote oficial

escolha **uma** opção para PyTorch:

### CPU no linux/windows (é a opção validada no linux)

```bash
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

### apple silicon

```bash
python -m pip install torch==2.8.0 torchaudio==2.8.0
```

### NVIDIA/CUDA (opcional, não testado aqui)

confira primeiro o driver e a combinação no [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/).
exemplo do upstream para CUDA 12.8, somente se compatível com seu sistema:

```bash
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

agora, em qualquer plataforma:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -c "from omnivoice.models.omnivoice import OmniVoice; print('omnivoice instalado')"
```

isso instala **`omnivoice==0.2.1`, pacote oficial do upstream no PyPI**, não o
VoiceStudio. não copiamos nem vendorizamos nenhum dos dois. o README do
[upstream](https://github.com/k2-fsa/OmniVoice#installation) também documenta a
instalação via `pip install git+https://github.com/k2-fsa/OmniVoice.git`, mas
**a instalação testada aqui foi a versão fixa do PyPI**, não um checkout Git.
não misture fontes/versões no mesmo ambiente.

`requirements-tested-linux-cpu.txt` registra todas as versões do ambiente
Linux validado. é um retrato para diagnóstico, não um lock universal para
Mac/Windows/GPU. para reproduzi-lo no Linux CPU, após instalar o PyTorch pelo
índice CPU, use `python -m pip install -r requirements-tested-linux-cpu.txt`.

## 3. gravar uma boa referência

crie a pasta `dados` (ignorada pelo Git) e coloque `minha_ref.wav` nela:

```bash
python -c "from pathlib import Path; Path('dados').mkdir(exist_ok=True)"
```

- use **sua voz ou de alguém que autorizou explicitamente esse uso**.
- prefira **3–10 segundos**; nosso script recusa referências com mais de 20 s.
- uma só pessoa, voz natural, sem música, eco, efeitos ou cortes no meio da fala.
- grave sem estourar o volume. WAV é a opção mais simples; FLAC também funciona.
- prefira o mesmo idioma do texto final. áudio estéreo vira mono automaticamente.
- transcreva **exatamente o que foi dito no mesmo trecho**. não use o roteiro de
  outro take nem invente palavras que não aparecem no áudio.

para converter seu próprio arquivo com FFmpeg, se necessário:

```bash
ffmpeg -i minha_gravacao.m4a -t 10 -ac 1 -ar 24000 dados/minha_ref.wav
```

FFmpeg é opcional e precisa ser instalado separadamente para esse comando.
após recortar, a transcrição precisa corresponder ao trecho recortado.

## 4. transcrever e revisar

```bash
python transcrever.py --ref dados/minha_ref.wav --lang pt --out dados/ref.txt
```

usa `faster-whisper`, modelo `small`, na CPU com int8. o modelo é baixado na
primeira vez. **abra `dados/ref.txt` e corrija os erros antes de continuar**.
o comando não sobrescreve uma transcrição que já existe.

você também pode escrever a transcrição manualmente e pular este passo.

## 5. clonar e gerar o áudio

com a transcrição revisada:

```bash
python clonar.py --ref dados/minha_ref.wav --ref-text-file dados/ref.txt --texto "olá! este áudio foi gerado localmente, com autorização." --lang pt --out outputs/primeiro.wav --consentimento
```

ou passando a transcrição diretamente:

```bash
python clonar.py --ref dados/minha_ref.wav --ref-text "aqui vai exatamente o que eu disse na referência." --texto "este é o novo texto que eu quero narrar." --out outputs/segundo.wav --consentimento
```

ou sem `--ref-text`/`--ref-text-file`, para transcrever automaticamente:

```bash
python clonar.py --ref dados/minha_ref.wav --texto "um teste de geração local." --out outputs/automatico.wav --consentimento
```

esse modo é prático, mas **pode usar uma transcrição errada**. para um resultado
mais confiável, use o fluxo de revisão acima. para referência em outro idioma,
passe `--ref-lang en`, por exemplo; `--lang pt` continua sendo o idioma da saída.

### um texto maior, sem virar um bloco gigante

guarde seu texto UTF-8 em `dados/roteiro.txt` e rode:

```bash
python clonar.py --ref dados/minha_ref.wav --ref-text-file dados/ref.txt --arquivo dados/roteiro.txt --lang pt --out outputs/narracao.wav --max-chars 240 --consentimento
```

o script separa frases, subdivide frases longas em palavras e concatena os
resultados com 0,15 s de silêncio. não é garantia de entonação perfeita entre
cortes. ele mantém as saídas em RAM até gravar: para textos de muitas páginas,
trabalhe em lotes menores. `--pause 0` remove a pausa extra.

### controles úteis

- `--device auto`: tenta CUDA, depois MPS, senão CPU; erros reconhecidos do
  backend durante a geração provocam uma nova tentativa inteira na CPU.
- `--device cpu`: o caminho de referência testado; float32.
- `--device cuda`: float16 quando disponível; `--device mps`: float32.
  o upstream mantém o tokenizer em CPU quando usa MPS.
- `--threads 4`: limite de threads CPU; aumentar nem sempre deixa mais rápido.
- `--steps 16`: menos passos que o padrão 32; pode reduzir a qualidade.
- `--asr-model small`: modelo da transcrição automática.
- `--model /caminho/do/modelo`: modelo local completo, incluindo arquivos do
  tokenizer quando necessários; não confunda com apenas um `.safetensors`.
- `--help`: todas as opções. a saída é WAV PCM16 mono, na taxa do modelo
  (24 kHz no teste). arquivos de saída existentes nunca são sobrescritos.

## cache e uso offline

o padrão do Hugging Face fica no cache do usuário. você pode apontar para
um cache já existente; **não copie os pesos para dentro do Git**.

linux/mac (substitua pelo seu caminho real):

```bash
export HF_HOME="/caminho/para/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
# somente DEPOIS que todos os componentes tiverem sido baixados:
export HF_HUB_OFFLINE=1
```

powershell:

```powershell
$env:HF_HOME = "C:\caminho\huggingface"
$env:HF_HUB_CACHE = "$env:HF_HOME\hub"
# somente com o cache completo:
$env:HF_HUB_OFFLINE = "1"
```

os scripts respeitam essas variáveis. `HF_HOME` é a pasta pai, `HF_HUB_CACHE`
é a pasta que contém `models--...`. OmniVoice, tokenizer e faster-whisper são
componentes separados; ter só o modelo principal não basta. remova a variável
`HF_HUB_OFFLINE` se precisar completar os downloads. não contorne restrições
ou aceite termos automaticamente: leia e siga os termos de cada componente.

## se der erro

| problema | o que conferir |
|---|---|
| `No module named omnivoice` | ative o ambiente certo e reinstale `requirements.txt`; execute com o mesmo `python` usado no pip. não precisa rodar de dentro do repo upstream. |
| `torchcodec` / `torchaudio.save` | este guia usa **soundfile** para ler a referência e gravar o WAV, evitando esse caminho da CLI upstream. não é necessário instalar torchcodec para os comandos deste guia. |
| `metadata_errors` no PyAV | reinstale `requirements.txt`; fixamos `av==16.0.1` porque a versão 19 é incompatível com faster-whisper 1.2.1. |
| libsndfile ausente | reinstale a wheel de `soundfile`; no Linux, instale `libsndfile1` pela sua distribuição. |
| memória insuficiente / CUDA OOM | reduza `--max-chars`, feche outros programas e tente `--device cpu`; CPU também precisa de RAM. |
| erro MPS / operação não suportada | force `--device cpu`. Mac/GPU não foram validados neste repo. |
| download falha / arquivo não encontrado offline | confira cache, espaço e conexão; retire `HF_HUB_OFFLINE` para baixar componentes faltantes, respeitando licenças. |
| referência maior que 20 s | recorte e transcreva novamente. o script não corta escondido, para não desalinhá-la da transcrição. |
| voz estranha / palavras erradas | revise transcrição, idioma e referência; escreva números e siglas por extenso. a clonagem não é perfeita. |
| áudio final já existe | escolha outro `--out`; o script se recusa a sobrescrever. |
| `modelo retornou áudio vazio ou inválido` | o script para sem salvar um resultado inválido. evite fragmentos muito curtos: volte a `--max-chars 240`, use frases completas e confira referência/transcrição; se persistir, tente outra referência autorizada. |
| CPU lenta | teste um texto curto, 4 threads e `--steps 16`; não prometa tempo real. |

## testes e limites do que foi validado

consulte [TESTES.md](TESTES.md) para comandos, versões, tempos e resultados.
para testes rápidos sem baixar/carregar pesos:

```bash
python -m unittest discover -s tests -v
```

não foram testados Windows, Mac, CUDA ou MPS. detectar um backend não equivale
a garantir compatibilidade. também não foi feita auditoria jurídica da cadeia
completa, nem avaliação humana de semelhança da voz.

## ética e licenças

1. **só clone sua própria voz ou uma voz com consentimento explícito.**
   ter um vídeo público de alguém não é consentimento. não use para fraude,
   se passar por outra pessoa, burlar autenticação ou fabricar declarações.
2. sinalize quando um áudio foi gerado por IA. o WAV deste script **não tem
   marca-d'água nem detector de impersonação**; a flag `--consentimento` é uma
   declaração sua, não uma verificação automática de identidade ou permissão.
3. referências, transcrições e tokens de voz são dados sensíveis. guarde-os
   em pasta privada; não publique no Git. `.gitignore` é uma proteção básica,
   não um scanner de segredos. revise sempre `git diff --cached`.
4. **scripts e documentação novos: [MIT](LICENSE)**, Mariana Morgs / DOTI.
5. **código OmniVoice: Apache-2.0**, autores k2-fsa; **pesos: CC-BY-NC**, como
   declara o [model card](https://huggingface.co/k2-fsa/OmniVoice#license).
   não confunda licença do código com licença dos pesos. não trate este guia
   como autorização para narrar anúncios, vender serviços ou explorar
   comercialmente saídas: confira as condições e obtenha permissão adequada.
6. **tokenizer Higgs:** há termos Boson/Meta e componentes com condições
   próprias. o [card do espelho usado pelo upstream](https://huggingface.co/eustlb/higgs-audio-v2-tokenizer)
   está incompleto; isso **não** significa ausência de restrições. consulte
   as fontes e o [NOTICE](NOTICE). não alegamos fechamento completo de licença.
7. demais dependências/modelos mantêm suas próprias licenças. nenhuma amostra
   de voz, código VoiceStudio (AGPL), pesos ou chaves foi incluída neste repo.

**créditos:** o trabalho de síntese é dos autores de
[OmniVoice](https://github.com/k2-fsa/OmniVoice), não uma tecnologia criada pela
DOTI. transcrição com [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
este é um guia independente da mariana para experimentar **com responsabilidade**.
