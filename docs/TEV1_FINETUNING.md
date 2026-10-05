# Tev1-4B — fine-tuning jurídico local

Este diretório adapta datasets/legal/train.jsonl e datasets/legal/eval.jsonl para o formato nativo de decisão do Tev1.

Dataset atual:
- train: 14.806 exemplos
- eval: 132 exemplos congelados
- 12 categorias

Mapeamento fixo, igual ao request /v1/systemone usado no Ollama:
A AUDIENCIA
B CONSULTA_JURIDICA
C CONSUMIDOR
D CONTRATO
E CRIMINAL
F EMPRESARIAL
G FAMILIA
H PETICAO
I PRAZO
J PREVIDENCIARIO
K PROCESSO
L TRABALHISTA

O modelo recebe state + question + options e é treinado para emitir somente a letra correta.

O state é:
Mensagem do usuario: <mensagem> [Dominio: legal]

O fine-tuning usa o checkpoint Hugging Face togethercomputer/Tev1-4B-experimental. O GGUF do Ollama é usado para inferência; ele não é o artefato de treinamento.

Hardware alvo inicial:
- RTX 4070 12 GB
- LoRA r=8
- alpha=16
- dropout=0
- target_modules=all-linear
- batch=1
- gradient accumulation=16
- gradient checkpointing
- fp16
- 1 epoch
- max length=768
- learning rate=5e-5

Executar na raiz do repo:

powershell -ExecutionPolicy Bypass -File .\scripts\finetuning\tev.ps1

Variáveis opcionais:
$env:TEV_EPOCHS="1"
$env:TEV_MAX_LENGTH="768"
$env:TEV_GRAD_ACCUM="16"
$env:TEV_BATCH_SIZE="1"
$env:TEV_LR="5e-5"

Saída:
outputs/tev1-legal/final/

Avaliação:
.venv-tev\Scripts\python.exe scripts\finetuning\tev_eval.py

O resultado usa o mesmo eval congelado de 132 exemplos para comparação direta com o benchmark do Laya.
