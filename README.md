# Intent Datasets 4 Fine-tuning

Datasets para classificação de intenção a partir de pedidos humanos.

## Legal

Primeiro experimento: pedidos em português do Brasil relacionados a advocacia.

Objetivo: classificar **o que o humano pretende**, sem ensinar o modelo a executar a ação.

Pipeline: `human request → category/intent → BehaviorID → Skill`

Arquivos:
- `datasets/legal/train.jsonl` — treinamento.
- `datasets/legal/eval.jsonl` — avaliação separada.
- `datasets/legal/labels.md` — taxonomia.

Formato: `{"text":"Quero revisar um contrato antes de assinar.","label":"CONTRATO","domain":"legal"}`

O conjunto é sintético e destinado a experimentação de classificação/fine-tuning; não é corpus jurídico nem fonte de aconselhamento jurídico.

Métricas: accuracy, macro-F1, matriz de confusão, confiança e taxa de confusão entre categorias.
