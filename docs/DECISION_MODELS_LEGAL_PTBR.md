# Comparação: Laya vs Strands Decider vs Jev vs Tev1 — intenção jurídica PT-BR

## Conclusão executiva

Para qualificar a intenção de um possível cliente em português-BR, usando a taxonomia jurídica deste projeto e o dataset real de 14.806 exemplos, a escolha atual é:

1. **Laya — melhor escolha hoje para produção**
2. **Strands Decider — melhor candidato para o próximo experimento**
3. **Tev1 4B — candidato forte, mas inferior em desenho de saída/calibração**
4. **Jev — excelente como serviço generalista/zero-shot, mas não é a melhor escolha para este pipeline local e específico**

A posição do Strands é provisória: o notebook deste repositório permite treiná-lo no mesmo protocolo. Depois que o treino for executado, o resultado deve substituir esta posição se ele superar o Laya no mesmo eval.

## O que está sendo comparado

| Modelo | Arquitetura | Fine-tuning neste projeto | Saída | Local | Calibração |
|---|---|---|---|---|---|
| Laya | ModernBERT-large + decision head | Sim, já executado | choice + probabilidades | Sim | Sim |
| Strands Decider 2B | Qwen3.5-2B + pointer head + LoRA | Sim, notebook criado | choice/score/noul | Sim | Sim |
| Tev1 4B | Qwen3.5-4B + LM head/LoRA | Sim, scripts criados | choice | Sim via Ollama/HF | Não é confiança calibrada por desenho |
| Jev 1.13 | System One proprietário | Não | choice/score/noul | Não | Sim |

O Strands remove a cabeça de linguagem do Qwen3.5 e usa uma pointer head que compara a representação da posição de decisão com a representação da própria opção. Isso evita que o modelo aprenda que uma posição fixa é mais provável que outra. O torso recebe LoRA. citeturn3view0

O Tev1 é diferente: é um fine-tune supervisionado do Qwen3.5-4B que mantém a cabeça autoregressiva e retorna uma letra correspondente à opção escolhida. O material público o descreve como choice-oriented de 2–24 opções. citeturn1search1turn1view0

## Laya: o ponto de referência real

O Laya que já treinamos teve, nos 132 exemplos jurídicos congelados:

- Accuracy: **95,4545%**
- Macro-F1: **95,4408%**
- Brier: **0,07992**
- ECE: **0,03795**
- Confiança média: **0,99250**
- 6 erros em 132
- p50: **35,07 ms**
- p95: **42,60 ms**

Esse é o número mais importante da comparação porque é medido no mesmo domínio, idioma, taxonomia e eval que queremos usar em produção.

Os seis erros já mostraram onde está o problema: CONSULTA_JURIDICA↔AUDIENCIA, CONSULTA_JURIDICA↔CONSUMIDOR, CONTRATO↔TRABALHISTA, FAMILIA↔PREVIDENCIARIO e CRIMINAL↔PRAZO.

Portanto, o próximo ganho provável não vem simplesmente de aumentar o modelo: vem de aprender melhor esses pares de confusão.

## Strands Decider

O Strands é, conceitualmente, o concorrente mais interessante do Laya para este projeto.

O modelo oficial atual usa Qwen3.5-2B como torso, uma pointer head de aproximadamente 1M de parâmetros e LoRA rank 16. A cabeça pontua cada opção usando o hidden state daquela própria opção; não existe uma tabela fixa de classes na cabeça. citeturn3view0

### Vantagens
- Muito mais próximo do conceito de decision model do que um LLM generativo.
- Pointer head evita viés estrutural por posição.
- Permite choice, noul e score.
- Calibração explícita por temperatura.
- Qwen3.5-2B fornece um torso muito maior que o ModernBERT do Laya.
- Permite adicionar novas taxonomias sem criar uma cabeça específica por conjunto de classes.
- O projeto é Apache-2.0. citeturn3view0

### Desvantagens
- Mais pesado que Laya.
- Treino mais complexo.
- O Qwen3.5 usa Gated DeltaNet e o treinamento oficial recomenda Linux/WSL2 por causa de flash-linear-attention. citeturn0view0
- Ainda não temos o resultado do Strands treinado especificamente no seu dataset jurídico PT-BR.

### Benchmark público

O Strands v21 reporta 0,762 no JevBench público, com Brier 0,323 e ECE 0,064. Esses números são de outro domínio e não devem ser comparados diretamente com os 95,45% do Laya jurídico. citeturn3view0

## Tev1 4B

Tev1 parte de um Qwen3.5-4B e foi explicitamente treinado como modelo de decisão. O repositório público da Together descreve 37.840 exemplos de treino e 4.568 de validação na receita publicada. citeturn1search1

No Ollama, o Tev1 4B expõe /v1/systemone, aceita choice, noul e score, e suporta até 64 perguntas por requisição; a faixa de treino de choice foi 2–24 opções. citeturn1view0

Mas existe uma diferença fundamental: Tev1 não possui a mesma arquitetura não-autoregressiva do Strands/Laya. Ele continua usando a cabeça LM do Qwen. Além disso, a confiança retornada pelo runtime não deve ser interpretada automaticamente como probabilidade calibrada de acerto. citeturn1view0turn1search0

Para classificação pura isso não impede o Tev1 de funcionar. Porém, para nosso pipeline de triagem, queremos transformar confiança em política operacional. Nesse cenário, calibração é parte do produto, não um detalhe.

### Benchmark público do Tev1

No benchmark público do Ollama, o Tev1 4B aparece com 73,3% em 3.880 decisões de 13 datasets, contra 76,0% do Jev 1.13. A própria página separa isso dos 88%/100% de desenvolvimento reportados pela Together, que não são um benchmark independente. citeturn1view0

Esses números não dizem que Tev1 será 73% no nosso dataset. Eles apenas mostram que não devemos assumir que 4B > 421M automaticamente.

## Jev

Jev é o benchmark conceitual da categoria.

Ele foi criado especificamente para decisões tipadas e devolve choice/score/noul com probabilidades. Há evidência independente de excelente desempenho em várias tarefas; por exemplo, um benchmark pré-registrado recente mediu 95,9% em seu conjunto de 400 itens. citeturn2search1

### Português-BR

A documentação do ecossistema Jev informa que inglês é a língua principal de treinamento e onde a acurácia é melhor; para outros idiomas, recomenda validar no próprio conteúdo antes de confiar na carga. citeturn4search3

Há testes comunitários mostrando que Jev funciona em português, mas eles são pequenos e não substituem um benchmark jurídico de 132+ casos. citeturn4search0

### Personalização

O Jev é serviço proprietário. Não podemos pegar seus pesos, treinar diretamente com os 14.806 exemplos e produzir um checkpoint jurídico local equivalente ao Laya/Strands/Tev1.

## Para o nosso problema, o que realmente importa

1. Accuracy
2. Macro-F1
3. Recall por intenção
4. Confusão entre classes juridicamente próximas
5. ECE/Brier
6. Confiança nos erros
7. Latência
8. Custo
9. Privacidade/execução local
10. Facilidade de retreino

A quantidade de parâmetros fica abaixo desses critérios.

## Ranking para este produto

### 1. Laya — melhor escolha atual

Porque é o único dos quatro para o qual já temos uma medição completa e específica do problema: 95,45% accuracy / 95,44% Macro-F1 / ECE 0,038 / p50 35 ms.

Ele também é extremamente pequeno para o nível de desempenho obtido.

### 2. Strands Decider — maior potencial

É o modelo que eu testaria imediatamente contra o Laya.

A arquitetura é especialmente adequada para o que estamos fazendo: opções são parte da entrada e a cabeça pontua cada opção diretamente. citeturn3view0

Se o notebook conseguir Accuracy > 95,5%, Macro-F1 > 95,5%, ECE <= 0,04, erros menores nos cinco pares conhecidos e latência aceitável, eu migraria a preferência para o Strands.

### 3. Tev1 4B

Eu manteria como terceiro candidato. Ele pode ser excelente em accuracy, mas a arquitetura autoregressiva e a ausência de uma calibração equivalente tornam-no menos atraente para o roteamento automático baseado em confiança.

### 4. Jev

Eu escolheria Jev se o objetivo fosse zero-shot, não querer manter modelo, aceitar API externa e querer decisões tipadas imediatamente. Para este projeto específico, não é o vencedor.

## Minha decisão

**Hoje eu colocaria o Laya em produção.**

Não porque ele seja teoricamente o melhor modelo da categoria, mas porque ele é o melhor modelo comprovado neste problema específico.

O Strands é o único que pode mudar essa decisão de maneira convincente porque agora podemos fazer uma comparação controlada: mesmo dataset → mesmo split → mesmo eval → mesmo idioma → mesma taxonomia → mesmas métricas.

O notebook criado é `notebooks/strands_decider_legal_intent_finetuning_colab.ipynb`.

Depois do treino, a decisão passa a ser puramente experimental.

### Regra objetiva de promoção

Strands substitui Laya somente se:
- Accuracy for superior;
- Macro-F1 não cair;
- ECE não piorar;
- nenhum par crítico de classes piorar significativamente;
- p95 continuar aceitável;
- e o ganho permanecer em um segundo seed.

Se empatar, fica Laya, porque ele já é menor e mais rápido.

## Arquitetura recomendada para o produto

WhatsApp → normalização → Laya/Strands → confiança → política determinística

Exemplo:

- confidence >= 0,95 → aceita automaticamente
- 0,80 <= confidence < 0,95 → aceita se não for uma classe de risco
- 0,60 <= confidence < 0,80 → segunda verificação
- confidence < 0,60 → LLM/revisor

E, principalmente, regras específicas para os pares que já mostraram confusão:
- CONSULTA_JURIDICA ↔ AUDIENCIA
- CONSULTA_JURIDICA ↔ CONSUMIDOR
- CONTRATO ↔ TRABALHISTA
- FAMILIA ↔ PREVIDENCIARIO
- CRIMINAL ↔ PRAZO