#!/usr/bin/env python3
"""Materialize identical canonical legal-intent examples in each model's expected JSONL schema."""
import json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/legal"
OUT = ROOT / "datasets/models"
SEED = 0
LABELS = [
    ("CONTRATO", "Contratos, cláusulas, revisão, elaboração, assinatura, rescisão e multas contratuais."),
    ("PROCESSO", "Andamento, consulta, decisões, recursos e situação de processo judicial."),
    ("PETICAO", "Elaboração, revisão ou protocolo de petições e manifestações processuais."),
    ("PRAZO", "Prazos processuais, vencimentos, contagem de dias e datas-limite."),
    ("AUDIENCIA", "Preparação, agendamento, participação ou dúvidas específicas sobre audiência."),
    ("CONSULTA_JURIDICA", "Orientação jurídica geral sobre direitos, opções e interpretação de uma situação."),
    ("TRABALHISTA", "Relação de emprego, demissão, salário, verbas trabalhistas e direitos do trabalhador."),
    ("FAMILIA", "Divórcio, guarda, pensão alimentícia, união estável e outros assuntos familiares."),
    ("CONSUMIDOR", "Problemas com produtos, serviços, cobranças, garantias e direitos do consumidor."),
    ("EMPRESARIAL", "Sociedades, operações empresariais, governança e questões jurídicas de empresas."),
    ("PREVIDENCIARIO", "INSS, aposentadoria, benefícios, contribuições e requerimentos previdenciários."),
    ("CRIMINAL", "Crimes, investigação policial, defesa criminal e procedimentos penais."),
]
INSTRUCTION = "Qual categoria jurídica representa melhor a intenção do usuário?"

def read_jsonl(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, row in enumerate(rows, 1):
        if not isinstance(row.get("text"), str) or not row["text"].strip() or row.get("label") not in dict(LABELS):
            raise ValueError(f"{path}:{i}: registro inválido: {row!r}")
    return rows

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")

def split_train(rows):
    rows = list(rows)
    random.Random(SEED).shuffle(rows)
    cut = round(len(rows) * 0.8)
    return rows[:cut], rows[cut:]

def prompt(row):
    return f"Mensagem do usuário: {row['text']} [Domínio: legal]"

def strands(row):
    return {"kind":"choice","state":prompt(row),"instruction":INSTRUCTION,
            "options":[{"id":i,"text":f"{name}: {desc}"} for i,(name,desc) in enumerate(LABELS)],
            "label": next(i for i,(name,_) in enumerate(LABELS) if name == row["label"])}

def tev(row):
    return {"text":prompt(row),"label":chr(ord("A")+next(i for i,(name,_) in enumerate(LABELS) if name == row["label"]))}

def laya(row):
    return {"text":row["text"],"label":row["label"],"domain":"legal"}

def main():
    train = read_jsonl(SOURCE / "train.jsonl")
    evaluation = read_jsonl(SOURCE / "eval.jsonl")
    labels = [x[0] for x in LABELS]
    if len(evaluation) != 132:
        raise ValueError(f"Eval congelado esperado: 132; encontrado: {len(evaluation)}")
    if set(x["label"] for x in train) != set(labels):
        raise ValueError("Taxonomia de treino não corresponde às 12 labels canônicas")
    tr, cal = split_train(train)
    # Canonical copies make each model's input provenance explicit.
    write_jsonl(OUT / "canonical/train.jsonl", train)
    write_jsonl(OUT / "canonical/eval.jsonl", evaluation)
    (OUT / "canonical/labels.json").write_text(json.dumps(
        [{"label":n,"description":d} for n,d in LABELS], ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    write_jsonl(OUT / "laya/train.jsonl", map(laya, tr))
    write_jsonl(OUT / "laya/calibration.jsonl", map(laya, cal))
    write_jsonl(OUT / "laya/eval.jsonl", map(laya, evaluation))
    write_jsonl(OUT / "tev1/train.jsonl", map(tev, tr))
    write_jsonl(OUT / "tev1/validation.jsonl", map(tev, cal))
    write_jsonl(OUT / "tev1/eval.jsonl", map(tev, evaluation))
    write_jsonl(OUT / "strands/train.jsonl", map(strands, tr))
    write_jsonl(OUT / "strands/calibration.jsonl", map(strands, cal))
    write_jsonl(OUT / "strands/eval.jsonl", map(strands, evaluation))
    write_jsonl(OUT / "jev/eval.jsonl", map(laya, evaluation))
    expected = {"train":len(train),"train_split":len(tr),"calibration":len(cal),"eval":len(evaluation)}
    (OUT / "manifest.json").write_text(json.dumps({
        "seed":SEED,"split":"80/20 of canonical train; evaluation remains frozen",
        "counts":expected,"labels":labels,
        "source_train_sha256_expected":"cbc70db9bd4c4a3acfbeb9e7cf0aa358e0956790",
        "note":"Model schemas differ; source texts, labels and frozen eval examples are shared."
    }, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(expected))
if __name__ == "__main__":
    main()
