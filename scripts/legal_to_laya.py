#!/usr/bin/env python3
"""Convert legal JSONL into Laya state/questions/gold JSONL."""
import argparse, json
from pathlib import Path

INSTRUCTIONS = "Qual categoria jurídica representa melhor a intenção do usuário?"

LABEL_DESCRIPTIONS = {
    "CONTRATO": "contratos, cláusulas, rescisão e revisão contratual",
    "PROCESSO": "andamento, situação ou consulta de processo",
    "PETICAO": "petições, peças processuais e documentos para apresentar",
    "PRAZO": "prazos processuais, recursos e vencimentos",
    "AUDIENCIA": "audiências, preparação e participação em audiência",
    "CONSULTA_JURIDICA": "orientação jurídica geral sobre direitos e situação",
    "TRABALHISTA": "emprego, salário, férias, demissão e relações de trabalho",
    "FAMILIA": "divórcio, guarda, pensão e relações familiares",
    "CONSUMIDOR": "produtos, serviços, cobranças e direitos do consumidor",
    "EMPRESARIAL": "empresas, sócios, negócios e questões empresariais",
    "PREVIDENCIARIO": "INSS, aposentadoria, benefícios e contribuição",
    "CRIMINAL": "acusações criminais, investigação, defesa e polícia",
}

def rows(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = list(rows(Path(args.input)))
    labels = sorted({r["label"] for r in data})
    unknown = [x for x in labels if x not in LABEL_DESCRIPTIONS]
    if unknown:
        raise SystemExit(f"Unknown labels: {unknown}")

    criteria = {x: LABEL_DESCRIPTIONS[x] for x in labels}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for row in data:
            probabilities = {label: 1.0 if label == row["label"] else 0.0 for label in labels}
            item = {
                "state": {"text": row["text"], "domain": row.get("domain", "legal")},
                "questions": {"intent": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}},
                "gold": {"intent": {"label": row["label"], "probabilities": probabilities}},
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(data)} Laya cases to {out}")
    print("Labels:", ", ".join(labels))

if __name__ == "__main__":
    main()