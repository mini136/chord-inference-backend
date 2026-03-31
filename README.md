# Chord Inference Backend

Tento repozitář obsahuje pouze složku `app` s backendem pro rozpoznávání akordů.

## Jak spustit backend (bez Dockeru)

1. Přejdi do složky `app/backend`:

```sh
cd app/backend
```

2. Nainstaluj závislosti (doporučeno použít virtuální prostředí):

```sh
python -m venv .venv
.venv\Scripts\activate  # Windows
# nebo
source .venv/bin/activate  # Linux/Mac
pip install -r requirements-backend.txt
```

3. Spusť backend pomocí uvicorn:

```sh
uvicorn app:app --reload
```

- API poběží na http://127.0.0.1:8000
- Dokumentace API: http://127.0.0.1:8000/docs

## Požadavky
- Python 3.10 nebo novější
- Závislosti v `requirements-backend.txt`
- Modelové soubory musí být ve složce `../model/` (relativně k `app/backend`)

## Poznámka
- Dockerfile je zde pouze pro referenci, není potřeba pro lokální spuštění.
