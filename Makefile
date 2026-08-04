.PHONY: help install test test-fast test-reference test-soluciones status next check demo train-tiny train-final sample data clean

UV := uv
RUN := $(UV) run

help:  ## Muestra esta ayuda
	@echo ""
	@echo "  Tres suites distintas y no hay que confundirlas:"
	@echo "    make test            -> tu progreso. ROJO es lo normal: son los ejercicios que faltan."
	@echo "    make test-reference  -> salud del curso. Aqui todo VERDE siempre."
	@echo "    uv run pytest tests/ -> la infraestructura del repo. Verde siempre."
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Crea el venv e instala todo (incluye extras de comparativa)
	$(UV) sync --extra compare

test:  ## TU progreso: infraestructura + tus ejercicios. Rojo hasta que los implementes.
	$(RUN) pytest

test-fast:  ## Suite sin los tests lentos
	$(RUN) pytest -m "not slow"

test-reference:  ## SALUD DEL CURSO: tests contra llmfs/reference/. Siempre verde; si no, es un bug del repo.
	LLMFS_TEST_REFERENCE=1 $(RUN) pytest modules/

test-soluciones:  ## Pega el codigo de cada SOLUTION.md y corre sus tests. Tarda un par de minutos.
	$(RUN) python scripts/verify_solutions.py

status:  ## Tabla de progreso del curriculo
	$(RUN) python -m llmfs status

next:  ## En que modulo estoy y que toca ahora
	$(RUN) python -m llmfs next

check:  ## Tests de un modulo. Uso: make check N=05
	$(RUN) python -m llmfs check $(N)

demo:  ## Experimento de un modulo. Uso: make demo N=05
	$(RUN) python -m llmfs demo $(N)

data:  ## Descarga y tokeniza TinyStories (tarda; se cachea en data/)
	$(RUN) python -m llmfs data prepare --config configs/tinystories_9m.yaml

train-tiny:  ## Entrena el modelo juguete de shakespeare (~1 min)
	$(RUN) python -m llmfs train --config configs/tiny_char.yaml

train-final:  ## La tirada real de TinyStories (horas). Reanudable.
	$(RUN) python -m llmfs train --config configs/tinystories_9m.yaml --resume

sample:  ## Genera texto del ultimo checkpoint. Uso: make sample PROMPT="Once upon a time"
	$(RUN) python -m llmfs sample --prompt "$(PROMPT)"

clean:  ## Borra cache de python y figuras generadas
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache runs/figures
