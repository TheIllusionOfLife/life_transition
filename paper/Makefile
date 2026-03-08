.PHONY: all figures semi_life_figures paper supplement clean reproduce

all: semi_life_figures paper supplement

# Full reproducibility pipeline: experiments → analysis → figures → paper
reproduce:
	cd .. && uv run python scripts/experiment_semi_life_v1v3.py > experiments/semi_life_v1v5_test.tsv
	cd .. && uv run python scripts/experiment_semi_life_shocks.py > experiments/semi_life_shocks.tsv
	cd .. && uv run python scripts/analyze_semi_life_capability_ladder.py
	cd .. && uv run python scripts/generate_latex_macros.py
	$(MAKE) all

figures:
	cd .. && uv run python scripts/generate_figures.py

semi_life_figures:
	cd .. && uv run python scripts/generate_semi_life_figures.py

paper: semi_life_figures
	tectonic main.tex

supplement:
	tectonic supplement.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out main.pdf
	rm -f figures/*.pdf
