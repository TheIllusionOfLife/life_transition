.PHONY: all figures semi_life_figures paper clean

all: semi_life_figures paper

figures:
	cd .. && uv run python scripts/generate_figures.py

semi_life_figures:
	cd .. && uv run python scripts/generate_semi_life_figures.py

paper: semi_life_figures
	tectonic main.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out main.pdf
	rm -f figures/*.pdf
