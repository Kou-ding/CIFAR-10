TEX_MAIN = neural1.tex

all: neural1

report: $(TEX_MAIN)
	latexmk -pdf  $(TEX_MAIN) 

clean:
	latexmk -C $(TEX_MAIN)
	rm -f *.bbl 