TEX = pdflatex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=out
SRC = main.tex

all: out/main.pdf

out/main.pdf: $(SRC)
	$(TEX) $(SRC)
	biber --output-directory=out main
	$(TEX) $(SRC)
	$(TEX) $(SRC)

quick:
	$(TEX) $(SRC)

clean:
	rm -rf out/*