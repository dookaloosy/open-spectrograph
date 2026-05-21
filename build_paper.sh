#!/bin/bash
# Build the LaTeX paper with embedded git version info.
set -e
cd "$(dirname "$0")"

GIT_VERSION=$(git describe --always --dirty --tags 2>/dev/null || echo "unknown")
GIT_DATE=$(git log -1 --format=%cd --date=format:'%-d %B %Y' 2>/dev/null || date +'%-d %B %Y')
cat > version.tex << VEOF
\newcommand{\gitversion}{$GIT_VERSION}
\newcommand{\gitdate}{$GIT_DATE}
VEOF

mkdir -p output
rm -f output/open-spectrograph.pdf output/open-spectrograph.aux \
      output/open-spectrograph.log output/open-spectrograph.out \
      output/open-spectrograph.toc
pdflatex -output-directory=output open-spectrograph.tex > /dev/null
pdflatex -output-directory=output open-spectrograph.tex | tail -3
