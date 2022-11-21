library(MASS)
library(lattice)

getOption("device")

x <- rnorm(1000)
y <- rnorm(1000)

truehist(c(x, y+3), nbins=25)
lol <- readline()
contour(dd <- kde2d(x,y))
