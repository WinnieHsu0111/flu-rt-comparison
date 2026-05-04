# Run once to install required R packages
pkgs <- c(
  "EpiEstim",      # Method 1: Cori et al. renewal-equation Rt
  "pomp",          # Method 2: POMP + iterated filtering (mif2)
  "tidyverse",     # data wrangling + ggplot
  "lubridate",     # date handling
  "patchwork",     # combining plots
  "scales",        # axis formatting
  "knitr",         # report
  "rmarkdown"
)

new_pkgs <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs) > 0) install.packages(new_pkgs)

invisible(lapply(pkgs, library, character.only = TRUE))
cat("All R packages loaded.\n")
