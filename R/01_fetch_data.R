## 01_fetch_data.R
## Pull weekly ILI (influenza-like illness) data from CDC FluView.
##
## Data source: CDC FluView ILINet, accessed via the public Delphi Epidata API.
## Endpoint:    https://api.delphi.cmu.edu/epidata/fluview/
## Docs:        https://cmu-delphi.github.io/delphi-epidata/api/fluview.html
##
## We pull weekly %ILI (percentage of outpatient visits for ILI) for a
## handful of US states over several seasons. This file produces:
##   data/raw/fluview_raw.csv
##   data/processed/fluview_weekly.csv

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
})

API_URL <- "https://api.delphi.cmu.edu/epidata/fluview/"

## Pull a single (region, epiweek-range) batch from the API
fetch_fluview <- function(region, epiweeks = "201740-202320") {
  url <- sprintf("%s?regions=%s&epiweeks=%s", API_URL, region, epiweeks)
  msg <- sprintf("Fetching %s ...", region)
  message(msg)

  res <- tryCatch(
    jsonlite::fromJSON(url),
    error = function(e) {
      warning(sprintf("Failed for %s: %s", region, conditionMessage(e)))
      NULL
    }
  )

  if (is.null(res) || res$result != 1) {
    warning(sprintf("API returned no data for %s", region))
    return(NULL)
  }
  as_tibble(res$epidata)
}

## Region codes (states + national). FluView uses lowercase 2-letter codes.
regions <- c("nat", "ca", "ny", "tx", "fl", "mi", "ma", "wa", "il")

raw_list <- lapply(regions, fetch_fluview)
raw <- bind_rows(raw_list)

dir.create("data/raw", showWarnings = FALSE, recursive = TRUE)
write_csv(raw, "data/raw/fluview_raw.csv")
message(sprintf("Saved raw data: %d rows", nrow(raw)))

## --- light cleaning -------------------------------------------------------
## epiweek e.g. 201740 -> ISO week-end Saturday date
epiweek_to_date <- function(yw) {
  y <- as.integer(substr(yw, 1, 4))
  w <- as.integer(substr(yw, 5, 6))
  ## ISO week: Monday of the week is given by ISOweek; CDC reports week-ending
  ## Saturday, so we add 5 days.
  jan4 <- as.Date(sprintf("%d-01-04", y))
  iso_mon <- jan4 - (as.integer(format(jan4, "%u")) - 1) + (w - 1) * 7
  iso_mon + 5
}

clean <- raw |>
  mutate(
    region = toupper(region),
    epiweek = as.integer(epiweek),
    week_end = epiweek_to_date(epiweek),
    season = case_when(
      epiweek %% 100 >= 40 ~ paste0(epiweek %/% 100, "-", (epiweek %/% 100) + 1),
      TRUE                 ~ paste0(epiweek %/% 100 - 1, "-", epiweek %/% 100)
    )
  ) |>
  select(region, season, epiweek, week_end,
         num_ili, num_patients, ili = wili) |>
  arrange(region, week_end)

dir.create("data/processed", showWarnings = FALSE, recursive = TRUE)
write_csv(clean, "data/processed/fluview_weekly.csv")
message(sprintf("Saved processed data: %d rows, %d regions, %d seasons",
                nrow(clean), n_distinct(clean$region), n_distinct(clean$season)))
