#json_convert.R
options(warn = -1)
library(arrow)
library(readr)
library(dplyr)
library(purrr)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
filename <- args[1]


# lines <- read_lines("diverse_human_faces_sorted/metadata.jsonl") |>
lines <- readr::read_lines(filename) |> 
  purrr::map(jsonlite::fromJSON) |>
  purrr::map(unlist) |>
  dplyr::bind_rows()

readr::write_csv(lines, 'metadata.csv')
arrow::write_parquet(lines, 'metadata.parquet')