
# this is for getting data from the diverse human faces so you can use that in your model

library(glue)
library(tidyverse)

# dataset metadata  ~/Documents/Github/xai_ppa/data/human_info.csv
human_info <- read_csv('data/human_info.csv', show_col_types = FALSE)

# this is only values that do not change per images for a class, e.g. skin tone wont change, but yaw position will
unique_humans <- human_info |>
  filter(skin_labels != 'misc', yaw_direction != 'vacant') |>
  distinct(human_id, .keep_all = TRUE) |>
  select(-c(render_id:task_id, image_path, yaw:pitch, yaw_direction)) 

# making sure the faces are what you used
df <- map_df(1:48, function(x) {
  # Add leading zero for single digits
  sub_path <- glue("~/Documents/Github/xai_ppa/data/qualtrics/trial_{sprintf('%02d', x)}")
  file_paths <- list.files(sub_path, full.names = TRUE)
  tibble(
    trial_n = x, # sprintf('%02d', x)
    folder = str_extract(string = sub_path, pattern = '[^/]+$'),
    file = str_extract(string = file_paths, pattern = '[^/]+$'))
}) |>
  # you originally had 6 explanations for each model, so remove those
  filter(str_detect(file, "^dark1_|^light1_|^fair1|^input_|^pred_")) |>
  mutate(
    # for joining later
    render_id = str_extract(file, '(?<=_)[[:digit:]]+') |> as.numeric(),
    condition = case_when(
      trial_n <= 16 ~ 'fair_dark',
      trial_n <= 32 ~ 'fair_light',
      trial_n > 32 ~ 'biased'),
    # create a logical so you can drop xai that are not relevant
    drop_row = case_when(
      condition == 'fair_dark' & str_starts(file, "light1_") ~ TRUE,
      condition == 'fair_light' & str_starts(file, "dark1_") ~ TRUE,
      condition == 'biased' & str_starts(file, "fair1") ~ TRUE,
      TRUE ~ FALSE),
    # just a name for the type of image
    image_label = str_extract(file, '^[^_\\d]+'), # if you want the number '^[^_]+'
    # label for which model the xai is from, to avoid confusion with the image label, e.g. skin tone of input image
    model = case_when(
      str_starts(file, "light1_") ~ 'light_undersampled',
      str_starts(file, "dark1_") ~ 'dark_undersampled',
      str_starts(file, "fair1_") ~ 'fair',
      TRUE ~ NA)
  ) |>
  # drop those columns
  filter(!drop_row) |>
  # can now remove that column
  select(-drop_row) |>
  left_join(human_info |> select(human_id, render_id, sex, yaw:skin_labels), by = c('render_id'))

# str_extract(pattern = '(\\d)+')

# 5 images per trial (most trials have 2 input_images in case)

# human_info |> filter(render_id %in% df[['render_id']])

write_csv(df, 'trial_dataset.csv')

trial_dataset <- read_csv(file = 'trial_dataset.csv', show_col_types = FALSE)