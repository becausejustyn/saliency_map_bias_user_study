This post-study analysis is written entirely in R. The helper functions `parse_drive()`, `rename_columns()`, `participant_condition()`, `reverse_conditions()`, `cond_label()`, and `preferred_model()` are provided by the [`brrr`](https://github.com/becausejustyn/brrr) package.

```r
# Install brrr once if it is not already available:
# install.packages("remotes")
# remotes::install_github("becausejustyn/brrr")

library(brrr)
library(uuid)
library(dplyr)
library(readr)
library(purrr)
library(stringr)
library(tidyr)
library(broom)
library(ggplot2)
options(scipen = 999)
```

```r
#### Named Values/Vectors, etc. ----

cond_prefix <- c("FD1_", "FD2_", "FL1_", "FL2_", "B1_", "B2_")

# column_ranges another name
cond_ranges <- list(
  FD1 = c("q1_1_25", "q16_2_56"),
  FD2 = c("q1_1_57", "q16_2_88"),
  FL1 = c("q1_1_89", "q16_2_120"),
  FL2 = c("q1_1_121", "q16_2_152"),
  B1 = c("q1_1_153", "q16_2_184"),
  B2 = c("q1_1_185", "q16_2_216")
)


#### Functions ----

# from the experiment block, get the models used, e.g.
# get_model_cond(df, condition) where condition == 'FD1'
# returns fair for cond1 and dark for cond2
get_model_cond <- function(data, column_name) {
  column_name <- enquo(column_name)
  data <- data |>
    mutate(
      cond1 = case_when(
        {{column_name}} == 'FD1' ~ 'fair',
        {{column_name}} == 'FD2' ~ 'dark',
        {{column_name}} == 'FL1' ~ 'fair',
        {{column_name}} == 'FL2' ~ 'light',
        {{column_name}} == 'B1' ~ 'light',
        {{column_name}} == 'B2' ~ 'dark') |> as.factor(),
      cond2 = case_when(
        {{column_name}} == 'FD1' ~ 'dark',
        {{column_name}} == 'FD2' ~ 'fair',
        {{column_name}} == 'FL1' ~ 'light',
        {{column_name}} == 'FL2' ~ 'fair',
        {{column_name}} == 'B1' ~ 'dark',
        {{column_name}} == 'B2' ~ 'light') |> as.factor()
    )
  return(data)
}

# does a few different methods for checking if the variance between groups
# is roughly equal
check_variance <- function(formula, data) {
  # Bartlett test: Check homogeneity of variances based on the mean
  bartlett_result <- bartlett.test(formula, data = data) |> tidy()
  
  # Levene test: Check homogeneity of variances based on the median, so it’s more robust to outliers
  levene_result <- car::leveneTest(formula, data = data) |>
    tidy() |>
    mutate(method = 'levene test')
  
  # Fligner-Killeen test: Check homogeneity of variances based on the median, so it’s more robust to outliers
  fligner_result <- fligner.test(formula, data = data) |> tidy()
  
  # Kruskal-Wallis test: Check homogeneity of distributions nonparametrically
  kruskal_result <- kruskal.test(formula, data = data) |> tidy()
  
  # Combine results into a data frame
  results_df <- bind_rows(bartlett_result, levene_result, fligner_result, kruskal_result, .id = "test") 
  
  return(results_df)
}


get_cols_between <- function(df, col_start, col_end) {
  col_idx <- match(col_start, names(df)):match(col_end, names(df))
  df |>
    # select(({{col_start}}:{{col_end}})) also works
    select(all_of(col_idx)) %>%
    names()
}

unique_prop <- function(x) {
  unique_count <- dplyr::n_distinct(x)
  prop <- unique_count / length(x)
  return(prop)
}

# convert the multi-item survey questions to wide, e.g. survey_multi_wide(exp1_df)
survey_multi_wide <- function(df){
  df <- df |> 
    select(uuid, user_view_general:rely_info_face) |>
    pivot_longer(
      cols = -c(uuid),
      names_to = c("variable", ".value"), names_pattern = "(.*)_(.*)"
    )
  return(df)
}

# so you can use this in a summarise call
summ_lst <- list(mean = mean, sd = sd, min = min, max = max, unique_count = dplyr::n_distinct, unique_prop = unique_prop)
```

```r
dem_study_df <- read_csv(file = '/content/prolific_demographic_study.csv', show_col_types = FALSE) |>
  rename_with(~str_to_lower(str_replace_all(., " ", "_")), everything()) |>
  # remove people that did not complete the study, e.g. withdrew consent
  filter(status == 'APPROVED') |>
  # if consent is revoked, age will be a string for that participant
  # removed it so now it should be fine to convert to numeric
  mutate(age = as.numeric(age)) |>
  # removing columns that are not needed
  select(-c(submission_id, reviewed_at, archived_at, student_status:employment_status)) |>
  # create a uuid so you can use that instead of prolific id
  group_by(participant_id) |>
  mutate(uuid = uuid::UUIDgenerate(use.time = FALSE)) |>
  ungroup()

# additional demographic data from the study, e.g. not directly from prolific
exp1_df <- read_csv(file = '/content/exp1_choice_data.csv', show_col_types = FALSE, col_names = FALSE) |>
  # not using the column names so I do not get output of readr changing the column names
  # since the original csv had weird namings
  filter(X5 == '100') |>
  select(
    # renaming them here
    PROLIFIC_PID = X240,
    consent = X18, # Q1...18
    prolific_id = X19, # Q1...19
    which_country = X20, # Q2...20
    age_years = X21, # Q3
    ethnicity_q = X22 # Q4
  ) |>
  # just to check if the values are the same as prolifics
  mutate(
    which_country = case_when(
      which_country == 1 ~ 'United States',
      which_country == 2 ~ 'United Kingdom',
      which_country == 3 ~ 'Australia',
      which_country == 4 ~ 'Prefer not to say'),
    age_years = case_when(
      age_years == 1 ~ '18 - 30',
      age_years == 2 ~ '30 - 45',
      age_years == 3 ~ '45 - 65',
      age_years == 4 ~ '65+',
      age_years == 5 ~ 'Prefer not to say')
  )

# join the two df
dem_study_df <- dem_study_df |>
  left_join(exp1_df, by = c('participant_id' = 'prolific_id')) |>
  select(-PROLIFIC_PID) |>
  relocate(uuid)

write_csv(x = dem_study_df, file = '/content/demographic_data.csv')
```

```r
### Study Data ----

# to remove pilot data, join the demographic data, then you can drop rows without a uuid

exp1_df <- suppressMessages(
  # supressing the output of the names readr creates when reading the df
  read_csv(file = '/content/exp1_choice_data.csv', show_col_types = FALSE)
  ) |>
  left_join(dem_study_df |> select(PROLIFIC_PID = participant_id, uuid), by = join_by(PROLIFIC_PID)) |>
  # remove participants from pilot
  filter(!is.na(uuid)) |>
  rename_columns() |>
  # add underscore between words, change to lowercase, and replace ... with _
  rename_with(~ str_to_lower(str_replace_all(str_replace_all(., "([a-z0-9])([A-Z])", "\\1_\\2"), "\\.\\.\\.", "_")), everything()) |>
  relocate(uuid) |>
  # safe to make numeric
  # will get a warning since NA values are present for trials a participant did not experience
  mutate(across(q1_1_25:q16_2_216, as.numeric))

# the df is currently wide, e.g. 40 rows by 232 columns, now making it long
# also renaming columns of the trials
# cond_ranges is a list of the different ranges of columns and their associated
# condition, e.g. FD1 is from columns q1_1_25 to q16_2_56
exp_df_long <- map(names(cond_ranges), function(cond) {
  # for the condition, select the start and end range
  range_cols <- cond_ranges[[cond]]
  cols <- get_cols_between(exp1_df, range_cols[1], range_cols[2])
  # cols <- get_col_names2(exp1_df, range_cols[1], range_cols[2])
  exp1_df |>
    #slice(-c(1:2)) |>
    select(uuid, all_of(cols)) |>
    pivot_longer(cols = -uuid, names_to = "question", names_prefix = "Q", values_to = "value") |>
    mutate(condition = cond)
}) |>
  bind_rows() |>
  mutate(
    # first or second question
    which_question = str_extract(question, '(?<=_)[0-9]+(?=_)') |> as.factor(),
    # e.g. trial_n
    question_n = str_extract(question, '(?<=q)\\d+') |> as.factor()
  ) |>
  # na values here are just the conditions that the part did not do
  filter(!is.na(value))

write_csv(x = exp1_df, file = '/content/experiment_data_wide.csv')
write_csv(x = exp_df_long, file = '/content/exp_df_long.csv')
```

```r
# only experiment result where each row is a single trial
# e.g. for each trial there is the value for both questions, e.g.
# uuid, condition, question_n, which_question, quest1_value, question_version2, quest2_value

exp_df_long1 <- exp_df_long |>
  filter(which_question == 1) |>
  select(uuid, condition, question_n, which_question, quest1_value = value) |>
  left_join(
    exp_df_long |>
      filter(which_question == 2) |>
      select(uuid, condition, question_n, question_version2 = which_question, quest2_value = value),
    by = join_by(uuid, condition, question_n)
  )

# should be 1280 rows, e.g. 16 trials * 2 questions * 40 participants
# thus, one trial per row
write_csv(x = exp_df_long1, file = '/content/experiment_data_by_trial.csv')
```


```r
exp_by_trial_path <- parse_drive(id = '1YR11qI6J9OXZFqRy6iG-kmjpiSup73re')
exp_long_path <- parse_drive(id = '11G-Exe4VTLxzLWWFufpi6qBISCA6MKek')
exp_wide_path <- parse_drive(id = '1MEl8BhB7TtAnAZ9NoCc9mDkM8Z8aGKyW')

exp_df_long <- read_csv(file = exp_long_path, show_col_types = FALSE)
exp_df_wide <- read_csv(file = exp_wide_path, show_col_types = FALSE)
exp_by_trial <- read_csv(file = exp_by_trial_path, show_col_types = FALSE)
```

```r
# list of the unique uuid for fair_dark and fair_light
cond_lst <- list(
  # fair dark
  fd = exp_df_long |>
    filter(condition %in% c('FD1', 'FD2')) |>
      distinct(uuid) |>
      pull(),
  # fair light
  fl = exp_df_long |>
    filter(condition %in% c('FL1', 'FL2')) |>
    distinct(uuid) |>
    pull()
)

df <- exp_by_trial |>
  participant_condition(id_col = uuid, id_lst = cond_lst) |>
  # adjust for the counterbalanced conditions
  reverse_conditions(cond_col = condition, q1_val = quest1_value, q2_val = quest2_value) |>
  # order of conditions
  get_model_cond(column_name = condition) |>
  # label for the condition
  cond_label(column_name = conditions_new) |>
  # which model was selected for the trial
  preferred_model(value = quest1_value, type = type) |>
  # bin confidence value into quartiles
  mutate(confidence = cut(quest2_value, breaks = 4, labels = c('low', 'below_avg', 'above_avg', 'high')))

df1 <- df |>
  select(uuid, question_n, exp_cond, conditions_new, cond1:confidence)

write_csv(df1, 'trial_data.csv')
```

Fair Light (Fair) Condition: Fair = 131, Light = 182, Total = 313  
Fair Light (Biased) Condition: Light = 132, Fair = 171, Total = 303

Fair Dark (Fair) Condition: Fair = 162, Dark = 145, Total = 307  
Fair Dark (Biased) Condition: Light = 142, Dark = 165, Total = 307

```r
binom.test(x = 131, n = 313, p = 0.5, alternative = c("two.sided"), conf.level = 0.95)
binom.test(x = 132, n = 303, p = 0.5, alternative = c("two.sided"), conf.level = 0.95)
binom.test(x = 162, n = 307, p = 0.5, alternative = c("two.sided"), conf.level = 0.95)
binom.test(x = 142, n = 307, p = 0.5, alternative = c("two.sided"), conf.level = 0.95)
```
