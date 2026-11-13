# Cortex curves and key traits from Rooh's cortex_pattern_<date>.xlsx (sheet Curves).
#
#   Rscript R/plot_curves.R <cortex_pattern.xlsx> [design.xlsx] [group column]
#
# design.xlsx (optional): one row per image, column image_file plus anything you like (genotype, treatment, ...).
#   Matched on the image's file name, so a path that differs (Maize/sibs/P1/x.tif vs sibs/P1/x.tif) still joins.
#   A plant column there replaces Rooh's. Its columns win over same-named ones in the Curves sheet.
# group: the column that colours the plots (default: the design's first column after image_file / plant).
#
# The plant is the unit: images -> plant mean, then group mean +- SE over plants. Writes PNGs beside the input,
# in R_figures/.

suppressPackageStartupMessages({
  library(readxl); library(dplyr); library(tidyr); library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) stop("Usage: Rscript R/plot_curves.R <cortex_pattern.xlsx> [design.xlsx] [group]")
curves_file <- args[1]
design_file <- if (length(args) >= 2 && nzchar(args[2])) args[2] else NA
group       <- if (length(args) >= 3) args[3] else NA

cv <- read_excel(curves_file, sheet = "Curves", guess_max = 1e5)
unit <- if (any(grepl("_um2$", names(cv)))) "µm²" else "px²"
area <- grep("^mean_area_", names(cv), value = TRUE)

# ---- design -------------------------------------------------------------------------------------------------
if (!is.na(design_file)) {
  ds <- read_excel(design_file, guess_max = 1e5) |> mutate(key = basename(image_file), image_file = NULL)
  if (anyDuplicated(ds$key)) stop("Two images in the design share a file name: ", ds$key[duplicated(ds$key)][1])
  cv <- cv |> mutate(key = basename(image_file)) |>
    select(-any_of(setdiff(names(ds), "key"))) |>
    left_join(ds, by = "key") |> select(-key)
  if (is.na(group)) group <- setdiff(names(ds), c("key", "plant"))[1]
  miss <- unique(cv$image_file[is.na(cv[[group]])])
  if (length(miss)) message(length(miss), " image(s) not in the design (left out): ", paste(head(miss, 3), collapse = ", "))
  cv <- filter(cv, !is.na(.data[[group]]))
}
if (is.na(group)) {   # no design: the last column Rooh put before plant (a folder level)
  ids <- names(cv)[2:(which(names(cv) == "plant") - 1)]
  group <- if (length(ids)) tail(ids, 1) else "plant"
}
cv[[group]] <- factor(cv[[group]])
cat("Grouping by", group, ":", paste(levels(cv[[group]]), collapse = ", "), "\n")

out <- file.path(dirname(curves_file), "R_figures")
dir.create(out, showWarnings = FALSE)
se <- function(x) sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x)))
theme_set(theme_classic(base_size = 12) + theme(strip.background = element_blank(), legend.position = "top"))
save <- function(p, name, w = 6, h = 4.5) ggsave(file.path(out, name), p, width = w, height = h, dpi = 200)

# ---- 1. cell size from the stele (0) to the epidermis (1) ----------------------------------------------------
plant_curve <- cv |> group_by(.data[[group]], plant, r_norm) |>
  summarise(area = mean(.data[[area]], na.rm = TRUE), .groups = "drop")
grp_curve <- plant_curve |> group_by(.data[[group]], r_norm) |>
  summarise(m = mean(area), se = se(area), n = n(), .groups = "drop")
p1 <- ggplot(grp_curve, aes(r_norm, m, colour = .data[[group]], fill = .data[[group]])) +
  geom_line(data = plant_curve, aes(y = area, group = plant), alpha = 0.25, linewidth = 0.4) +
  geom_ribbon(aes(ymin = m - se, ymax = m + se), alpha = 0.2, colour = NA) +
  geom_line(linewidth = 1) + geom_point(size = 1.8) +
  labs(x = "Position across the cortex (0 = stele, 1 = epidermis)", y = paste0("Mean cell area (", unit, ")"),
       colour = NULL, fill = NULL,
       caption = paste0("Thin lines: plants. Mean ± SE over plants (n = ",
                        paste(tapply(grp_curve$n, grp_curve[[group]], max), collapse = ", "), ")."))
save(p1, "1_cell_area_by_position.png")

# ---- 2. the same per quarter ---------------------------------------------------------------------------------
q_curve <- cv |> group_by(.data[[group]], plant, quarter, r_norm) |>
  summarise(area = mean(.data[[area]], na.rm = TRUE), .groups = "drop") |>
  group_by(.data[[group]], quarter, r_norm) |> summarise(m = mean(area), se = se(area), .groups = "drop")
p2 <- ggplot(q_curve, aes(r_norm, m, colour = .data[[group]], fill = .data[[group]])) +
  geom_ribbon(aes(ymin = m - se, ymax = m + se), alpha = 0.2, colour = NA) + geom_line(linewidth = 0.8) +
  facet_wrap(~quarter, nrow = 1) +
  labs(x = "Position across the cortex (0 = stele, 1 = epidermis)", y = paste0("Mean cell area (", unit, ")"),
       colour = NULL, fill = NULL)
save(p2, "2_cell_area_by_position_quarters.png", w = 11, h = 4)

# ---- 3. key traits, one point per plant ----------------------------------------------------------------------
traits <- intersect(c("ccfn", "cortex_width_um", "cortex_width_px", "file1_cell_area_um2", "file1_cell_area_px2",
                      "median_cortex_cell_area_um2", "median_cortex_cell_area_px2", "aerenchyma_share",
                      "root_diameter_um", "root_diameter_px", "stele_diameter_um", "stele_diameter_px",
                      "stele_area_um2", "stele_area_px2", "mx_n", "mx_area_mean_um2", "mx_area_mean_px2"), names(cv))
label <- c(ccfn = "Cortical cell files", cortex_width = "Cortex width", file1_cell_area = "Innermost file cell area",
           median_cortex_cell_area = "Median cortical cell area", aerenchyma_share = "Aerenchyma share",
           root_diameter = "Root diameter", stele_diameter = "Stele diameter", stele_area = "Stele area",
           mx_n = "Metaxylem vessels", mx_area_mean = "Mean metaxylem area")
plant_traits <- cv |> distinct(image_file, quarter, .keep_all = TRUE) |>   # quarter traits: one row per quarter
  group_by(.data[[group]], plant, image_file) |> summarise(across(all_of(traits), \(x) mean(x, na.rm = TRUE)), .groups = "drop") |>
  group_by(.data[[group]], plant) |> summarise(across(all_of(traits), \(x) mean(x, na.rm = TRUE)), .groups = "drop") |>
  pivot_longer(all_of(traits), names_to = "trait") |>
  mutate(unit = sub(".*_(um2?|px2?)$", "\\1", trait), unit = ifelse(unit == trait, "", paste0(" (", sub("um", "µm", sub("2", "²", unit)), ")")),
         trait = paste0(label[sub("_(um2?|px2?)$", "", trait)], unit))
p3 <- ggplot(plant_traits, aes(.data[[group]], value, colour = .data[[group]])) +
  geom_jitter(width = 0.12, height = 0, alpha = 0.6, size = 1.8) +
  stat_summary(fun = mean, fun.min = \(x) mean(x) - se(x), fun.max = \(x) mean(x) + se(x), colour = "black", size = 0.4) +
  facet_wrap(~trait, scales = "free_y") + labs(x = NULL, y = NULL, colour = NULL) +
  theme(legend.position = "none", axis.text.x = element_text(angle = 30, hjust = 1))
save(p3, "3_key_traits_per_plant.png", w = 10, h = 7)

cat("Wrote", out, "\n")
