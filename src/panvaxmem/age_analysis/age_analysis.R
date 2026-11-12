# This script was contributed by Maike Nägele.
#
# Analyze age-related differences in gene expression before and after vaccination:
# (1) Compare baseline M20 gene-set scores between young and older participants.
# (2) Track changes in JUN, FOS, and mean M20 gene expression at days 7 and 28
#     relative to each participant's baseline.
# (3) Compare the size and direction of M20 expression changes at day 28
#     between young and older participants.
#
# Data download (login required):
# https://datatools.immunespace.org/project/HIPC/IS2/begin.view?pageId=study.DATA_ANALYSIS
# Adapted from:
# https://github.com/RGLab/ImmuneSignatures2/blob/main/inst/preprocess/create_final_esets.Rmd


library(ImmuneSignatures2)
library(Biobase)
library(GSVA)
library(tidyr)
library(ggplot2)
library(dplyr)  

# paths
expression_eset_path <- "/mnt/virtual_drug_screen_2/data/HIPC/all_noNorm_eset.rds"
output_dir           <- "/mnt/virtual_drug_screen_2/results/HIPC_data/GitHub"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# studies filtered by: TIV vaccination, PBMC samples, young and old participants, data at day 0, 7 and 28
selected_studies <- c("SDY63", "SDY400", "SDY404")

# age groups
young_max <- 45
old_min   <- 65

# days counted as day 28
day28_days <- c(24, 28, 32, 35, 36)

# gene sets
gene_sets <- list(
  AP1_ATF = c("FOS", "FOSB", "JUN", "JUND"),
  M20     = c("ATF3", "CCL2", "EGR1", "FOS", "FOSB", "FOSL1", "FOSL2",
              "IFNG", "IL6", "IL8", "JUN", "JUNB", "MMP1", "MMP9", "PLAU")
)

# ssGSEA scores
# FALSE: scores are divided by the range of all scores, so they depend on which gene sets and samples are scored together
# TRUE:  raw scores, independent of other gene sets and samples
# p-values are the same
use_raw_scores <- FALSE

age_colors <- c(young = "#0072B2", old = "#D55E00")


# load data, keep selected studies and ages 0-90
eset <- readRDS(expression_eset_path)
eset <- eset[, pData(eset)$study_accession %in% selected_studies &
               !is.na(pData(eset)$age_imputed) & pData(eset)$age_imputed <= 90]
rownames(pData(eset)) <- pData(eset)$uid

# batch correction: remove study effects, keep sex
eset <- batchCorrect(eset, batch.vars = "study_accession", covariates = "y_chrom_present")

# keep genes measured in all samples
eset <- eset[complete.cases(exprs(eset)), ]


# samples: day 0, 7, 28, young/old, one sample per participant and day (real day 28 first)
samples <- pData(eset) %>%
  mutate(sample_id = sampleNames(eset),
         day       = ifelse(study_time_collected %in% day28_days, 28, study_time_collected),
         age_group = factor(case_when(age_reported <= young_max ~ "young",
                                      age_reported >= old_min   ~ "old"),
                            levels = c("young", "old"))) %>%
  filter(day %in% c(0, 7, 28), !is.na(age_group)) %>%
  arrange(participant_id, day, abs(study_time_collected - day)) %>%
  distinct(participant_id, day, .keep_all = TRUE) %>%
  select(sample_id, participant_id, day, age_group)

# expression; gene sets limited to genes on the platform (IL8 = CXCL8)
expr      <- exprs(eset)[, samples$sample_id]
gene_sets <- lapply(gene_sets, function(g) intersect(sub("^IL8$", "CXCL8", g), rownames(expr)))

# log2FC vs. own day 0 (genes x samples)
# samples without day 0 are dropped
day0_samples <- filter(samples, day == 0)
fc_samples   <- filter(samples, participant_id %in% day0_samples$participant_id)
day0_match   <- day0_samples$sample_id[match(fc_samples$participant_id, day0_samples$participant_id)]
log2fc       <- expr[, fc_samples$sample_id] - expr[, day0_match]


# ssGSEA on all samples and gene sets together
# the score depends on which genes are in the expression matrix
scores      <- gsva(ssgseaParam(expr, gene_sets, minSize = 1, normalize = !use_raw_scores), verbose = FALSE)
samples$M20 <- scores["M20", samples$sample_id]


# plot: Is the M20 baseline level different between young and old? 

# day-0 scores of participants who also have a day-28 sample
day0 <- filter(samples, day == 0, participant_id %in% samples$participant_id[samples$day == 28])

# Wilcoxon test young vs. old
test <- wilcox.test(M20 ~ age_group, data = day0, exact = FALSE)
print(test)

y_top  <- max(day0$M20) + 0.12 * diff(range(day0$M20))
y_bot  <- min(day0$M20) - 0.12 * diff(range(day0$M20))
counts <- count(day0, age_group)

p <- ggplot(day0, aes(age_group, M20, fill = age_group)) +
  geom_boxplot(width = 0.55, outlier.shape = NA, alpha = 0.35) +
  geom_jitter(aes(color = age_group), width = 0.1, height = 0, size = 1, alpha = 0.6) +
  annotate("segment", x = 1, xend = 2, y = y_top, yend = y_top) +
  geom_text(data = counts, aes(age_group, y_bot, label = paste0("n = ", n)), inherit.aes = FALSE,
            size = 2.6, color = "grey40") +
  scale_fill_manual(values = age_colors) +
  scale_color_manual(values = age_colors) +
  labs(x = NULL, y = "Day-0 ssGSEA score",
       title = sprintf("Day 0 M20 | Wilcoxon p = %.3g", test$p.value)) +
  theme_classic(base_size = 11) +
  theme(legend.position = "none") +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5))


print(p)
ggsave(file.path(output_dir, "M20_day0_ssGSEA_boxplot.pdf"), p, width = 2.5, height = 3.9)


# plot: how JUN, FOS and M20 change after vaccination (day 7, 28) relative to each person's day 0

# log2FC of JUN, FOS and mean of the M20 genes
timecourse <- fc_samples %>%
  mutate(JUN = log2fc["JUN", ],
         FOS = log2fc["FOS", ],
         M20 = colMeans(log2fc[gene_sets$M20, ])) %>%
  pivot_longer(c(JUN, FOS, M20), names_to = "feature", values_to = "log2FC") %>%
  mutate(feature = factor(feature, levels = c("JUN", "FOS", "M20")))

# mean +/- SE per group and day; t-test vs. day 0, BH-adjusted 
# t-test: tests the mean shown in the plot
tc_summary <- timecourse %>%
  group_by(feature, age_group, day) %>%
  summarise(n    = n(),
            mean = mean(log2FC),
            se   = sd(log2FC) / sqrt(n()),
            p    = if (all(log2FC == 0)) NA else t.test(log2FC)$p.value,
            .groups = "drop") %>%
  mutate(p_adj = p.adjust(p, method = "BH"),
         stars = case_when(p_adj < 0.001 ~ "***", p_adj < 0.01 ~ "**", p_adj < 0.05 ~ "*", TRUE ~ ""))
print(tc_summary)

dodge <- position_dodge(width = 0.15)
p_tc <- ggplot(tc_summary, aes(factor(day), mean, color = age_group, group = age_group)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey60") +
  geom_line(linewidth = 1, position = dodge) +
  geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.05, position = dodge) +
  geom_point(size = 2.4, position = dodge) +
  geom_text(aes(y = mean + se, label = stars), vjust = -0.3, size = 4, position = dodge, show.legend = FALSE) +
  scale_color_manual(values = age_colors, name = NULL) +
  facet_wrap(~ feature, nrow = 1, scales = "free_y") +
  labs(x = "Day", y = "log2FC vs. day 0",
       title = "t-test vs. day 0, BH | * p < 0.05  ** < 0.01  *** < 0.001") +
  theme_classic(base_size = 11) +
  theme(legend.position = "bottom", strip.background = element_blank(),
        plot.title = element_text(size = 10, face = "bold", hjust = 0.5))

print(p_tc)
ggsave(file.path(output_dir, "timecourse_log2FC.pdf"), p_tc, width = 6, height = 3.8)



# plot: How strongly M20 genes change by day 28 (mean |log2FC| per person)

magnitude <- fc_samples %>%
  mutate(abs_fc = colMeans(abs(log2fc[gene_sets$M20, ]))) %>%
  filter(day == 28)

# Welch t-test young vs. old
welch <- t.test(abs_fc ~ age_group, data = magnitude, var.equal = FALSE)
print(welch)

# mean per group (label below the boxes) and bracket height
rng      <- diff(range(magnitude$abs_fc))
mean_lab <- magnitude %>%
  group_by(age_group) %>%
  summarise(mean_val = mean(abs_fc), .groups = "drop") %>%
  mutate(y = min(magnitude$abs_fc) - 0.25 * rng)
y_bracket <- max(magnitude$abs_fc) + 0.08 * rng

p_mag <- ggplot(magnitude, aes(age_group, abs_fc)) +
  geom_boxplot(width = 0.5, outlier.shape = NA, fill = "white", color = "black", linewidth = 0.6) +
  geom_jitter(width = 0.08, height = 0, size = 1, alpha = 0.45, color = "grey20") +
  geom_text(data = mean_lab, aes(age_group, y, label = sprintf("Mean = %.2f", mean_val)),
            inherit.aes = FALSE, size = 2.6, color = "#5A5A5A") +
  # red bracket + star only if p < 0.05
  { if (welch$p.value < 0.05) list(
      annotate("segment", x = 1, xend = 2, y = y_bracket, yend = y_bracket, color = "#D55E00", linewidth = 0.6),
      annotate("text", x = 1.5, y = y_bracket, label = "*", vjust = -0.2, size = 5, color = "#D55E00")) } +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.15))) +
  labs(x = NULL, y = "|log2FC(Day 28 / Day 0)|",
       title = sprintf("M20 magnitude d28")) +
  theme_classic(base_size = 11) +
  theme(axis.text = element_text(size = 9), legend.position = "none") +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5))


print(p_mag)
ggsave(file.path(output_dir, "M20_day28_response_magnitude_box.pdf"), p_mag, width = 3.2, height = 4.2, bg = "white")


# plot: How many people have M20 going up vs. down by day 28, and whether this split differs by age

# per participant: sign of the mean log2FC over the M20 genes
direction <- fc_samples %>%
  mutate(fc = colMeans(log2fc[gene_sets$M20, ])) %>%
  filter(day == 28) %>%
  mutate(sign = factor(ifelse(fc > 0, "positive", "negative"), levels = c("negative", "positive")))

# Fisher's exact test: age group x sign
fisher <- fisher.test(table(direction$age_group, direction$sign))
print(table(direction$age_group, direction$sign))
print(fisher)

# counts and concordance = max(n+, n-) / n per age group
counts <- count(direction, age_group, sign, .drop = FALSE)
conc   <- counts %>%
  group_by(age_group) %>%
  summarise(n_total = sum(n), concordance = max(n) / sum(n), .groups = "drop")
y_bracket <- 1.15 * max(conc$n_total)

p_conc <- ggplot(counts, aes(age_group, n, fill = sign)) +
  geom_col(width = 0.6, color = "black", linewidth = 0.3) +
  geom_text(data = conc, aes(age_group, n_total, label = sprintf("%.0f%%", 100 * concordance)),
            inherit.aes = FALSE, vjust = -0.5, size = 2.6, color = "grey30") +
  # red bracket + star only if p < 0.05
  { if (fisher$p.value < 0.05) list(
      annotate("segment", x = 1, xend = 2, y = y_bracket, yend = y_bracket, color = "#D55E00", linewidth = 0.5),
      annotate("text", x = 1.5, y = y_bracket, label = "*", vjust = -0.2, size = 5, color = "#D55E00")) } +
  scale_fill_manual(values = c(negative = "#3A76B4", positive = "#CC6A2C"), name = "log2FC sign") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.25))) +
  labs(x = NULL, y = "Number of subjects",
       title = sprintf("M20 direction d28 | Fisher p = %.3g", fisher$p.value)) +
  theme_classic(base_size = 11) +
  theme(axis.text = element_text(size = 9), legend.position = "top") +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5))


print(p_conc)
ggsave(file.path(output_dir, "M20_day28_directional_concordance.pdf"), p_conc, width = 3.4, height = 4.2, bg = "white")
