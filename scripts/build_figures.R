args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grep("^--file=", args)][1])
root <- normalizePath(file.path(dirname(script), ".."), mustWork = TRUE)
setwd(root)
source("scripts/figure_style.R")
dir.create("figures", recursive = TRUE, showWarnings = FALSE)
python <- Sys.getenv("PYTHON", "python3")
stopifnot(system2(python, shQuote(c("scripts/render_architecture.py",
          "figures/architecture.svg", "figures/architecture.pdf"))) == 0)
library(jsonlite)
library(gridExtra)

seed_colours <- c("0" = "#277d9f", "1" = "#cc692f", "2" = "#55934c")
seed_shapes <- c("0" = 16, "1" = 15, "2" = 18)
background_colours <- c("setty" = "#7b2d8e", "dentate" = "#1f4e79", "endo" = "#2e7d4f", "lung" = "#8a4b08")
plot_counts <- list()

read_json <- function(path) fromJSON(path, flatten = TRUE)
historical <- read.csv("scripts/inputs/historical_plot_rows.csv", na.strings = c("", "nan", "NA"))
historical$seed <- factor(historical$seed, levels = 0:2)
grid_rows <- read.csv("evidence/historical_rows.csv", na.strings = c("", "nan", "NA"))

record <- function(name, count) {
  plot_counts[[name]] <<- count
}

source("scripts/figure_composition.R")

seed_scales <- function() list(scale_colour_manual(values = seed_colours, name = "Seed"),
                               scale_shape_manual(values = seed_shapes, name = "Seed"))

dose_summary <- function(rows, metric) {
  groups <- split(rows, interaction(rows$dataset, rows$value, rows$n_components, drop = TRUE))
  do.call(rbind, lapply(groups, function(group) {
    values <- group[[metric]]
    valid <- all(is.finite(values))
    data.frame(dataset = group$dataset[1], value = group$value[1],
               capacity = paste0("K = ", group$n_components[1]),
               mean = if (valid) mean(values) else NA_real_,
               sd = if (valid) sd(values) else NA_real_)
  }))
}

dose_plot <- function(rows, metric, title, ylabel, xlabel = "Concentration alpha") {
  data <- dose_summary(rows, metric)
  data <- data[is.finite(data$mean), ]
  ggplot(data, aes(value, mean, colour = dataset, shape = dataset,
                   group = interaction(dataset, capacity), linetype = capacity)) +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.04, linewidth = 0.4) +
    geom_line(linewidth = 0.65) + geom_point(size = 2) +
    geom_vline(xintercept = 1, linetype = "dashed", colour = "black", linewidth = 0.35) +
    scale_x_log10(breaks = sort(unique(rows$value))) +
    scale_colour_manual(values = background_colours, name = "Background") +
    labs(title = title, x = xlabel, y = ylabel, shape = "Background", linetype = "Capacity") +
    guides(colour = guide_legend(nrow = 1), shape = guide_legend(nrow = 1))
}

umap_figure <- function(directory, name, doses, titles, height) {
  files <- sort(list.files(directory, pattern = "\\.csv$", full.names = TRUE))
  data <- do.call(rbind, lapply(files, read.csv))
  data$label <- factor(data$label, levels = 0:9)
  label_colours <- c("#1f4e79", "#2e7d4f", "#b5473f", "#8a4b08", "#7b2d8e",
                     "#0e7c7b", "#c27c0e", "#4a6fa5", "#6b6b6b", "#a34d7a")
  plots <- list()
  for (dose_index in seq_along(doses)) {
    for (seed_index in 0:2) {
      panel <- subset(data, dose == doses[dose_index] & seed == seed_index)
      stopifnot(nrow(panel) == 450)
      half_span <- max(diff(range(panel$x)), diff(range(panel$y))) * 0.55
      x_limits <- mean(range(panel$x)) + c(-half_span, half_span)
      y_limits <- mean(range(panel$y)) + c(-half_span, half_span)
      plots[[length(plots) + 1]] <- ggplot(panel, aes(x, y, colour = label)) +
        geom_point(size = 0.4, alpha = 0.85, stroke = 0) +
        scale_colour_manual(values = label_colours, drop = FALSE, name = "Stored label") +
        coord_equal(xlim = x_limits, ylim = y_limits, expand = FALSE) + labs(x = "UMAP 1", y = "UMAP 2",
                             title = NULL) +
        guides(colour = guide_legend(nrow = 1, override.aes = list(size = 2, alpha = 1)))
    }
  }
  record(name, nrow(data))
  save_figure(plots, name, ncol = 3, width = 7.2, height = height, shared_legend = TRUE, base_size = 9,
              row_labels = titles, column_labels = paste("Seed", 0:2))
}

alpha_rows <- subset(grid_rows, axis == "alpha")
save_figure(list(dose_plot(alpha_rows, "allocation_coherence", "Allocation coherence", "Exact-pair coherence"),
                 dose_plot(alpha_rows, "allocation_perplexity_mean", "Allocation perplexity", "Effective topic count")),
            "topic-transformer_dose", width = 7.8, height = 3.25)
record("topic-transformer_dose", nrow(alpha_rows))
branch_rows <- subset(historical, lock == "topic-transformer-prior-20260907" & dataset == "setty" & axis == "alpha")
save_figure(list(dose_plot(branch_rows, "branch_knn_mean_r2", "Setty branch prediction", "Palantir kNN R²")),
            "topic-transformer_branch", ncol = 1, height = 2.6)
umap_figure("scripts/inputs/umap400", "topic-transformer_setty_umap_400",
            c("alpha=1", "control"), c("Topic; 400 cap", "Gaussian; 400 cap"), 5.15)

old <- subset(historical, lock == "topic-transformer-prior-20260907" & axis == "alpha")
old$budget <- "200"
long <- subset(historical, lock == "topic-variants-alpha-1000ep-fourbg-20260908" & axis == "alpha")
long$budget <- "1000"
middle <- subset(historical, lock == "topic-transformer-alpha0p1-400ep-setty-20260908" & axis == "alpha" & value == 0.1)
middle$budget <- "400"
data <- rbind(old, long, middle)
data <- subset(data, value %in% c(0.1, 0.5, 1, 5))
data$x <- log10(data$value) + c("200" = -0.04, "400" = 0, "1000" = 0.04)[data$budget]
data$budget <- factor(data$budget, levels = c("200", "400", "1000"))
budget_colours <- c("200" = "#5d6d7e", "400" = "#c27c0e", "1000" = "#277d9f")
plots <- list()
for (metric in c("allocation_coherence", "allocation_perplexity_mean")) {
  for (dataset_name in c("setty", "dentate", "endo", "lung")) {
    panel <- subset(data, dataset == dataset_name)
    if (metric == "allocation_coherence") panel <- subset(panel, budget != "400" & is.finite(allocation_coherence) & allocation_degenerate != "True")
    plot <- ggplot(panel, aes(x, .data[[metric]], colour = budget, shape = seed)) +
      geom_point(size = 2, show.legend = TRUE) +
      scale_colour_manual(values = budget_colours, limits = names(budget_colours), drop = FALSE, name = "Cap (epochs)") +
      scale_shape_manual(values = seed_shapes, name = "Seed") +
      scale_x_continuous(breaks = log10(c(0.1, 0.5, 1, 5)), labels = c("0.1", "0.5", "1", "5"),
                         limits = c(-1.2, 0.87)) +
      coord_cartesian(ylim = if (metric == "allocation_coherence") c(-0.05, 1.05) else c(-0.2, 8.5)) +
      labs(title = tools::toTitleCase(dataset_name), x = "Concentration alpha",
            y = if (metric == "allocation_coherence") "Exact-pair coherence" else "Allocation perplexity") +
      guides(colour = guide_legend(nrow = 1))
    if (metric == "allocation_coherence" && dataset_name == "setty") {
      plot <- plot +
        annotate("rect", xmin = -1.18, xmax = -0.82, ymin = -Inf, ymax = Inf, fill = "#f5b7b1", alpha = 0.3) +
        annotate("text", x = -1.14, y = 0.63, hjust = 0, label = "3/3 degenerate\n200/400/1000\nNo ratio", size = 3.1, family = "Arial", colour = "black")
    }
    plots[[length(plots) + 1]] <- plot
  }
}
save_figure(plots, "collapse_alpha", ncol = 4, width = 8.0, height = 4.8, shared_legend = TRUE, base_size = 10,
            row_labels = c("Exact-pair allocation coherence", "Mean allocation perplexity"),
            column_labels = c("Setty", "Dentate", "Endo", "Lung"))
record("collapse_alpha", c(input_rows = nrow(data), coherence_points = sum(is.finite(data$allocation_coherence) & data$budget != "400"),
                          perplexity_points = nrow(data)))

data <- subset(historical, dataset == "setty" &
                 lock %in% c("topic-transformer-prior-20260907", "topic-variants-400ep-setty-20260907",
                             "topic-variants-alpha-1000ep-fourbg-20260908") &
                 (axis == "alpha" | axis == "control") & is.finite(reachability_survival))
data$budget <- ifelse(data$lock == "topic-transformer-prior-20260907", "200",
                      ifelse(data$lock == "topic-variants-400ep-setty-20260907", "400", "1000"))
control_x <- log10(5) + 0.55
data$x <- ifelse(data$axis == "control", control_x, log10(data$value)) +
  c("200" = -0.045, "400" = 0, "1000" = 0.045)[data$budget]
data$budget <- factor(data$budget, levels = c("200", "400", "1000"))
plot <- ggplot(data, aes(x, reachability_survival, colour = seed, shape = seed, size = budget)) +
  geom_point() +
  scale_colour_manual(values = seed_colours, name = "Seed") +
  scale_shape_manual(values = seed_shapes, name = "Seed") +
  scale_size_manual(values = c("200" = 1.7, "400" = 2.3, "1000" = 3), name = "Cap (epochs)") +
  geom_hline(yintercept = 0.8376639445681762, linetype = "dashed", colour = "black", linewidth = 0.4) +
  annotate("text", x = -1, y = 0.87, label = "0.837664", hjust = 0, colour = "black", family = "Arial", size = 3.3) +
  annotate("text", x = 0.43, y = 0.935, label = "0.841336\n200 epochs; alpha 1; seed 0", hjust = 0, colour = "black", family = "Arial", size = 3.3) +
  annotate("segment", x = 0.43, xend = -0.045, y = 0.915, yend = 0.8413363029, colour = "black", linewidth = 0.3,
           arrow = arrow(length = unit(0.07, "in"))) +
  scale_x_continuous(breaks = c(log10(c(0.1, 0.5, 1, 2, 5)), control_x), labels = c("0.1", "0.5", "1", "2", "5", "Control")) +
  coord_cartesian(ylim = c(0.80, 1.06)) +
  labs(title = "Setty reachability across concentrations and budgets", x = "Concentration alpha / Gaussian control", y = "Reachability survival") +
  guides(colour = guide_legend(nrow = 1, byrow = TRUE),
         shape = guide_legend(nrow = 1), size = guide_legend(nrow = 1))
save_figure(list(plot), "tear_vs_alpha", ncol = 1, width = 7.8, height = 3.1)
record("tear_vs_alpha", nrow(data))

training <- read_json("evidence/training_results.json")
training$seed <- factor(training$seed)
plots <- list()
for (metric in c("mean_perplexity", "max_coordinate_range")) {
  for (dataset_name in c("setty", "dentate")) {
    plot <- ggplot(subset(training, dataset == dataset_name), aes(alpha, .data[[metric]], colour = seed, shape = seed, group = seed)) +
      geom_line(linewidth = 0.6) + geom_point(size = 2) + seed_scales() +
      scale_x_log10(breaks = c(0.1, 1), labels = c("0.1", "1")) +
      labs(title = tools::toTitleCase(dataset_name), x = "Concentration alpha",
           y = if (metric == "mean_perplexity") "Mean allocation perplexity" else "Maximum coordinate range")
    if (metric == "max_coordinate_range") {
      plot <- plot + scale_y_log10() +
        geom_hline(yintercept = 1e-8, colour = "black", linetype = "dashed", linewidth = 0.4) +
        coord_cartesian(ylim = c(1e-14, 2))
    } else {
      plot <- plot + coord_cartesian(ylim = c(0.95, 2))
    }
    plots[[length(plots) + 1]] <- plot
  }
}
save_figure(plots, "matched_training", width = 7.6, height = 4.9, shared_legend = TRUE)
record("matched_training", nrow(training) * 2)

probes <- subset(read_json("evidence/probe_results.json"), mode == "eval")
probes$alpha <- factor(probes$alpha)
probes$group <- interaction(probes$dataset, probes$seed, probes$alpha)
invariant <- subset(probes, intervention %in% c("reverse_tokens", "alter_companions", "reverse_batch", "isolated"))
invariant$x <- match(invariant$intervention, c("reverse_tokens", "alter_companions", "reverse_batch", "isolated")) + (invariant$seed - 1) * 0.07
invariant$value <- pmax(1e-16, invariant$max_abs)
left <- ggplot(invariant, aes(x, value, colour = dataset, shape = alpha)) +
  geom_point(size = 2, alpha = 0.7) +
  geom_hline(yintercept = 1e-5, linetype = "dashed", colour = "black", linewidth = 0.4) +
  scale_y_log10() + coord_cartesian(ylim = c(3e-17, 1e-4)) +
  scale_x_continuous(breaks = 1:4, labels = c("Token\norder", "Companions", "Batch\norder", "Isolated")) +
  labs(title = "Invariance controls", x = "Intervention", y = "Maximum allocation change")
content <- subset(probes, intervention %in% c("mean_tokens", "duplicate_first"))
content$x <- match(content$intervention, c("mean_tokens", "duplicate_first"))
right <- ggplot(content, aes(x, mean_l1, colour = dataset, shape = alpha, group = group)) +
  geom_line(linewidth = 0.5, alpha = 0.65) + geom_point(size = 2, alpha = 0.7) +
  scale_x_continuous(breaks = 1:2, labels = c("Mean tokens", "Duplicate first"), expand = expansion(mult = 0.18)) +
  labs(title = "Content-changing interventions", x = "Intervention", y = "Mean allocation L1 change")
plots <- lapply(list(left, right), function(plot) plot +
  scale_colour_manual(values = c("setty" = "#267fa5", "dentate" = "#cb6430"), name = "Background") +
  scale_shape_manual(values = c("0.1" = 16, "1" = 15), name = "Alpha"))
save_figure(plots, "token_probes", width = 7.8, height = 3.2, shared_legend = TRUE)
record("token_probes", c(invariance_points = nrow(invariant), content_points = nrow(content)))
write_json(plot_counts, "scripts/figure-data-counts.json", pretty = TRUE, auto_unbox = TRUE)
