save_figure <- function(plots, name, ncol = 2, width = 7.2,
                        height = 3.1 * ceiling(length(plots) / ncol), shared_legend = TRUE,
                        base_size = 10, row_labels = NULL, column_labels = NULL) {
  measure_file <- tempfile(fileext = ".pdf")
  cairo_pdf(measure_file, width = width, height = height, family = "Arial")
  on.exit({ dev.off(); unlink(measure_file) }, add = TRUE)
  row_count <- ceiling(length(plots) / ncol)
  styled <- lapply(seq_along(plots), function(index) {
    panel <- plots[[index]] + paper_theme(base_size) +
      theme(legend.box = "horizontal", legend.text = element_text(size = base_size - 0.5),
            legend.title = element_text(size = base_size - 0.5),
            legend.key.width = unit(0.8, "lines"),
            legend.spacing.x = unit(0.12, "cm"), legend.box.spacing = unit(0.1, "cm"))
    if (is.null(row_labels) && length(plots) > 1) panel <- panel + labs(tag = LETTERS[index])
    if (!is.null(row_labels)) {
      panel <- panel + labs(title = NULL)
      if ((index - 1) %% ncol != 0) panel <- panel + labs(y = NULL)
    }
    panel
  })
  legend_sources <- if (shared_legend) 1 else seq_along(styled)
  legends <- lapply(legend_sources, function(index) {
    grob <- ggplotGrob(styled[[index]])
    candidates <- which(grepl("^guide-box", grob$layout$name))
    candidates <- candidates[vapply(grob$grobs[candidates], inherits, logical(1), "gtable")]
    if (length(candidates)) grob$grobs[[candidates[1]]] else NULL
  })
  legends <- Filter(Negate(is.null), legends)
  legend_heights <- lapply(legends, function(grob) grobHeight(grob) + unit(0.02, "in"))
  legend_height <- if (length(legends)) Reduce(`+`, legend_heights) else unit(0, "in")
  legend <- if (length(legends)) arrangeGrob(grobs = legends, ncol = 1,
    heights = do.call(unit.c, legend_heights)) else NULL
  panels <- lapply(styled, function(panel) ggplotGrob(panel + theme(legend.position = "none")))
  common_widths <- do.call(unit.pmax, lapply(panels, function(panel) panel$widths))
  panels <- lapply(panels, function(panel) { panel$widths <- common_widths; panel })
  pieces <- list()
  sizes <- list()
  add_piece <- function(grob, size) {
    pieces[[length(pieces) + 1]] <<- grob
    sizes[[length(sizes) + 1]] <<- size
  }
  if (!is.null(column_labels)) add_piece(
    arrangeGrob(grobs = lapply(column_labels, textGrob,
      gp = gpar(fontfamily = "Arial", fontsize = base_size + 1, col = "black")), ncol = ncol),
    unit(0.20, "in"))
  for (row_index in seq_len(row_count)) {
    if (!is.null(row_labels)) add_piece(grobTree(
      textGrob(LETTERS[row_index], x = unit(0.08, "in"), just = "left",
               gp = gpar(fontfamily = "Arial", fontface = "bold", fontsize = base_size + 2, col = "black")),
      textGrob(row_labels[row_index],
               gp = gpar(fontfamily = "Arial", fontsize = base_size + 1, col = "black"))),
      unit(0.22, "in"))
    indices <- seq.int((row_index - 1) * ncol + 1, min(row_index * ncol, length(panels)))
    add_piece(arrangeGrob(grobs = panels[indices], ncol = ncol), unit(1, "null"))
    if (!is.null(legend) && row_index == max(1, floor(row_count / 2))) {
      add_piece(legend, legend_height + unit(0.02, "in"))
    }
  }
  figure <- arrangeGrob(grobs = pieces, ncol = 1, heights = do.call(unit.c, sizes))
  ggsave(file.path("figures", paste0(name, ".pdf")), figure, device = cairo_pdf,
         width = width, height = height, bg = "transparent", family = "Arial", limitsize = FALSE)
  ggsave(file.path("figures", paste0(name, ".png")), figure, device = "png",
         width = width, height = height, dpi = 150, bg = "white", limitsize = FALSE)
}
