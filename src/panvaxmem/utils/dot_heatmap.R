library(circlize)
library(ComplexHeatmap)
library(grid)


draw_dot_heatmap <- function(
    values_path,
    sizes_path,
    legend_title,
    max_size_value = NULL,
    dotsize = 0.5,
    header_textsize = 12,
    row_textsize = 10,
    height_scale = 2.0,
    width_scale = 1.2,
    col_rot = 0,
    col_just = "center",
    legend_dir = "vertical",
    legend_width = unit(0.25, "npc"),
    cluster_rows = TRUE,
    show_row_dend = FALSE,
    cluster_columns = FALSE,
    show_column_dend = FALSE,
    column_name_gap = unit(4, "line")
) {
    values <- as.matrix(read.csv(values_path, check.names = FALSE, row.names = 1))
    sizes <- as.matrix(read.csv(sizes_path, check.names = FALSE, row.names = 1))
    if (!is.null(max_size_value)) {
        sizes <- pmin(sizes, max_size_value)
    }

    values_max <- ceiling(max(abs(values), na.rm = TRUE))
    col_fun <- colorRamp2(c(-values_max, 0, values_max), c("#4A6FA5", "#F7F7F7", "#B85C5C"))

    top_annotation <- HeatmapAnnotation(
        entry = anno_text(
            colnames(values),
            location = 0.5,
            rot = col_rot,
            just = col_just,
            gp = gpar(fontsize = header_textsize)
        ),
        height = column_name_gap
    )

    heatmap <- Heatmap(
        values,
        col = col_fun,
        na_col = "grey",
        rect_gp = gpar(type = "none"),
        show_row_dend = show_row_dend,
        cluster_rows = cluster_rows,
        show_column_dend = show_column_dend,
        cluster_columns = cluster_columns,
        show_column_names = FALSE,
        row_names_max_width = unit(30, "line"),
        column_title = NULL,
        top_annotation = top_annotation,
        column_dend_side = "bottom",
        cell_fun = function(j, i, x, y, width, height, fill) {
            grid.rect(
                x = x, y = y, width = width, height = height,
                gp = gpar(lwd = 0.4, col = "grey93", fill = NA, alpha = 1.0)
            )
            grid.circle(
                x = x, y = y, r = unit(sizes[i, j] * dotsize, "mm"),
                gp = gpar(fill = col_fun(values[i, j]), col = NA, alpha = 0.9)
            )
        },
        width = unit(dim(values)[2] * width_scale + 8, "line"),
        height = unit(dim(values)[1] * height_scale + 8, "line"),
        row_names_side = "left",
        row_names_gp = gpar(fontsize = row_textsize),
        show_heatmap_legend = FALSE
    )

    legend <- Legend(
        title = legend_title,
        col_fun = col_fun,
        direction = legend_dir,
        title_position = "topcenter",
        legend_width = legend_width,
        at = c(-values_max, 0, values_max)
    )

    return(list("heatmap" = heatmap, "legend" = legend, "values" = values, "sizes" = sizes))
}
