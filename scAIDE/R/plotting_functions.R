# Producing marker plots:


#' Prepares the data for plotting marker genes
#'
#' This function calculates the cluster-specific average gene expressions and the percentage of zeros in each cluster. Prepares the data for plotting
#'@param gene_cell_matrix This is the pre-processed gene expression matrix (normalized and logged). The subset expression from the markers list is used
#'@param cell_labels Cell labels which can be annotated or just cluster labels.
#'@export
#'@keywords process_marker_expression
#'@section Biological Analysis:
#'@examples
#'gene_expression_data <- read.csv("single_cell_dataset.csv", header = T, row.names = 1) # make sure that genes are in rows
#'cluster_vector <- read.table("results_from_rpkmeans.txt")$V1
#'
#'# example marker list:
#'selected_marker_genes <- c("SOX2", "ALDOC", "CCND2", "OLIG1", "OLIG2")
#'gene_expression_subset <- gene_expression_data[match(tolower(selected_marker_genes), tolower(rownmaes(gene_expression_data))), ]
#'processed_markers <- process_marker_expression(gene_expression_subset, cluster_vector)

process_marker_expression <- function(gene_cell_matrix, cell_labels){
  n_genes <- nrow(gene_cell_matrix)
  n_celltypes <- length(unique(cell_labels))
  percent_matrix <- matrix(0, n_genes, n_celltypes)
  average_matrix <- matrix(0, n_genes, n_celltypes)
  unique_labels <- unique(cell_labels)

  for(i in c(1:n_genes)){
    tmp_exprs <- gene_cell_matrix[i, ]
    #print(length(tmp_exprs))
    for(j in c(1:n_celltypes)){
      tmp_obs <- tmp_exprs[which(cell_labels==unique_labels[j])]
      s_percent <- length(which(tmp_obs==0)) / length(tmp_obs)

      s_average <- mean(as.numeric(tmp_obs))
      #str(tmp_obs)
      percent_matrix[i, j] <- s_percent
      average_matrix[i, j] <- s_average

    }
  }
  rownames(percent_matrix) <- rownames(gene_cell_matrix)
  colnames(percent_matrix) <- unique_labels
  rownames(average_matrix) <- rownames(gene_cell_matrix)
  colnames(average_matrix) <- unique_labels
  return(list(percent = percent_matrix, average = average_matrix))
}

#' Plot for visualizing cluster-specific markers
#'
#' This function plots the average expression values (denoted by color of dots) and the percentage of zeros in the cluster (denoted size of dots)
#'@param processed_list This is the pre-processed list from the function 'process_marker_expression', which returns an average expression matrix and the percentage matrix (ratio of zeros)
#'@param gene_levels This is a character vector containing gene names in the order that you want to display in the plot
#'@param cell_levels This is a character vector containing cell type annotations / cluster labels in the order that you want to display in the plot
#'@export
#'@keywords plot_marker_expressions
#'@section Biological Analysis:
#'@examples
#'gene_expression_data <- read.csv("single_cell_dataset.csv", header = T, row.names = 1) # make sure that genes are in rows
#'cluster_vector <- read.table("results_from_rpkmeans.txt")$V1
#'
#'# example marker list:
#'selected_marker_genes <- c("SOX2", "ALDOC", "CCND2", "OLIG1", "OLIG2")
#'gene_expression_subset <- gene_expression_data[match(tolower(selected_marker_genes), tolower(rownmaes(gene_expression_data))), ]
#'processed_markers <- process_marker_expression(gene_expression_subset, cluster_vector)
#'cell_levels <- unique(cluster_vector)
#'gene_levels <- selected_marker_genes
#'marker_plot <- plot_marker_expression(processed_markers, gene_levels=gene_levels, cell_levels=cell_levels)

plot_marker_expressions <- function(processed_list, gene_levels = NULL, cell_levels = NULL){

  levels2 <- cell_levels
  levels1 <- gene_levels
  if(is.null(levels1) & is.null(levels2)){
    #labels <- cell_labels
    #res <- as.data.frame(cbind(res, labels))
    res1 <- as.data.frame(melt(processed_list$percent))
    res2 <- as.data.frame(melt(processed_list$average))

    res <- res1
    res$value2 <- res2$value

    #res$labs = as.character(labels)
  }else{

    res1 <- as.data.frame(melt(processed_list$percent))
    res2 <- as.data.frame(melt(processed_list$average))

    res <- res1
    res$value2 <- res2$value

    res <- res %>% mutate(cellnames = factor(Var2, levels = levels2))
    res <- res %>% mutate(genenames = factor(Var1, levels = levels1))


  }
  res <- as.data.frame(res)
  colnames(res) <- c("Var1", "Var2", "percentage", "Mean", "cellnames", "genenames")

  spec_colors <- brewer.pal(n=5, name = "RdYlBu") #brewer.pal(n=5, name = "Spectral")

  Percentage <- (1 - res$percentage) * 50
  g <- ggplot(aes(genenames, cellnames, color = Mean), data = res) + geom_point(aes(size = Percentage)) +
    theme_aide() + scale_color_continuous(name = "Mean expression", low = spec_colors[3], high = spec_colors[1]) +
    theme(axis.text.x = element_text(size=15, angle=45, vjust = 0.5), axis.ticks.x = element_line(), legend.title = element_text())+
    labs(x = "Marker genes", y = "Discovered cell types")
  return(g)

}

#library(RColorBrewer)
# Rcolorbrewer custom set:
#aide_colors <- c( brewer.pal(8, "Dark2")[-5], brewer.pal(12, "Paired"))
#windowsFonts("helvetica"=windowsFont("Helvetica"))
#library(extrafont)
#font_import()
#loadfonts(device = "win")
#loadfonts(device = "pdf")
#windowsFonts(helvetica = "Helvetica")
theme_aide <- function(base_size=16, base_family="sans serif") {
  library(grid)
  library(ggthemes)
  (theme_foundation(base_size=base_size, base_family=base_family)
    + theme(plot.title = element_text(face = "bold",
                                      size = rel(1.2), hjust = 0.5),
            text = element_text(),
            panel.background = element_rect(colour = NA),
            plot.background = element_rect(colour = NA),
            panel.border = element_rect(colour = NA),
            axis.title = element_text(face = "bold",size = rel(1)),
            axis.title.y = element_text(angle=90,vjust =2),
            axis.title.x = element_text(vjust = -0.2),
            axis.text.y = element_text(size=15),#element_text(angle=75, vjust = 0.5),
            axis.text.x = element_blank(),
            axis.line = element_line(colour="black", size = 1),
            axis.ticks.x = element_blank(),
            axis.ticks.y = element_line(),
            panel.grid.major = element_blank(),#element_line(colour="#f0f0f0"),
            panel.grid.minor = element_blank(),
            legend.key = element_rect(colour = NA),
            legend.position = "right", #c(.9,.15), #"bottom", #(.9, .25)
            #legend.position = "bottom",
            legend.direction = "vertical",
            legend.key.size= unit(0.25, "cm"), # unit(0.2, "cm"),
            legend.margin = unit(0, "cm"),
            #legend.title = element_text(face="italic"),
            legend.title = element_blank(),
            plot.margin=unit(c(10,5,5,5),"mm"),
            strip.background = element_blank(),
            #strip.background=element_rect(colour="#f0f0f0",fill="#f0f0f0", size = 0.5),

            strip.text = element_text(face="bold", size = 12)
    ))

}
