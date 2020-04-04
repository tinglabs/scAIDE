#scAIDE:
#library(parallel)

find_markers <- function(input_data_matrix, cluster_labels, identity=1, threads = 8){
  groups = as.character(cluster_labels)
  identity = as.character(identity)
  groups[which(groups==identity)] <- "a"
  groups[which(groups!="a")] <- "b"
  print(table(groups))
  cl <- makeCluster(threads)

  res1 <- parLapply(cl, 1:nrow(input_data_matrix), fun=function(x){
    wilcox.test(input_data_matrix[x, ]~factor(groups), correct = T)$p.value})
  stopCluster(cl)
  closeAllConnections()

  tt <- Sys.time()
  avg.a = apply(input_data_matrix[,groups=="a"], 1, mean)
  avg.na = apply(input_data_matrix[,groups!="a"], 1, mean)

  log2fc = avg.a / avg.na
  print(length(log2fc))
  te <- Sys.time()
  print(paste("AVG Time: ", te-tt))
  return(list(res1, log2fc))
}



#' Calculate the wilcox rank sum test and log fold change values
#'
#' This function calculates the wilcox rank sum test and log fold change ratio for all genes in each cluster.
#'@param input_data_matrix A pre-processed gene expression matrix (ie normalization and log). Matrix input is a genes (row) by cells (col) gene expression matrix with all genes.
#'@param labels_vector A numeric vector containing the cell cluster assignment of each cell. If minimum value is 0, it will automatically add 1 to the vector.
#'@param threads The number of threads used for parallel computing. Defaults to 8
#'@export
#'@keywords store_markers
#'@section Biological Analysis:
#'@examples
#'gene_expression_data <- read.csv("single_cell_dataset.csv", header = T, row.names = 1) # make sure that genes are in rows
#'cluster_vector <- read.table("results_from_rpkmeans.txt")$V1
#'eval_gene_markers <- store_markers(gene_expression_data, cluster_vector, threads = 8)

store_markers <- function(input_data_matrix, labels_vector, threads = 8){
  data = input_data_matrix
  store_lists <- list()
  #labels <- read.csv("/data/ken/1m_analytics_scripts/sample_5percent_labels.csv", header = T)
  if(min(labels_vector)==0){
    labels_vec <- labels_vector + 1
  }else{
    labels_vec <- labels
  }
  s1_list <- sort(unique(labels[,1]))+1
  #a <- Sys.time()
  for(i in c(1:length(unique(s1_list)))){
    print(i)
    store_lists[[s1_list[i]]] <- find_markers(data, labels_vec, identity = s1_list[i], threads = 8)
    #store_lists[[i]] <- fm
    #if(i==3){
    #  print(store_lists[[2]])
    #}
  }
  return(store_lists)

}
#' Find the expression markers specific to your cluster assignments
#'
#' This function calculates the wilcox rank sum test and log fold change ratio for all genes in each cluster.
#'@param whole_list  This is the result returned from the function 'store_markers', containing the wilcox test and log fold change values.
#'@param gene_names A character vector containing the gene names in the same order as your input gene expression matrix. e.g. c("SOX2", "ALDOC")
#'@param wilcox_threshold This is the p-value cut-off for the wilcox test, it is set to 0.001 by default. Recommend a smaller value for a stricter detection
#'@param logfc_threshold This is the log fold change threshold, which is 1.5 by default. This can be increased for stricter restriction on sample sizes.
#'@export
#'@keywords curate_markers
#'@section Biological Analysis:
#'@examples
#'gene_expression_data <- read.csv("single_cell_dataset.csv", header = T, row.names = 1) # make sure that genes are in rows
#'cluster_vector <- read.table("results_from_rpkmeans.txt")$V1
#'eval_gene_markers <- store_markers(gene_expression_data, cluster_vector, threads = 8)
#'gene_names = rownames(gene_expression_data)
#'cluster_markers_list <- curate_markers(eval_gene_markers, gene_names, wilcox_threshold=0.001, logfc_threshold=1.5)


curate_markers <- function(whole_list, gene_names, wilcox_threshold = 0.001, logfc_threshold = 1.5){
  store_genes <- list()
  n_clusters <- length(whole_list)
  non_na <- which(!is.na(gene_names))
  for(i in c(1:n_clusters)){
    c1 <- intersect(which(as.numeric(whole_list[[i]][[1]]) <= wilcox_threshold), which(as.numeric(whole_list[[i]][[2]]) > logfc_threshold))

    c_markers <- gene_names[intersect(c1, non_na)]
    print(length(c_markers))
    if(length(c_markers)==0){
      subset1 <- which(as.numeric(whole_list[[i]][[2]]) > logfc_threshold)

      sorted_wilcox <- order(as.numeric(whole_list[[i]][[1]]), decreasing = F)

      len_pos <- c(1:length(whole_list[[i]][[1]]))
      len_sorted <- len_pos[sorted_wilcox]

      selected_pos <- intersect(len_sorted, subset1)[1:100]
      print(as.numeric(whole_list[[i]][[1]])[selected_pos])

      c_markers <- gene_names[selected_pos]
    }
    store_genes[[i]] <- c_markers
  }
  return(store_genes)

}

###########
#' Utility function for calculating the overlaps of specific markers vs each cluster
#'
#' This is the utility function behind calculate_celltype_prob, which implements the Jaccard, accuracy and F1 to calculate the value of overlaps between marker genes and cluster specific genes.
#'@param gene_name List of gene names that we want to query in the clusters.
#'@param f_list A list object that contains the marker genes from each specific cluster.
#'@param type This determines the type of overlap to calculate. Defaults to the Jaccard index, "jacc". Accuracy ("ac") and F1 ("f1") are also available. We recommend using jaccard or accuracy in applications.
#'@export
#'@keywords find_specific_marker
#'@section Biological Analysis:

find_specific_marker <- function(gene_name, f_list, type = "jacc"){
  store_present <- c()
  # type: f1 / ac
  for(i in c(1:length(f_list))){

    g1 <- match(gene_name, tolower(f_list[[i]]))
    tp = length(which(!is.na(g1)))
    fp = length(f_list[[i]]) - tp
    fn = length(gene_name) - tp
    pr = tp / (tp+fp)
    re = tp / (tp+fn)
    if(pr==0 & re==0){
      f1 = 0
    }else{
      f1 = 2*(pr*re / (pr+re))
    }
    acc_sc <- length(which(!is.na(g1)))/length(gene_name)

    jacc <- length(which(!is.na(g1))) / (length(gene_name) + length(f_list[[i]]))
    #print(g1)
    #print(paste(i, ": ", length(which(!is.na(g1)))/length(gene_name)))
    #print(paste(i, ": ", f1))
    #if(sum(g1) > 0){
    #if(!is.null(g1)){
    #  store_present <- c(store_present, i)
    #}
    if(type == "ac"){
      store_present <- c(store_present, acc_sc)

    }else if(type=="f1"){
      store_present <- c(store_present, f1)
    }else{
      store_present <- c(store_present, jacc)
    }
  }
  return(store_present)
}

#' Calculate the cell type probablity assignment according to a markers database
#'
#' This function calculates probability of each cell type in the desired database, according to the number of overlapping genes. We have included the "Panglao" database as of date 7th Feb 2020 in our package.
#'@param clt_marker_list This is the result returned from the function 'curate_markers', a list object containing marker genes for each specific cluster.
#'@param marker_database A list object that contains the marker genes from each specific cell type. A pre-processed variable is stored as 'panglao_marker_list', which contains markers for neural and immune cell types.
#'@param type This determines the type of overlap to calculate. Defaults to the Jaccard index, "jacc". Accuracy ("ac") and F1 ("f1") are also available. We recommend using jaccard or accuracy in applications.
#'@export
#'@keywords calculate_celltype_prob
#'@section Biological Analysis:
#'@examples
#'gene_expression_data <- read.csv("single_cell_dataset.csv", header = T, row.names = 1) # make sure that genes are in rows
#'cluster_vector <- read.table("results_from_rpkmeans.txt")$V1
#'eval_gene_markers <- store_markers(gene_expression_data, cluster_vector, threads = 8)
#'gene_names = rownames(gene_expression_data)
#'cluster_markers_list <- curate_markers(eval_gene_markers, gene_names, wilcox_threshold=0.001, logfc_threshold=1.5)
#'
#'celltype_prob <- calculate_celltype_prob(cluster_markers_list, panglao_marker_list, type = "jacc")
#'celltype_id <- rowMaxs(celltype_prob)

calculate_celltype_prob <- function(clt_marker_list, marker_database_list, type="jacc"){
  marker_database_list <- marker_databse_list
  no_clts <- length(clt_marker_list)
  no_celltypes <- length(marker_databse_list)
  result_matrix <- matrix(0, nrow = no_clts, ncol = no_celltypes)
  #pval_matrix <- matrix(0, nrow = no_clts, ncol = no_celltypes)

  for(i in c(1:no_celltypes)){
    tmp_markers <- tolower(marker_databse_list[[i]])
    tmp_results <- find_specific_marker(tmp_markers, clt_marker_list, type)
    result_matrix[,i] <- tmp_results

    #len_N <- total_background_genes
    #len_A <- length(marker_databse_list[[i]])
    #len_B <- length(clt_marker_list)
  }
  return(result_matrix)
}

#require(gmp)

enrich_pvalue <- function(N, A, B, k)
{
  m <- A + k
  n <- B + k
  i <- k:min(m,n)

  as.numeric( sum(chooseZ(m,i)*chooseZ(N-m,n-i))/chooseZ(N,n) )
}


#' Calculate the enrichment probablity of each cell type based on a hypergeometric distribution.
#'
#' This function calculates enrichment probability for each cell type vs each cluster. In principle, it determines the statistical validity of the cell type assignment, based on the number of overlapping genes.
#'@param clt_marker_list This is the result returned from the function 'curate_markers', a list object containing marker genes for each specific cluster.
#'@param marker_database A list object that contains the marker genes from each specific cell type. A pre-processed variable is stored as 'panglao_marker_list', which contains markers for neural and immune cell types.
#'@param total_background_genes This is the number of genes in your dataset
#'@param type This determines the type of overlap to calculate. Defaults to the Jaccard index, "jacc". Accuracy ("ac") and F1 ("f1") are also available. We recommend using jaccard or accuracy in applications.
#'@export
#'@keywords calculate_enrichment_prob
#'@section Biological Analysis:
#'@examples
#'gene_expression_data <- read.csv("single_cell_dataset.csv", header = T, row.names = 1) # make sure that genes are in rows
#'cluster_vector <- read.table("results_from_rpkmeans.txt")$V1
#'eval_gene_markers <- store_markers(gene_expression_data, cluster_vector, threads = 8)
#'gene_names = rownames(gene_expression_data)
#'cluster_markers_list <- curate_markers(eval_gene_markers, gene_names, wilcox_threshold=0.001, logfc_threshold=1.5)
#'
#'n_genes <- nrow(gene_expression_data)
#'enrichment_prob <- calculate_enrichment_prob(cluster_markers_list, panglao_marker_list, n_genes, type = "jacc")


calculate_enrichment_prob <- function(clt_marker_list, marker_databse_list, total_background_genes = 10000, type="jacc"){

  no_clts <- length(clt_marker_list)
  no_celltypes <- length(marker_databse_list)
  #result_matrix <- matrix(0, nrow = no_clts, ncol = no_celltypes)
  pval_matrix <- matrix(0, nrow = no_clts, ncol = no_celltypes)

  for(i in c(1:no_celltypes)){
    tmp_markers <- tolower(marker_databse_list[[i]])
    #tmp_results <- find_specific_marker(tmp_markers, clt_marker_list, type)
    #result_matrix[,i] <- tmp_results
    len_N <- total_background_genes
    len_A <- length(marker_databse_list[[i]])
    for(j in c(1:no_clts)){
      len_B <- length(clt_marker_list[[j]])

      len_k <- length(which(!is.na(match(tmp_markers, tolower(clt_marker_list[[j]])))))
      pval_matrix[j, i] <- enrich_pvalue(len_N, len_A, len_B, len_k)
    }

  }
  return(pval_matrix)
}


#################################################################:
# Process panglao markers database;

process_specific_type <- function(cell_type_name, db_list){
  get_position <- which(tolower(db_list$cell.type)==cell_type_name)
  print(get_position)
  if(length(get_position)==0){
    print("cannot find cell type: ")
    print(cell_type_name)
    #break
  }else{
    subset_type <- db_list[get_position, ]
    print(dim(subset_type))
    return(subset_type)
  }

}

#' Pre-processes the Panglao database to curate markers for desired cell types.
#'
#' This function processes the panglao database to the format of lists, where each list is a cell type containing marker genes.
#'@param cell_type_name_list This is the corresponding cell type names in the database.
#'@param database The database table which is loaded.
#'@export
#'@keywords calculate_enrichment_prob
#'@section Biological Analysis:
#'@examples
#'panglao_table <-read.csv("panglao_database.csv", header = T, sep = "\t")
#'type_names <- c("astrocytes", "oligodendrocytes")
#'preprocessed_pl <- process_panglao_db(type_names, panglao_table)
#'
#'# Preloaded Panglao database objects:
#'# panglao_db: the full panglao marker list as of date 7th Feb 2020
#'# panglao_marker_list: the pre-processed list for all neural and immune cell types.

process_panglao_db <- function(cell_type_name_list, database){
  store_markers <- c()
  for(i in c(1:length(cell_type_name_list))){
    store_markers <- rbind(store_markers, process_specific_type(cell_type_name_list[i], database))
  }

  marker_list <- list()
  for(i in c(1:length(cell_type_name_list))){
    selected_genes <- store_markers$official.gene.symbol[which(tolower(store_markers$cell.type)==filter_cell_names[i])]
    marker_list[[cell_type_name_list[i]]] <- as.character(selected_genes)
  }
  return(marker_list)
}

