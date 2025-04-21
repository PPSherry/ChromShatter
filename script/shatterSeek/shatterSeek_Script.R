#!/usr/bin/env Rscript
library(VariantAnnotation)
library(GenomicRanges)
library(dplyr)
library(stringr)
library(ShatterSeek)
library(ggplot2)

# import the plot_SV.R file
source("/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/script/shatterSeek/plot_SV.R")

# 1. Get command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("Usage: Rscript shatterSeek_Script.R <SV_file> <CNV_file> <case_id> <results_dir> <graph_dir>")
}
sv_file <- args[1]
cnv_file <- args[2]
case_id <- args[3]
results_dir <- args[4]
graph_dir <- args[5]

# ----------------------------
# 2. SV preprocessing
# ----------------------------
vcf <- readVcf(sv_file, "hg38")
gr <- rowRanges(vcf)
chrom1 <- as.character(seqnames(gr))
pos1 <- start(gr)
alts <- as.character(unlist(alt(vcf)))

sv_df <- data.frame(
  chrom1 = chrom1,
  pos1 = pos1,
  ALT = alts,
  stringsAsFactors = FALSE
)

# 2.1 Define a function to preprocess the info from ALT column
parseAlt <- function(alt) {
  res <- list(chrom2 = NA, pos2 = NA, strand1 = NA, strand2 = NA)
  
  # Derive strand2 value and content for chr2 and pos2
  if (str_detect(alt, "\\[")) {
    res$strand2 <- "-"
    content <- str_match(alt, ".*\\[([^\\[\\]]+)\\[.*")[,2]
  } else if (str_detect(alt, "\\]")) {
    res$strand2 <- "+"
    content <- str_match(alt, ".*\\]([^\\[\\]]+)\\].*")[,2]
  } else {
    content <- NA
  }
  
  # Derive chr2 and pos2 from content
  if (!is.na(content)) {
    parts <- str_split(content, ":", simplify = TRUE)
    if(ncol(parts) == 2) {
      res$chrom2 <- parts[1]
      res$pos2 <- as.numeric(parts[2])
    }
  }
  
  # Get the strand1 value
  if (str_detect(alt, "^[ACGTNacgtn]+")) {
    res$strand1 <- "+"
  } else if (str_detect(alt, "[ACGTNacgtn]+$")) {
    res$strand1 <- "-"
  }
  
  return(res)
}

sv_df <- sv_df %>% 
  rowwise() %>%
  mutate(parsed = list(parseAlt(ALT))) %>%
  mutate(
    chrom2 = parsed$chrom2,
    pos2 = parsed$pos2,
    strand1 = parsed$strand1,
    strand2 = parsed$strand2
  ) %>%
  ungroup() %>%
  select(-parsed)

# 2.2 Remove the "chr" part to meet the requirement of ShatterSeek
sv_df <- sv_df %>%
  mutate(
    chrom1 = str_replace(chrom1, "^chr", ""),
    chrom2 = str_replace(chrom2, "^chr", "")
  ) %>%
  mutate(
    num_chrom1 = case_when(
      chrom1 == "X" ~ 23,
      TRUE ~ as.numeric(chrom1)
    ),
    num_chrom2 = case_when(
      chrom2 == "X" ~ 23,
      TRUE ~ as.numeric(chrom2)
    )
  ) %>%
  # Remove the duplicated entries
  mutate(
    keep = ifelse(num_chrom1 == num_chrom2, pos1 < pos2, num_chrom1 < num_chrom2)
  ) %>%
  filter(keep) %>%
  select(-keep, -num_chrom1, -num_chrom2)

# 2.3 Add SVtype column
sv_shatter <- sv_df %>%
  mutate(SVtype = case_when(
    chrom1 != chrom2 ~ "TRA",
    chrom1 == chrom2 & strand1 == "+" & strand2 == "-" ~ "DEL",
    chrom1 == chrom2 & strand1 == "-" & strand2 == "+" ~ "DUP",
    chrom1 == chrom2 & strand1 == "+" & strand2 == "+" ~ "h2hINV",
    chrom1 == chrom2 & strand1 == "-" & strand2 == "-" ~ "t2tINV",
    TRUE ~ "UNK"
  ))

# ----------------------------
# 3. CNV preprocessing
# ----------------------------
cnv <- read.delim(cnv_file, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
cnv_shatter <- cnv %>%
  mutate(
    chrom = sub("^chr", "", Chromosome),
    start = Start,
    end = End,
    total_cn = Copy_Number
  ) %>%
  select(chrom, start, end, total_cn)

# ----------------------------
# 4. Run shatterSeek
# ----------------------------
# 4.1 Input SV & CNV info to ShatterSeek
SV_data <- SVs(chrom1 = as.character(sv_shatter$chrom1), 
               pos1 = as.numeric(sv_shatter$pos1),
               chrom2 = as.character(sv_shatter$chrom2), 
               pos2 = as.numeric(sv_shatter$pos2), 
               SVtype = as.character(sv_shatter$SVtype), 
               strand1 = as.character(sv_shatter$strand1), 
               strand2 = as.character(sv_shatter$strand2))

CN_data <- CNVsegs(chrom = as.character(cnv_shatter$chrom), 
                   start = cnv_shatter$start,
                   end = cnv_shatter$end,
                   total_cn = cnv_shatter$total_cn)

# 4.2 Main function
invisible(capture.output(chromothripsis <- shatterseek(SV.sample = SV_data, seg.sample = CN_data, genome = "hg38")))

# 4.3 Set the cut-offs
chromSummary <- chromothripsis@chromSummary
chromSummary$original_order <- 1:nrow(chromSummary)
results <- merge(chromSummary, 
                 chromothripsis@detail$maxClusterSize, 
                 by = "chrom")
results <- results[order(results$original_order), ]
results$original_order <- NULL

# The cutoffs are recommended values from ShatterSeek

# 1st High Confidence filter
intra_chr_num_6 = results$clusterSize >= 6
max_num_cn2_num_7 = results$max_number_oscillating_CN_segments_2_states >= 7
equal_distribution_SVtype = results$pval_fragment_joins >= 0.05
breakpoint_enrich = (results$chr_breakpoint_enrichment <= 0.05) | (results$pval_exp_chr <= 0.05)

HC_standard = rep("PASS", nrow(results))
for (i in 1:nrow(results)) {
  failed_criteria = c()
  if (is.na(intra_chr_num_6[i]) || !intra_chr_num_6[i]) failed_criteria = c(failed_criteria, "intra_chr_num_6")
  if (is.na(max_num_cn2_num_7[i]) || !max_num_cn2_num_7[i]) failed_criteria = c(failed_criteria, "max_num_cn2_num_7")
  if (is.na(equal_distribution_SVtype[i]) || !equal_distribution_SVtype[i]) failed_criteria = c(failed_criteria, "equal_distribution_SVtype")
  if (is.na(breakpoint_enrich[i]) || !breakpoint_enrich[i]) failed_criteria = c(failed_criteria, "breakpoint_enrich")
  if (length(failed_criteria) > 0) HC_standard[i] = paste(failed_criteria, collapse=",")
}
results$HC_standard <- HC_standard

# 2nd High Confidence filter
intra_inter_chr_num_7 = (results$clusterSize >= 3) & (results$number_TRA >= 4)
max_num_cn2_num_7 = results$max_number_oscillating_CN_segments_2_states >= 7
equal_distribution_SVtype = results$pval_fragment_joins >= 0.05

HC_supplement1 = rep("PASS", nrow(results))
for (i in 1:nrow(results)) {
  failed_criteria = c()
  if (is.na(intra_inter_chr_num_7[i]) || !intra_inter_chr_num_7[i]) failed_criteria = c(failed_criteria, "intra_inter_chr_num_7")
  if (is.na(max_num_cn2_num_7[i]) || !max_num_cn2_num_7[i]) failed_criteria = c(failed_criteria, "max_num_cn2_num_7")
  if (is.na(equal_distribution_SVtype[i]) || !equal_distribution_SVtype[i]) failed_criteria = c(failed_criteria, "equal_distribution_SVtype")
  if (length(failed_criteria) > 0) HC_supplement1[i] = paste(failed_criteria, collapse=",")
}
results$HC_supplement1 <- HC_supplement1

# 3rd High Confidence filter
cluster_size_gte40 = results$clusterSize_including_TRA >= 40
equal_distribution_SVtype = results$pval_fragment_joins >= 0.05

HC_supplement2 = rep("PASS", nrow(results))
for (i in 1:nrow(results)) {
  failed_criteria = c()
  if (is.na(cluster_size_gte40[i]) || !cluster_size_gte40[i]) failed_criteria = c(failed_criteria, "cluster_size_gte40")
  if (is.na(equal_distribution_SVtype[i]) || !equal_distribution_SVtype[i]) failed_criteria = c(failed_criteria, "equal_distribution_SVtype")
  if (length(failed_criteria) > 0) HC_supplement2[i] = paste(failed_criteria, collapse=",")
}
results$HC_supplement2 <- HC_supplement2

# Low Confidence filter
intra_chr_num_6 = results$clusterSize >= 6
max_num_cn2_num_4to6 = (results$max_number_oscillating_CN_segments_2_states >= 4) & (results$max_number_oscillating_CN_segments_2_states <= 6)
equal_distribution_SVtype = results$pval_fragment_joins >= 0.05
breakpoint_enrich = (results$chr_breakpoint_enrichment <= 0.05) | (results$pval_exp_chr <= 0.05)

LC = rep("PASS", nrow(results))
for (i in 1:nrow(results)) {
  failed_criteria = c()
  if (is.na(intra_chr_num_6[i]) || !intra_chr_num_6[i]) failed_criteria = c(failed_criteria, "intra_chr_num_6")
  if (is.na(max_num_cn2_num_4to6[i]) || !max_num_cn2_num_4to6[i]) failed_criteria = c(failed_criteria, "max_num_cn2_num_4to6")
  if (is.na(equal_distribution_SVtype[i]) || !equal_distribution_SVtype[i]) failed_criteria = c(failed_criteria, "equal_distribution_SVtype")
  if (is.na(breakpoint_enrich[i]) || !breakpoint_enrich[i]) failed_criteria = c(failed_criteria, "breakpoint_enrich")
  if (length(failed_criteria) > 0) LC[i] = paste(failed_criteria, collapse=",")
}
results$LC <- LC

# Mark the chromothripsis status
results$chromothripsis_status <-
  ifelse(
    (results$HC_standard  == "PASS") |
    (results$HC_supplement1 == "PASS") |
    (results$HC_supplement2 == "PASS"),
    "High Confidence",
    ifelse(results$LC == "PASS", "Low Confidence", "Not Significant")
  )

# ----------------------------
# 5. Generate plots and prepare final output
# ----------------------------
# Add plot_path column to results
results$plot_path <- NA

# Create a data frame to store the final results
final_output <- data.frame()

# For each chromosome with clusterSize > 0, generate a plot
for (i in 1:nrow(results)) {
  chr <- results$chrom[i]
  if (!is.na(results$clusterSize[i]) && results$clusterSize[i] > 0) {
    # Set plot file path
    plot_file <- file.path(graph_dir, paste0(case_id, "_chr", chr, "_SV_plot.png"))
    
    # Generate the plot
    tryCatch({
      sv_plot <- plot_sv_arcs(
        ShatterSeek_output = chromothripsis,
        chr = chr,
        save_plot = TRUE,
        output_file = plot_file,
        width = 1050,
        height = 320,
        res = 300
      )
      # Update plot path in results
      results$plot_path[i] <- paste0(case_id, "_chr", chr, "_SV_plot.png")
    }, error = function(e) {
      cat("Error generating plot for chromosome", chr, ":", e$message, "\n")
    })
  }
}

# Combine all rows into the final output
final_output <- rbind(final_output, results)

# Clean up any newlines in the data (especially in inter_other_chroms_coords_all column)
for (col in colnames(final_output)) {
  if (is.character(final_output[[col]])) {
    final_output[[col]] <- gsub("\n", ";", final_output[[col]])
  }
}

# Write results to a file
output_file <- file.path(results_dir, paste0(case_id, "_shatterSeek_results.tsv"))
write.table(final_output, output_file, sep = "\t", row.names = FALSE, quote = TRUE)

