#!/usr/bin/env Rscript
library(VariantAnnotation)
library(GenomicRanges)
library(dplyr)
library(stringr)
library(ShatterSeek)

# 1. get SV and CNV file path from command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript shatterSeek_Script.R <SV_file> <CNV_file>")
}
sv_file <- args[1]
cnv_file <- args[2]

# extract the sample name
sample_name <- strsplit(basename(sv_file), "\\.")[[1]][1]

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

# 2.1 define a function to preprocess the info from ALT column
parseAlt <- function(alt) {
  res <- list(chrom2 = NA, pos2 = NA, strand1 = NA, strand2 = NA)
  
  # derive strand2 value and content for chr2 and pos2
  if (str_detect(alt, "\\[")) {
    res$strand2 <- "-"
    content <- str_match(alt, ".*\\[([^\\[\\]]+)\\[.*")[,2]
  } else if (str_detect(alt, "\\]")) {
    res$strand2 <- "+"
    content <- str_match(alt, ".*\\]([^\\[\\]]+)\\].*")[,2]
  } else {
    content <- NA
  }
  
  # derive chr2 and pos2 from content
  if (!is.na(content)) {
    parts <- str_split(content, ":", simplify = TRUE)
    if(ncol(parts) == 2) {
      res$chrom2 <- parts[1]
      res$pos2 <- as.numeric(parts[2])
    }
  }
  
  # get the strand1 value
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

# 2.2 remove the "chr" part to meet the requirement of ShatterSeek
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
  # remove the duplicated entries
  mutate(
    keep = ifelse(num_chrom1 == num_chrom2, pos1 < pos2, num_chrom1 < num_chrom2)
  ) %>%
  filter(keep) %>%
  select(-keep, -num_chrom1, -num_chrom2)

# 2.3 add SVtype column
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

# 4.2 main function
# chromothripsis <- shatterseek(SV.sample = SV_data, seg.sample = CN_data, genome = "hg38")
invisible(capture.output(chromothripsis <- shatterseek(SV.sample = SV_data, seg.sample = CN_data, genome = "hg38")))

# 4.3 set the cut-offs
# create a column with the intrachromosomal SVs count for the cluster
results <- chromothripsis@chromSummary
results$number_intrachromosomal_SVs <- results %>% 
  dplyr::select(number_DEL, number_DUP, number_h2hINV, number_t2tINV) %>% 
  apply(1, sum)

# the cutoffs are recommended values from ShatterSeek
# the code below provided by ahwanpandey

# 1st High Confidence filter
filt1 = results$number_intrachromosomal_SVs >= 6
filt2 = results$max_number_oscillating_CN_segments_2_states >= 7
filt3 = results$pval_fragment_joins >= 0.05
filt4 = (results$chr_breakpoint_enrichment <= 0.05) | (results$pval_exp_chr <= 0.05)
HC1 = (filt1) & (filt2) & (filt3) & (filt4)
HC1[is.na(HC1)] <- FALSE
results$HC1 <- HC1

# 2nd High Confidence filter
filt1 = results$number_intrachromosomal_SVs >= 3
filt2 = results$number_TRA >= 4
filt3 = results$max_number_oscillating_CN_segments_2_states >= 7
filt4 = results$pval_fragment_joins >= 0.05
HC2 = (filt1) & (filt2) & (filt3) & (filt4)
HC2[is.na(HC2)] <- FALSE
results$HC2 <- HC2

# 3rd High Confidence filter
filt1 = results$clusterSize_including_TRA >= 40
filt2 = results$pval_fragment_joins >= 0.05
HC3gte40 = (filt1) & (filt2)
HC3gte40[is.na(HC3gte40)] <- FALSE
results$HC3gte40 <- HC3gte40

# Low Confidence filter
filt1 = results$number_intrachromosomal_SVs >= 6
filt2 = (results$max_number_oscillating_CN_segments_2_states >= 4) & (results$max_number_oscillating_CN_segments_2_states <= 6)
filt3 = results$pval_fragment_joins >= 0.05
filt4 = (results$chr_breakpoint_enrichment <= 0.05) | (results$pval_exp_chr <= 0.05)
LC1 = (filt1) & (filt2) & (filt3) & (filt4)
LC1[is.na(LC1)] <- FALSE
results$LC1 <- LC1

# results$HCany if TRUE means a Chromothripsis event has occurred
results$HCany <- results$HC1 | results$HC2 | results$HC3gte40

num_chrom_HCany <- sum(results$HCany, na.rm = TRUE)

# ----------------------------
# 5. Final Output
# ----------------------------
cat(sample_name, num_chrom_HCany, sep = "\t")
