#===========================安装包准备========================
require(devtools)
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
# 检查是否安装了 BiocManager，如果没有则进行安装
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

# 使用 BiocManager 安装所需的依赖包
BiocManager::install(c("BiocGenerics", "graph", "S4Vectors", "GenomicRanges", "IRanges"))
install.packages(c("gridExtra", "ggplot2", "foreach"))
# 安装 BiocManager
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

# 安装 Bioconductor 依赖包
BiocManager::install(c("BiocGenerics", "graph", "S4Vectors", "GenomicRanges", "IRanges"))

# 安装 CRAN 依赖包
install.packages(c("gridExtra", "ggplot2", "foreach"))

# 安装 remotes 包
if (!requireNamespace("remotes", quietly = TRUE))
  install.packages("remotes")

# 安装 ShatterSeek
remotes::install_github("parklab/ShatterSeek")

#=======================模拟数据==================================
set.seed(123)  # 固定随机种子
min_cluster_size <- 10        # 每个染色体碎裂簇的最小SV数量
max_chromosomes <- 10          # 最多模拟染色体碎裂的染色体数量
sv_types <- c("DEL", "DUP", "h2hINV", "t2tINV", "TRA")  # 允许的SV类型

# 染色体长度定义（全局变量）
# hg19（GRCh37）：这是人类基因组计划广泛使用的参考基因组版本之一。列表中的染色体长度与 hg19 的官方数据一致：
chr_lengths <- list(
  "1" = 249250621, "2" = 243199373, "3" = 198022430,
  "4" = 191154276, "5" = 180915260, "6" = 171115067,
  "7" = 159138663, "8" = 146364022, "9" = 141213431,
  "10" = 135534747, "11" = 135006516, "12" = 133851895,
  "13" = 115169878, "14" = 107349540, "15" = 102531392,
  "16" = 90354753, "17" = 81195210, "18" = 78077248,
  "19" = 59128983, "20" = 63025520, "21" = 48129895,
  "22" = 51304566, "X" = 155270560
)
# 生成 SV 数据（完全随机化）
generate_SV_data <- function(n_sv, chromothripsis_prob) {
  # 染色体列表（字符型，1-22, X）
  chromosomes <- as.character(c(1:22, "X"))
  
  # 随机选择发生染色体碎裂的染色体（最多选max_chromosomes个）
  chromoth_chrs <- sample(chromosomes, sample(1:max_chromosomes, 1))
  
  # 初始化数据框
  SV_data <- data.frame(
    chrom1 = character(n_sv),
    start1 = numeric(n_sv),
    end1 = numeric(n_sv),
    chrom2 = character(n_sv),
    start2 = numeric(n_sv),
    end2 = numeric(n_sv),
    sv_id = character(n_sv),
    pe_support = numeric(n_sv),
    strand1 = character(n_sv),
    strand2 = character(n_sv),
    svclass = character(n_sv),
    svmethod = character(n_sv),
    is_chromothripsis = logical(n_sv),  # 标记是否为染色体碎裂
    stringsAsFactors = FALSE
  )
  
  # 生成每个SV的特征
  for (i in 1:n_sv) {
    # ==== 染色体选择 ====
    # 根据概率决定是否为染色体碎裂事件
    is_chromoth <- runif(1) < chromothripsis_prob
    
    if (is_chromoth) {
      # 染色体碎裂事件：在指定染色体生成密集SV
      chrom_main <- sample(chromoth_chrs, 1)  # 随机选择一个染色体
      chr_pair <- rep(chrom_main, 2)          # 染色体内事件
      
      # 断点集中在50M-150M区域（模拟密集簇）
      cluster_center <- sample(50e6:150e6, 1)
      start1 <- cluster_center + sample(-1e5:1e5, 1)
      end1 <- start1 + sample(1:100, 1)
      start2 <- cluster_center + sample(-1e5:1e5, 1)
      end2 <- start2 + sample(1:100, 1)
      
      # SV类型严格平衡（DEL/DUP/h2hINV/t2tINV各25%）
      svclass <- sample(c("DEL", "DUP", "h2hINV", "t2tINV"), 1)
    } else {
      
      # 非染色体碎裂事件：随机染色体和断点
      chr_pair <- sample(chromosomes, 2, replace = TRUE)
      
      # 断点随机分布（允许跨染色体）
      start1 <- sample(1:(chr_lengths[[chr_pair[1]]] - 100), 1)
      end1 <- start1 + sample(1:100, 1)
      start2 <- sample(1:(chr_lengths[[chr_pair[2]]] - 100), 1)
      end2 <- start2 + sample(1:100, 1)
      
      
      # SV类型偏向DEL/DUP（70% DEL，30% DUP）
      svclass <- if (chr_pair[1] != chr_pair[2]) {
        "TRA"
      } else {
        sample(c("DEL", "DUP"), 1, prob = c(0.7, 0.3))
      }
    }
    
    # ==== 链方向和检测方法 ====
    strands <- switch(
      svclass,
      "DEL" = c("+", "-"),
      "DUP" = c("-", "+"),
      "h2hINV" = c("+", "+"),
      "t2tINV" = c("-", "-"),
      "TRA" = sample(c("+", "-"), 2, replace = TRUE)
    )
    
    svmethod <- switch(
      svclass,
      "DEL" = "SNOWMAN_BRASS_DELLY",
      "DUP" = "SNOWMAN_BRASS_dRANGER_DELLY",
      "h2hINV" = "SNOWMAN_dRANGER",
      "t2tINV" = "SNOWMAN_dRANGER",
      "TRA" = "SNOWMAN_DELLY"
    )
    
    # ==== 填充数据 ====
    SV_data[i, ] <- list(
      chrom1 = chr_pair[1],
      start1 = start1,
      end1 = end1,
      chrom2 = chr_pair[2],
      start2 = start2,
      end2 = end2,
      sv_id = paste0("SVMERGE", sample(1000:9999, 1)),
      pe_support = sample(10:150, 1),
      strand1 = strands[1],
      strand2 = strands[2],
      svclass = svclass,
      svmethod = svmethod,
      is_chromothripsis = is_chromoth  # 标记是否为染色体碎裂
    )
  }
  
  # 返回SV数据和涉及的染色体
  return(list(
    SV_data = SV_data,
    chromoth_chrs = chromoth_chrs  # 新增：返回染色体列表
  ))
}


# 生成 CNV 数据（需传递染色体列表和长度）
generate_CN_data <- function(chromoth_chrs, chr_lengths) {  # 新增chr_lengths参数
  # 随机为每个染色体生成振荡模式
  cnv_segments <- lapply(chromoth_chrs, function(chrom) {
    n_oscillations <- sample(5:15, 1)
    starts <- c(1, sort(sample(50e6:150e6, n_oscillations - 1)))
    ends <- c(starts[-1] - 1, chr_lengths[[chrom]])  # 使用实际染色体长度
    
    data.frame(
      chromosome = chrom,
      start = starts,
      end = ends,
      total_cn = rep(c(2, 1), length.out = n_oscillations)
    )
  })
  
  # 其他染色体保持正常
  normal_chrs <- setdiff(names(chr_lengths), chromoth_chrs)
  normal_segments <- lapply(normal_chrs, function(chrom) {
    data.frame(
      chromosome = chrom,
      start = 1,
      end = chr_lengths[[chrom]],  # 使用实际染色体长度
      total_cn = 2
    )
  })
  
  rbind(do.call(rbind, cnv_segments), do.call(rbind, normal_segments))
}

# 生成SV数据并获取涉及的染色体
sv_result <- generate_SV_data(n_sv = 5000, chromothripsis_prob = 0.4)
SV_DO17373 <- sv_result$SV_data
chromoth_chrs <- sv_result$chromoth_chrs  # 获取染色体列表

# 生成CNV数据（传递必要参数）
SCNA_DO17373 <- generate_CN_data(
  chromoth_chrs = chromoth_chrs,
  chr_lengths = chr_lengths  # 传递染色体长度
)

# 统计结果
chromoth_count <- sum(SV_DO17373$is_chromothripsis)
cat("染色体碎裂相关SV数量:", chromoth_count, "\n")
cat("涉及染色体:", chromoth_chrs, "\n")


# 保存数据（可选）
#write.csv(SV_DO17373, "simulated_SV_data.csv", row.names = FALSE)
#write.csv(SCNA_DO17373, "simulated_CNV_data.csv", row.names = FALSE)

#=======================ShatterSeek==============================
library(ShatterSeek)
library(gridExtra)
library(fs)  # 用于文件夹操作
SV_data <-SVs(chrom1=as.character(SV_DO17373$chrom1),
              pos1=as.numeric(SV_DO17373$start1),
              chrom2=as.character(SV_DO17373$chrom2),
              pos2=as.numeric(SV_DO17373$end2),
              SVtype=as.character(SV_DO17373$svclass),
              strand1=as.character(SV_DO17373$strand1),
              strand2=as.character(SV_DO17373$strand2))
CN_data <-CNVsegs(chrom=as.character(SCNA_DO17373$chromosome),
                  start=SCNA_DO17373$start,
                  end=SCNA_DO17373$end,
                  total_cn=SCNA_DO17373$total_cn)

start_time <-Sys.time()
chromothripsis <-shatterseek(SV.sample=SV_data, seg.sample=CN_data)
end_time <-Sys.time()
print(paste0("Runningtime(s):",round(end_time-start_time,digits=2)))
print(head(chromothripsis@chromSummary))
names(chromothripsis@detail)

plots_chr3 =plot_chromothripsis(ShatterSeek_output =chromothripsis,chr ="3")
plot_chr3 =arrangeGrob(plots_chr3[[1]],
                       plots_chr3[[2]],
                       plots_chr3[[3]],
                       plots_chr3[[4]],
                       nrow=4,ncol=1,heights=c(0.2,.4,.4,.4))
plots_chr2 =plot_chromothripsis(ShatterSeek_output =chromothripsis,chr ="2")
plot_chr2 =arrangeGrob(plots_chr2[[1]],
                       plots_chr2[[2]],
                       plots_chr2[[3]],
                       plots_chr2[[4]],
                       nrow=4,ncol=1,heights=c(0.2,.4,.4,.4))
plots_chr21=plot_chromothripsis(ShatterSeek_output =chromothripsis,chr ="21")
plot_chr21 =arrangeGrob(plots_chr21[[1]],
                        plots_chr21[[2]],
                        plots_chr21[[3]],
                        plots_chr21[[4]],
                        nrow=4,ncol=1,heights=c(0.2,.4,.4,.4))
plots_chrX =plot_chromothripsis(ShatterSeek_output =chromothripsis,chr ="X")
plot_chrX =arrangeGrob(plots_chrX[[1]],
                       plots_chrX[[2]],
                       plots_chrX[[3]],
                       plots_chrX[[4]],
                       nrow=4,ncol=1,heights=c(0.2,.4,.4,.4))
library(cowplot)
plot_grid(plot_chr3,plot_chr2)      



                                                            