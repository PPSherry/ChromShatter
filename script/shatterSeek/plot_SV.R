#' Plot structural variation arcs
#' 
#' @param ShatterSeek_output the output of the function shatterseek
#' @param chr chromosome for which the plot will be generated
#' @param DEL_color colour to show the deletion-like SVs
#' @param DUP_color colour to show the duplication-like SVs
#' @param t2tINV_color colour to show the t2tINV SVs
#' @param h2hINV_color colour to show the h2hINV SVs
#' @param arc_size size of the arcs representing intrachromosomal SVs
#' @param save_plot logical, whether to save the plot
#' @param output_file path to save the plot
#' @param width width of the output plot
#' @param height height of the output plot
#' @param res resolution of the output plot
#' @return a ggplot object showing the SV arcs
#' 
#' @export
plot_sv_arcs <- function(ShatterSeek_output, chr,
                        DEL_color='darkorange1', DUP_color='blue1',
                        t2tINV_color="forestgreen", h2hINV_color="black",
                        arc_size=0.2,
                        save_plot=FALSE,
                        output_file="SV_plot.png",
                        width=1050, height=320, res=300) {
  
  # 通用ggplot2主题设置
  common_ggplot2 <- theme_bw() + theme(axis.text.x=element_text(size=7,angle=0),
                                      axis.text.y=element_text(size=7),
                                      axis.title.y=element_text(size=7),
                                      axis.title.x=element_blank(),
                                      legend.position="none",
                                      legend.text = element_text(size=7),
                                      legend.key = element_blank(),
                                      plot.margin=unit(c(0.1,0.1,0,0.1),"cm"),
                                      plot.title=element_blank(),
                                      panel.grid.major = element_blank(),
                                      panel.grid.minor = element_blank(), 
                                      legend.title=element_blank(),
                                      plot.background = element_blank(),
                                      axis.line.x = element_line(linewidth = 0.5, linetype = "solid", colour = "black"),
                                      axis.line.y = element_line(linewidth = 0.5, linetype = "solid", colour = "black"))  
  
  # 处理染色体名称
  cand = gsub("chr","",chr)
  chr = paste("chr",cand,sep="")
  
  # 获取染色体上的SV
  SVsnow <- ShatterSeek_output@detail$SV
  SVsnow <- unique(SVsnow[SVsnow$chrom1 == cand, ])
  
  # 设置绘图参数
  y1=4; y2=12
  df = SVsnow
  df$y1 = rep(y1,nrow(df))
  df$y2 = rep(y2,nrow(df))
  
  # 确定坐标范围和曲率
  if (nrow(df)!=0){
    min_coord = min(df$pos1)
    max_coord = max(df$pos2)
    df$diff = abs(df$pos1 - df$pos2)
    df$curv = 1 -(df$diff / max(df$diff))
    max_diff = max(df$diff)
    df$curv[(df$diff / max_diff) > 0.2] <- .15
    df$curv[(df$diff / max_diff) > 0.8] <- .08
    df$curv[(df$diff / max_diff) < 0.2] <- 1
  } else {
    min_coord = 0
    max_coord = 10
  }
  
  # 初始化SV图
  d = data.frame(x=c(min_coord),y=c(1),leg=c("DEL","DUP","t2tINV","h2hINV"))
  
  SV_plot = ggplot(d,aes(x=x,y=y)) +
    geom_point(colour="white") + ylim(0,y2+5) + common_ggplot2 +
    geom_line(data=rbind(d,d),aes(x=x,y=y,colour=leg)) 
  
  # 处理染色体间SV
  inter <- ShatterSeek_output@detail$SVinter   
  inter <- inter[which(inter$chrom1 == cand | inter$chrom2 == cand), ] 
  
  if (nrow(inter)>0){
    inter$SVtype = factor(inter$SVtype,levels=c("DEL","DUP","h2hINV","t2tINV"))
    inter$y = rep(0,nrow(inter))
    inter$y[which(inter$SVtype %in% c("DUP","DEL"))] = 4
    inter$y[!(inter$SVtype %in% c("DUP","DEL"))] = 12
    inter$type_SV = rep("",nrow(inter))
    inter$type_SV[which(inter$strand1 == "-" & inter$strand2 == "-")] = "t2tINV"
    inter$type_SV[which(inter$strand1 == "-" & inter$strand2 == "+")] = "DUP"
    inter$type_SV[which(inter$strand1 == "+" & inter$strand2 == "-")] = "DEL"
    inter$type_SV[which(inter$strand1 == "+" & inter$strand2 == "+")] = "h2hINV"
    inter$SVtype = inter$type_SV; inter$type_SV=NULL
    inter$colour = rep("",nrow(inter))
    inter$colour[which(inter$SVtype == "DUP")] = DUP_color
    inter$colour[which(inter$SVtype == "DEL")] = DEL_color
    inter$colour[which(inter$SVtype == "h2hINV")] = h2hINV_color
    inter$colour[which(inter$SVtype == "t2tINV")] = t2tINV_color
    
    inter = data.frame(chr = c(inter$chrom1,inter$chrom2), pos = c(inter$pos1,inter$pos2), y=c(inter$y,inter$y), SVtype=c(inter$SVtype,inter$SVtype))
    inter = inter[inter$chr == cand, c(2:4)]
    SV_plot = SV_plot + geom_point(data=inter,size=1,alpha=1,aes(x=pos,y=as.numeric(y),colour=SVtype))
  }
  
  # 添加水平参考线
  SV_plot = SV_plot + geom_hline(yintercept=y1,linewidth=0.5) + geom_hline(yintercept=y2,linewidth=0.5) 
  
  # 处理大数据集
  if(nrow(df)>300){options(expressions= 100000)}
  
  # 创建索引用于图例
  idx = c()
  
  # 绘制DUP类型SV
  now = df[df$SVtype == "DUP",]
  if (nrow(now) > 0){
    for (i in 1:nrow(now)){
      SV_plot = SV_plot + geom_curve(linewidth=arc_size,data = now[i,], 
                                    aes(x = pos1, y = y1, xend = pos2, yend = y1), 
                                    curvature = now$curv[i],colour=DUP_color,ncp=8)
    }
    idx= c(idx,1)
  }
  SV_plot = SV_plot + geom_point(data=now,size=.5,aes(x=pos1,y=y1)) + 
    geom_point(data=now,size=.5,aes(x=pos2,y=y1))
  
  # 绘制DEL类型SV
  now = df[df$SVtype == "DEL",]
  if (nrow(now) > 0){
    for (i in 1:nrow(now)){
      SV_plot = SV_plot + geom_curve(linewidth=arc_size,data = now[i,], 
                                    aes(x = pos1, y = y1, xend = pos2, yend = y1), 
                                    curvature = -1*now$curv[i],colour=DEL_color) 
    }
    idx= c(idx,2)
  }
  SV_plot = SV_plot + geom_point(data=now,size=.5,aes(x=pos1,y=y1)) + 
    geom_point(data=now,size=.5,aes(x=pos2,y=y1))
  
  # 绘制t2tINV类型SV
  now = df[df$SVtype == "t2tINV",]
  if (nrow(now) > 0){
    for (i in 1:nrow(now)){
      SV_plot = SV_plot + geom_curve(linewidth=arc_size,data = now[i,], 
                                    aes(x = pos1, y = y2, xend = pos2, yend = y2), 
                                    curvature = now$curv[i],colour=t2tINV_color) 
    }
    idx= c(idx,3)
  }
  SV_plot = SV_plot + geom_point(data=now,size=.5,aes(x=pos1,y=y2)) + 
    geom_point(data=now,size=.5,aes(x=pos2,y=y2))
  
  # 绘制h2hINV类型SV
  now = df[df$SVtype == "h2hINV",]
  if (nrow(now) > 0){
    for (i in 1:nrow(now)){
      SV_plot = SV_plot + geom_curve(linewidth=arc_size,data = now[i,], 
                                    aes(x = pos1, y = y2, xend = pos2, yend = y2), 
                                    curvature = -1*now$curv[i],colour=h2hINV_color) 
    }
    idx= c(idx,4)
  }
  SV_plot = SV_plot + geom_point(data=now,size=.5,aes(x=pos1,y=y2)) + 
    geom_point(data=now,size=.5,aes(x=pos2,y=y2))
  
  # 应用主题和坐标设置
  SV_plot = SV_plot + theme(axis.ticks.x=element_blank(),
                           panel.border = element_blank(),
                           axis.title.y=element_blank(),
                           axis.text.y=element_blank(),
                           axis.ticks.y=element_blank(),
                           # 隐藏X轴文本和标记
                           axis.text.x=element_blank(),
                           # 确保Y轴完全不可见
                           axis.line.y=element_blank()) + 
    scale_x_continuous(expand = c(0.01,0.01)) + 
    coord_cartesian(xlim=c(min_coord,max_coord))
  
  # 添加颜色图例
  idx = c(1,2,3,4)
  vals = c(DEL_color=DEL_color,DUP_color=DUP_color,
           t2tINV_color=t2tINV_color,h2hINV_color=h2hINV_color)
  labs = c('DEL','DUP',"t2tINV","h2hINV")
  
  SV_plot = SV_plot + scale_colour_manual(name = 'SV type', 
                                        values = c(DEL_color,DUP_color,t2tINV_color,h2hINV_color),
                                        labels = labs[idx]) + theme(legend.position="none")
  
  # 保存图像
  if(save_plot) {
    ggsave(output_file, plot = SV_plot, width = width/res, height = height/res, 
           dpi = res, units = "in")
  }
  
  # 返回ggplot对象
  return(SV_plot)
}
