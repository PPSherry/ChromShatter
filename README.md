# File Structure

## model
* ./model_data contains model training with only PCAWG labeled data (*using whether chromothripsis occurs in a chromosome as Y label*).  
* ./label_model_data contains model training with PCAWG labeled data and part of TCGA data.  
which can be surely labelled by the author of this research (*using whether shatterSeek output aligns with manual inspection as Y label*)  
* ./model_extended_data contains model training with PCAWG labeled data and part of TCGA data, the same as the dataset used in label_model (*but using whether chromothripsis occurs in a chromosome as Y label*)
* ./semi_supervised_learning contains model training through supervised learning (*data: PCAWG data + 50 TCGA cases/chromosomes which ShatterSeek regarded as negative falsely due to the criteria related to CNV*), and through semi_supervised learning (*data: PCAWG data + all TCAG data*)


## data
PCAWG data distribution: Chromothripsis-Pos 1987 chromosomes, 
Chromothripsis_Neg 861 chromosomes.  

TCGA data distribution: [According to ShatterSeek output]  
- High Confidence: 637 (2.86%)
- Low Confidence: 326 (1.46%)
- Not Significant: 21324 (95.68%)
