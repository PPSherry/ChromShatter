#!/usr/bin/env bash

# Set paths for input, output, and data directories
INPUT_FILE="/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/SV_graph.simulated/SV_CNV_CaseID_table.unique.tsv"
OUTPUT_DIR="/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/SV_graph.simulated"
SV_BASE_DIR="/Volumes/T7-shield/TCGA/SV-Data" # set SV dir
CNV_BASE_DIR="/Volumes/T7-shield/TCGA/CNV" # set CNV dir

SAMPLE_RESULTS_DIR="${OUTPUT_DIR}/TCGA_single_sample"
SAMPLE_GRAPH_DIR="${OUTPUT_DIR}/TCGA_graph"
OUTPUT_FILE="${OUTPUT_DIR}/all_samples_results.tsv"
LOG_FILE="${OUTPUT_DIR}/processing_log.txt"

# Initialize or read existing output file
if [ ! -f "$OUTPUT_FILE" ]; then
    # We'll copy the header from the first sample's result and prepend case_id
    HEADER_INITIALIZED=false
else
    # Read samples already processed from the output file
    PROCESSED_SAMPLES=$(awk -F'\t' 'NR>1 {print $1}' "$OUTPUT_FILE" | sort | uniq)
fi

# Initialize or append to log file
if [ ! -f "$LOG_FILE" ]; then
    echo "Processing Log - $(date)" > "$LOG_FILE"
    echo "Case_ID,Status,Timestamp" >> "$LOG_FILE"
fi

# Process each sample in the input file
{
    # Skip header
    read -r header
    
    # Process each line
    while IFS=$'\t' read -r case_id sv_path cnv_path; do
        # Check if sample has already been processed
        if echo "$PROCESSED_SAMPLES" | grep -q "^$case_id$"; then
            echo "Sample $case_id already processed, skipping..."
            continue
        fi
        
        echo "Processing sample: $case_id"
        
        # Construct full paths
        full_sv_path="${SV_BASE_DIR}/${sv_path}"
        full_cnv_path="${CNV_BASE_DIR}/${cnv_path}"
        
        # Check if files exist
        if [ ! -f "$full_sv_path" ]; then
            echo "SV file not found: $full_sv_path" 
            echo "$case_id,FAILED_SV_MISSING,$(date +"%Y-%m-%d %H:%M:%S")" >> "$LOG_FILE"
            continue
        fi
        
        if [ ! -f "$full_cnv_path" ]; then
            echo "CNV file not found: $full_cnv_path"
            echo "$case_id,FAILED_CNV_MISSING,$(date +"%Y-%m-%d %H:%M:%S")" >> "$LOG_FILE"
            continue
        fi
        
        # Run ShatterSeek analysis
        echo "Running ShatterSeek for $case_id..."
        Rscript /Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/script/shatterSeek/shatterSeek_Script.R "$full_sv_path" "$full_cnv_path" "$case_id" "$SAMPLE_RESULTS_DIR" "$SAMPLE_GRAPH_DIR" >> /Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/SV_graph.simulated/R_log.txt 2>&1
        
        # Check if the R script execution was successful
        if [ $? -eq 0 ]; then
            status="SUCCESS"
        else
            status="FAILED_EXECUTION"
        fi
        
        # Get the sample results
        sample_results="${SAMPLE_RESULTS_DIR}/${case_id}_shatterSeek_results.tsv"
        
        if [ -f "$sample_results" ]; then
            # Initialize header if needed
            if [ "$HEADER_INITIALIZED" = false ] && [ ! -f "$OUTPUT_FILE" ]; then
                # Get the header from the first sample result and prepend case_id
                header_line=$(head -n 1 "$sample_results")
                echo -e "case_id\t$header_line" > "$OUTPUT_FILE"
                HEADER_INITIALIZED=true
            fi
            
            # Append sample results with case_id prepended to each row
            # Use -F to specify tab as field separator to ensure correct parsing
            awk -F'\t' -v cid="$case_id" -v OFS='\t' 'NR>1 {print cid,$0}' "$sample_results" >> "$OUTPUT_FILE"
            echo "Results for $case_id added to $OUTPUT_FILE"
            echo "$case_id,$status,$(date +"%Y-%m-%d %H:%M:%S")" >> "$LOG_FILE"
        else
            echo "No results file found for $case_id"
            echo "$case_id,FAILED_NO_RESULTS,$(date +"%Y-%m-%d %H:%M:%S")" >> "$LOG_FILE"
        fi
    done
} < "$INPUT_FILE"

echo "All samples processed. Results saved to $OUTPUT_FILE"
echo "Processing log saved to $LOG_FILE"
