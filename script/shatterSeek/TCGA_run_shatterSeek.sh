#!/usr/bin/env bash

# input file containing the list of TCGA samples
INPUT_FILE="/Volumes/T7-shield/CS-Bachelor-Thesis/TCGA-wgs/tcga_wgs_list.txt"
OUTPUT_FILE="/Volumes/T7-shield/CS-Bachelor-Thesis/TCGA-wgs/shatterSeek_results.tsv"

SV_DIR="/Volumes/T7-shield/TCGA/Data"
CNV_DIR="/Volumes/T7-shield/CNV"

# store the SV and CNV files
declare -A sv_files
declare -A cnv_files

{
    read header 
    while IFS=$'\t' read -r id filename md5 size state; do
        sample=$(echo "$filename" | cut -d'.' -f1)
        # match the sample's SV file
        if [[ "$filename" =~ raw_structural_variation\.vcf\.gz$ ]]; then
            sv_files["$sample"]="$filename"
        # match the sample's CNV file
        elif [[ "$filename" =~ ASCAT\.copy_number_variation\.seg\.txt$ ]]; then
            cnv_files["$sample"]="$filename"
        fi
    done
} < "$INPUT_FILE"

# empty the output file
> "$OUTPUT_FILE"

for sample in "${!sv_files[@]}"; do
    if [[ -n "${cnv_files[$sample]}" ]]; then
        sv_candidate="${sv_files[$sample]}"
        cnv_candidate="${cnv_files[$sample]}"

        # find the full path of the SV and CNV files
        sv_path=$(find "$SV_DIR" -type f -name "$sv_candidate" 2>/dev/null)
        cnv_path=$(find "$CNV_DIR" -type f -name "$cnv_candidate" 2>/dev/null)

        if [[ -n "$sv_path" && -n "$cnv_path" ]]; then
            # call R script to detect chromothripsis
            result=$(Rscript SV_CNV_transformation_TCGA.R "$sv_path" "$cnv_path")
            echo -e "$result" >> "$OUTPUT_FILE"
        else
            echo "can't find $sample SV/CNV files" >> "$OUTPUT_FILE"
        fi
    else
        echo "sample $sample don't have corresponding CNV file" >> "$OUTPUT_FILE"
    fi
done
