This repository is a component of ChromShatter designed for detecting somatic structural variants from BAM files. It is a modified version of Delly, adapted to fit the overall workflow of the ChromShatter algorithm. 

# Install ChromShatter

You can build ChromShatter from source using a recursive clone and make.

`git clone --recursive https://github.com/PPSherry/ChromShatter.git`

`cd ChromShatter/`

`make all`

# How to use it

`chromshatter call [OPTIONS] -g <ref.fa> <tumor.sort.bam> <normal.sort.bam> -o output.tsv`

This command would output a .tsv file, which contain SV info and can be used in the SV visualization step.

License
-------
ChromShatter is distributed under the BSD 3-Clause license. Consult the accompanying [LICENSE](https://github.com/PPSherry/ChromShatter/blob/main/LICENSE) file for more details.

Credits
-------
[Delly](https://github.com/dellytools/delly) is a widely used bioinformatics tool for structure variant detection. [HTSlib](https://github.com/samtools/htslib) is heavily used for all genomic alignment and variant processing. [Boost](https://www.boost.org/) for various data structures and algorithms and [Edlib](https://github.com/Martinsos/edlib) for pairwise alignments using edit distance.

