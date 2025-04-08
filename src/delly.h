#ifndef DELLY_H
#define DELLY_H


#include <iostream>
#include <fstream>

// boost libraries
#include <boost/container/flat_set.hpp>
#include <boost/unordered_map.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/icl/split_interval_map.hpp>
#include <boost/tokenizer.hpp>
#include <boost/functional/hash.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem.hpp>

#include <htslib/faidx.h>
#include <htslib/vcf.h>
#include <htslib/sam.h>

#include "version.h"
#include "util.h"
#include "bolog.h"
#include "tags.h"
#include "coverage.h"
#include "msa.h" 
#include "split.h"
#include "shortpe.h"
// #include "modvcf.h" // Not needed for SV table output

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>
#include <stdio.h>

namespace torali
{

  // Config arguments
  struct Config {
    uint16_t minMapQual;
    uint16_t minTraQual;
    uint16_t minGenoQual;
    uint16_t madCutoff;
    uint16_t madNormalCutoff;
    int32_t nchr;
    int32_t minimumFlankSize;
    int32_t indelsize;
    int32_t minConsWindow;
    uint32_t graphPruning;
    uint32_t minRefSep;
    uint32_t maxReadSep;
    uint32_t minClip;
    uint32_t maxGenoReadCount;
    uint32_t minCliqueSize;
    float flankQuality;
    bool hasExcludeFile;
    bool hasVcfFile;
    bool hasDumpFile;
    std::set<int32_t> svtset;
    DnaScore<int> aliscore;
    boost::filesystem::path outfile;
    boost::filesystem::path vcffile;
    boost::filesystem::path genome;
    boost::filesystem::path exclude;
    boost::filesystem::path dumpfile;
    std::vector<boost::filesystem::path> files;
    std::vector<std::string> sampleName;
  };

  inline bool isValidChromosome(const std::string& chr) {
    if (chr.substr(0, 3) == "chr") {
      std::string num = chr.substr(3);
      if (num.length() > 0) {
        if (num == "X") return true;
        try {
          int chrNum = std::stoi(num);
          return (chrNum >= 1 && chrNum <= 22);
        } catch (...) {
          return false;
        }
      }
    }
    try {
      int chrNum = std::stoi(chr);
      return (chrNum >= 1 && chrNum <= 22);
    } catch (...) {
      return (chr == "X");
    }
    return false;
  }

  // Determine if an SV has sufficient support to pass quality filters
  inline bool _qualityPass(StructuralVariantRecord const& svRec) {
    if (svRec.chr == svRec.chr2) {
      // Intra-chromosomal
      return !(((svRec.peSupport < 3) || (svRec.peMapQuality < 20)) && 
               ((svRec.srSupport < 3) || (svRec.srMapQuality < 20)));
    } else {
      // Inter-chromosomal
      return !(((svRec.peSupport < 5) || (svRec.peMapQuality < 20)) && 
               ((svRec.srSupport < 5) || (svRec.srMapQuality < 20)));
    }
  }

  // For comparing two samples to identify somatic SVs
  template<typename TConfig, typename TStructuralVariantRecord>
  inline void
  outputSomaticSVTable(TConfig const& c, std::vector<TStructuralVariantRecord> const& svs1, std::vector<TStructuralVariantRecord> const& svs2)
  {
    // Open file for writing
    std::ofstream svFile;
    svFile.open(c.outfile.string().c_str());
    
    // Get BAM header for chromosome names
    samFile* samfile = sam_open(c.files[0].string().c_str(), "r");
    bam_hdr_t* hdr = sam_hdr_read(samfile);
    
    // Write header
    svFile << "CHR1\tPOS1\tCHR2\tPOS2\tSVTYPE\tSTRAND1\tSTRAND2\tPRECISE\tFILTER\tVARIANT_TYPE" << std::endl;
    
    // Process SVs from sample1 (tumor)
    for(typename std::vector<TStructuralVariantRecord>::const_iterator svIter = svs1.begin(); svIter != svs1.end(); ++svIter) {
      // Skip if no support
      if ((svIter->srSupport == 0) && (svIter->peSupport == 0)) continue;
      
      // Get chromosome names
      std::string chr1 = hdr->target_name[svIter->chr];
      std::string chr2 = hdr->target_name[svIter->chr2];
      
      // Get SV type and orientation
      std::string svtype = _addID(svIter->svt);
      std::string orientation = _addOrientation(svIter->svt);
      
      // Get strand information from orientation
      std::string strand1, strand2;
      if (orientation == "3to3") { strand1 = "+"; strand2 = "+"; }  // head-to-head inversion (+/+)
      else if (orientation == "5to5") { strand1 = "-"; strand2 = "-"; }  // tail-to-tail inversion (-/-)
      else if (orientation == "3to5") { strand1 = "+"; strand2 = "-"; }  // deletion-like (+/-)
      else if (orientation == "5to3") { strand1 = "-"; strand2 = "+"; }  // duplication-like (-/+)
      else { strand1 = "."; strand2 = "."; } // insertion
      
      // Quality filter
      std::string filter = _qualityPass(*svIter) ? "PASS" : "LowQual";
      
      // Find matching SV in normal sample (sample2)
      bool is_somatic = true;  // Assume somatic until proven otherwise
      bool found_match = false;
      
      // Define distance threshold for finding nearby SVs
      const int max_distance = 500;  // Maximum distance to consider for matching normal SVs
      
      // Define variables to track best match
      double best_match_score = 0.0;
      const TStructuralVariantRecord* best_match = nullptr;

      // Use lower_bound on sorted svs2 to quickly find potential matching SVs in normal sample
      // Create a temporary SV for binary search
      TStructuralVariantRecord searchSV;
      searchSV.chr = svIter->chr;
      searchSV.chr2 = svIter->chr2;
      searchSV.svt = svIter->svt;
      searchSV.svStart = std::max(0, svIter->svStart - max_distance);  // Search start position (tumor SV - 500bp)
      
      // Use binary search to quickly locate potential matches in normal sample
      typename std::vector<TStructuralVariantRecord>::const_iterator lowerBound = 
          std::lower_bound(svs2.begin(), svs2.end(), searchSV);
      
      // Only check SVs from lower_bound that are within 1000bp window around tumor SV
      for(; lowerBound != svs2.end(); ++lowerBound) {
        // Stop searching if we've moved beyond the search window
        if (lowerBound->chr != svIter->chr || lowerBound->svStart > svIter->svStart + max_distance)
          break;
        
        // Must be on same chromosome
        if (svIter->chr2 != lowerBound->chr2) 
          continue;
        
        // Must be same SV type
        if (svIter->svt != lowerBound->svt) 
          continue;
        
        // Calculate breakpoint distances
        int start_distance = std::abs(svIter->svStart - lowerBound->svStart);
        int end_distance = std::abs(svIter->svEnd - lowerBound->svEnd);
        
        // Check if breakpoints are within distance threshold
        if (start_distance > max_distance || end_distance > max_distance)
          continue;
        
        // Calculate confidence interval overlap
        int start_min1 = svIter->svStart + svIter->ciposlow;
        int start_max1 = svIter->svStart + svIter->ciposhigh;
        int start_min2 = lowerBound->svStart + lowerBound->ciposlow;
        int start_max2 = lowerBound->svStart + lowerBound->ciposhigh;
        
        int end_min1 = svIter->svEnd + svIter->ciendlow;
        int end_max1 = svIter->svEnd + svIter->ciendhigh;
        int end_min2 = lowerBound->svEnd + lowerBound->ciendlow;
        int end_max2 = lowerBound->svEnd + lowerBound->ciendhigh;
        
        // Check for confidence interval overlap
        bool start_overlap = (start_min1 <= start_max2 && start_min2 <= start_max1);
        bool end_overlap = (end_min1 <= end_max2 && end_min2 <= end_max1);
        
        if (start_overlap && end_overlap) {
          // Calculate a match score based on breakpoint distance and support
          double match_score = 1.0 / (1.0 + start_distance + end_distance);
          
          // If better match than current best, update
          if (match_score > best_match_score) {
            best_match_score = match_score;
            best_match = &(*lowerBound);
            found_match = true;
          }
        }
      }
      
      // If we found a matching SV in normal sample
      if (found_match && best_match) {
        // Check if normal sample has sufficient support to call this germline
        bool normal_good_support = (best_match->peSupport >= 2 || best_match->srSupport >= 2);
        
        if (normal_good_support) {
          is_somatic = false;  // If normal has good support, it's likely germline
        } else {
          // Even if normal support is weak but the confident intervals overlap well,
          // it's likely the same SV, just with less coverage in normal
          is_somatic = true;
        }
      } else {
        // No matching SV found in normal - likely a true somatic event
        is_somatic = true;
      }
      
      // Output to file
      svFile << chr1 << "\t" 
             << svIter->svStart << "\t" 
             << chr2 << "\t" 
             << svIter->svEnd << "\t" 
             << svtype << "\t" 
             << strand1 << "\t" 
             << strand2 << "\t" 
             << (svIter->precise ? "PRECISE" : "IMPRECISE") << "\t" 
             << filter << "\t" 
             << (is_somatic ? "SOMATIC" : "GERMLINE") << std::endl;
    }
    
    // Clean up
    bam_hdr_destroy(hdr);
    sam_close(samfile);
    svFile.close();
  }

  template<typename TConfigStruct>
  inline int dellyRun(TConfigStruct& c) {
#ifdef PROFILE
    ProfilerStart("delly.prof");
#endif

    // Collect all promising structural variants
    typedef std::vector<StructuralVariantRecord> TVariants;
    TVariants svs;
    
    // Open header
    samFile* samfile = sam_open(c.files[0].string().c_str(), "r");
    bam_hdr_t* hdr = sam_hdr_read(samfile);
    
    // Exclude intervals
    typedef boost::icl::interval_set<uint32_t> TChrIntervals;
    typedef std::vector<TChrIntervals> TRegionsGenome;
    TRegionsGenome validRegions;
    if (!_parseExcludeIntervals(c, hdr, validRegions)) {
      std::cerr << "Delly couldn't parse exclude intervals!" << std::endl;
      bam_hdr_destroy(hdr);
      sam_close(samfile);
      return 1;
    }
    
    // Debug code
    //for(int32_t refIndex = 0; refIndex < hdr->n_targets; ++refIndex) {
    //for(typename TChrIntervals::const_iterator vRIt = validRegions[refIndex].begin(); vRIt != validRegions[refIndex].end(); ++vRIt) {
    //std::cerr << std::string(hdr->target_name[refIndex]) << "\t" << vRIt->lower() << "\t" << vRIt->upper() << std::endl;
    //}
    //}
    
    // Create library objects
    typedef std::vector<LibraryInfo> TSampleLibrary;
    TSampleLibrary sampleLib(c.files.size(), LibraryInfo());
    getLibraryParams(c, validRegions, sampleLib);
    for(uint32_t i = 0; i<sampleLib.size(); ++i) {
      if (sampleLib[i].rs == 0) {
	std::cerr << "Sample has not enough data to estimate library parameters! File: " << c.files[i].string() << std::endl;
	bam_hdr_destroy(hdr);
	sam_close(samfile);
	return 1;
      }
    }
    
    // SV Discovery
    if (!c.hasVcfFile) {
      // Split-read SVs
      typedef std::vector<StructuralVariantRecord> TVariants;
      TVariants srSVs;
      
      // SR Store
      {
	typedef std::pair<int32_t, std::size_t> TPosRead;
	typedef boost::unordered_map<TPosRead, int32_t> TPosReadSV;
	typedef std::vector<TPosReadSV> TGenomicPosReadSV;
	TGenomicPosReadSV srStore(c.nchr, TPosReadSV());
	scanPEandSR(c, validRegions, svs, srSVs, srStore, sampleLib);
	
	// Assemble split-read calls
	assembleSplitReads(c, validRegions, srStore, srSVs);
      }

      // Sort and merge PE and SR calls
      mergeSort(svs, srSVs);
    } 
    // Clean-up
    bam_hdr_destroy(hdr);
    sam_close(samfile);

    // Re-number SVs
    sort(svs.begin(), svs.end());
    uint32_t cliqueCount = 0;
    for(typename TVariants::iterator svIt = svs.begin(); svIt != svs.end(); ++svIt, ++cliqueCount) svIt->id = cliqueCount;
    
    // Process and output SVs
    boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();
    std::cerr << '[' << boost::posix_time::to_simple_string(now) << "] " << "Processing and outputting SVs" << std::endl;
    
    // Check if we have two files - only tumor-normal comparison is supported
    if (c.files.size() != 2) {
      std::cerr << "Error: This program requires exactly two BAM files (tumor and normal)." << std::endl;
      return 1;
    }
    
    // Discover SVs in second sample (normal)
    TVariants svs2;
    
    // Open header for second file
    samFile* samfile2 = sam_open(c.files[1].string().c_str(), "r");
    bam_hdr_t* hdr2 = sam_hdr_read(samfile2);
    
    // Perform discovery on second sample
    // Split-read SVs
    TVariants srSVs2;
    
    // SR Store
    {
      typedef std::pair<int32_t, std::size_t> TPosRead;
      typedef boost::unordered_map<TPosRead, int32_t> TPosReadSV;
      typedef std::vector<TPosReadSV> TGenomicPosReadSV;
      TGenomicPosReadSV srStore2(c.nchr, TPosReadSV());
      scanPEandSR(c, validRegions, svs2, srSVs2, srStore2, sampleLib);
      
      // Assemble split-read calls
      assembleSplitReads(c, validRegions, srStore2, srSVs2);
    }
    
    // Sort and merge PE and SR calls
    mergeSort(svs2, srSVs2);
    
    // Clean-up
    bam_hdr_destroy(hdr2);
    sam_close(samfile2);
    
    // Re-number SVs
    sort(svs2.begin(), svs2.end());
    cliqueCount = 0;
    for(typename TVariants::iterator svIt = svs2.begin(); svIt != svs2.end(); ++svIt, ++cliqueCount) svIt->id = cliqueCount;
    
    // Generate SV table with somatic information
    outputSomaticSVTable(c, svs, svs2);
    
    // Output library statistics
    now = boost::posix_time::second_clock::local_time();
    std::cerr << '[' << boost::posix_time::to_simple_string(now) << "] " << "Sample statistics" << std::endl;
    for(uint32_t file_c = 0; file_c < c.files.size(); ++file_c) {
      std::cerr << "Sample:" << c.sampleName[file_c] << ",ReadSize=" << sampleLib[file_c].rs << ",Median=" << sampleLib[file_c].median << ",MAD=" << sampleLib[file_c].mad << ",UniqueDiscordantPairs=" << sampleLib[file_c].abnormal_pairs << std::endl;
    }
    
#ifdef PROFILE
    ProfilerStop();
#endif
  
    // End
    now = boost::posix_time::second_clock::local_time();
    std::cerr << '[' << boost::posix_time::to_simple_string(now) << "] Done." << std::endl;;
    return 0;
  }


  int delly(int argc, char **argv) {
    Config c;
    c.madNormalCutoff = 5;
    
    // Define generic options
    std::string svtype;
    boost::program_options::options_description generic("Generic options");
    generic.add_options()
      ("help,?", "show help message")
      ("svtype,t", boost::program_options::value<std::string>(&svtype)->default_value("ALL"), "SV type to compute [DEL, INS, DUP, INV, BND, ALL]")
      ("genome,g", boost::program_options::value<boost::filesystem::path>(&c.genome), "genome fasta file")
      ("exclude,x", boost::program_options::value<boost::filesystem::path>(&c.exclude), "file with regions to exclude")
      ("outfile,o", boost::program_options::value<boost::filesystem::path>(&c.outfile), "SV table output file")
      ;
    
    boost::program_options::options_description disc("Discovery options");
    disc.add_options()
      ("map-qual,q", boost::program_options::value<uint16_t>(&c.minMapQual)->default_value(1), "min. paired-end (PE) mapping quality")
      ("qual-tra,r", boost::program_options::value<uint16_t>(&c.minTraQual)->default_value(20), "min. PE quality for translocation")
      ("mad-cutoff,s", boost::program_options::value<uint16_t>(&c.madCutoff)->default_value(9), "insert size cutoff, median+s*MAD (deletions only)")
      ("minclip,c", boost::program_options::value<uint32_t>(&c.minClip)->default_value(25), "min. clipping length")
      ("min-clique-size,z", boost::program_options::value<uint32_t>(&c.minCliqueSize)->default_value(2), "min. PE/SR clique size")
      ("minrefsep,m", boost::program_options::value<uint32_t>(&c.minRefSep)->default_value(25), "min. reference separation")
      ("maxreadsep,n", boost::program_options::value<uint32_t>(&c.maxReadSep)->default_value(40), "max. read separation")
      ;

    // Define hidden options
    boost::program_options::options_description hidden("Hidden options");
    hidden.add_options()
      ("input-file", boost::program_options::value< std::vector<boost::filesystem::path> >(&c.files), "input file")
      ("pruning,j", boost::program_options::value<uint32_t>(&c.graphPruning)->default_value(1000), "PE graph pruning cutoff")
      ("cons-window,w", boost::program_options::value<int32_t>(&c.minConsWindow)->default_value(100), "consensus window")
      ;
    
    boost::program_options::positional_options_description pos_args;
    pos_args.add("input-file", -1);
    
    // Set the visibility
    boost::program_options::options_description cmdline_options;
    cmdline_options.add(generic).add(disc).add(hidden);
    boost::program_options::options_description visible_options;
    visible_options.add(generic).add(disc);
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(cmdline_options).positional(pos_args).run(), vm);
    boost::program_options::notify(vm);

    // Check command line arguments
    if ((vm.count("help")) || (!vm.count("input-file")) || (!vm.count("genome"))) { 
      std::cerr << std::endl;
      std::cerr << "Usage: delly " << argv[0] << " [OPTIONS] -g <ref.fa> <tumor.sort.bam> <normal.sort.bam>" << std::endl;
      std::cerr << "For somatic SV calling: first sample is tumor, second sample is control" << std::endl;
      std::cerr << visible_options << "\n";
      return 0;
    }

    // SV types to compute?
    if (!_svTypesToCompute(c, svtype)) {
      std::cerr << "Please specify a valid SV type, i.e., -t INV or -t DEL,INV without spaces." << std::endl;
      return 1;
    }
    
    // Check input files (exactly 2 files required)
    if (c.files.size() != 2) {
      std::cerr << "Error: Exactly two input BAM files are required (tumor and normal)." << std::endl;
      return 1;
    }
    
    // Exclude file
    if (vm.count("exclude")) {
      if (!(boost::filesystem::exists(c.exclude) && boost::filesystem::is_regular_file(c.exclude) && boost::filesystem::file_size(c.exclude))) {
	std::cerr << "Exclude file is missing: " << c.exclude.string() << std::endl;
	return 1;
      }
      c.hasExcludeFile = true;
    } else c.hasExcludeFile = false;
    
    // Input VCF (not supported in this modified version)
    c.hasVcfFile = false;
    
    // Default outfile
    if (!vm.count("outfile")) c.outfile = "somatic_sv_calls.tsv";
    
    // Show cmd
    boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();
    std::cerr << '[' << boost::posix_time::to_simple_string(now) << "] ";
    std::cerr << "delly ";
    for(int i=0; i<argc; ++i) { std::cerr << argv[i] << ' '; }
    std::cerr << std::endl;
    
    // Check reference
    if (!(boost::filesystem::exists(c.genome) && boost::filesystem::is_regular_file(c.genome) && boost::filesystem::file_size(c.genome))) {
      std::cerr << "Reference file is missing: " << c.genome.string() << std::endl;
      return 1;
    } else {
      faidx_t* fai = fai_load(c.genome.string().c_str());
      if (fai == NULL) {
	if (fai_build(c.genome.string().c_str()) == -1) {
	  std::cerr << "Fail to open genome fai index for " << c.genome.string() << std::endl;
	  return 1;
	} else fai = fai_load(c.genome.string().c_str());
      }
      fai_destroy(fai);
    }

    // Check input files
    c.sampleName.resize(c.files.size());
    c.nchr = 0;
    for(unsigned int file_c = 0; file_c < c.files.size(); ++file_c) {
      if (!(boost::filesystem::exists(c.files[file_c]) && boost::filesystem::is_regular_file(c.files[file_c]) && boost::filesystem::file_size(c.files[file_c]))) {
	std::cerr << "Alignment file is missing: " << c.files[file_c].string() << std::endl;
	return 1;
      }
      samFile* samfile = sam_open(c.files[file_c].string().c_str(), "r");
      if (samfile == NULL) {
	std::cerr << "Fail to open file " << c.files[file_c].string() << std::endl;
	return 1;
      }
      hts_idx_t* idx = sam_index_load(samfile, c.files[file_c].string().c_str());
      if (idx == NULL) {
	std::cerr << "Fail to open index for " << c.files[file_c].string() << std::endl;
	return 1;
      }
      bam_hdr_t* hdr = sam_hdr_read(samfile);
      if (hdr == NULL) {
	std::cerr << "Fail to open header for " << c.files[file_c].string() << std::endl;
	return 1;
      }
      if (!c.nchr) c.nchr = hdr->n_targets;
      else {
	if (c.nchr != hdr->n_targets) {
	  std::cerr << "BAM files have different number of chromosomes!" << std::endl;
	  return 1;
	}
      }
      faidx_t* fai = fai_load(c.genome.string().c_str());
      for(int32_t refIndex=0; refIndex < hdr->n_targets; ++refIndex) {
	std::string tname(hdr->target_name[refIndex]);
  // only check the existence of chr1-22, chrX in the reference genome
	if (isValidChromosome(tname)) {
	  if (!faidx_has_seq(fai, tname.c_str())) {
	    std::cerr << "BAM file chromosome " << hdr->target_name[refIndex] << " is NOT present in your reference file " << c.genome.string() << std::endl;
	    return 1;
	  }
	}
      }
      fai_destroy(fai);
      std::string sampleName = "unknown";
      getSMTag(std::string(hdr->text), c.files[file_c].stem().string(), sampleName);
      c.sampleName[file_c] = sampleName;
      bam_hdr_destroy(hdr);
      hts_idx_destroy(idx);
      sam_close(samfile);
    }
    checkSampleNames(c);
    
    // Run the main program
    c.aliscore = DnaScore<int>(5, -4, -10, -1);
    c.flankQuality = 0.95;
    c.minimumFlankSize = 13;
    c.indelsize = 1000;
    return dellyRun(c);
  }

}

#endif