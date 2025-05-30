#define _SECURE_SCL 0
#define _SCL_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>

#define BOOST_DISABLE_ASSERTS
#define BOOST_UUID_RANDOM_PROVIDER_FORCE_POSIX

#ifdef OPENMP
#include <omp.h>
#endif

#ifdef PROFILE
#include "gperftools/profiler.h"
#endif

#include "version.h"
#include "chromshatter.h"
#include "filter.h"
// #include "classify.h"
// #include "merge.h"
// #include "tegua.h" // long read SV calling
// #include "coral.h" // copy number variant calling
// #include "asmode.h"

using namespace torali;


inline void
displayUsage() {
  std::cerr << "Usage: chromshatter <command> <arguments>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Short-read SV calling:" << std::endl;
  std::cerr << "    call         discover and genotype structural variants" << std::endl;
  // std::cerr << "    merge        merge structural variants across VCF/BCF files and within a single VCF/BCF file" << std::endl;
  // std::cerr << "    filter       filter somatic or germline structural variants" << std::endl;
  // std::cerr << std::endl;
  // std::cerr << "Long-read SV calling:" << std::endl;
  // std::cerr << "    lr           long-read SV discovery" << std::endl;
  // std::cerr << std::endl;
  //std::cerr << "Assembly-based SV calling (work-in-progress):" << std::endl;
  //std::cerr << "    asm          assembly SV site discovery" << std::endl;
  //std::cerr << std::endl;
  // std::cerr << "Copy-number variant calling:" << std::endl;
  // std::cerr << "    cnv          discover and genotype copy-number variants" << std::endl;
  // std::cerr << "    classify     classify somatic or germline copy-number variants" << std::endl;
  // std::cerr << std::endl;
  //std::cerr << "Deprecated:" << std::endl;
  //std::cerr << "    dpe          double paired-end signatures" << std::endl;
  //std::cerr << "    chimera      ONT chimera flagging" << std::endl;
  //std::cerr << std::endl;
}

int main(int argc, char **argv) {
    if (argc < 2) { 
      printTitle("ChromShatter");
      displayUsage();
      return 0;
    }

    if ((std::string(argv[1]) == "version") || (std::string(argv[1]) == "--version") || (std::string(argv[1]) == "--version-only") || (std::string(argv[1]) == "-v")) {
      std::cerr << "ChromShatter version: v" << chromShatterVersionNumber << std::endl;
      std::cerr << " using Boost: v" << BOOST_VERSION / 100000 << "." << BOOST_VERSION / 100 % 1000 << "." << BOOST_VERSION % 100 << std::endl;
      std::cerr << " using HTSlib: v" << hts_version() << std::endl;
      return 0;
    }
    else if ((std::string(argv[1]) == "help") || (std::string(argv[1]) == "--help") || (std::string(argv[1]) == "-h") || (std::string(argv[1]) == "-?")) {
      printTitle("ChromShatter");
      displayUsage();
      return 0;
    }
    else if ((std::string(argv[1]) == "warranty") || (std::string(argv[1]) == "--warranty") || (std::string(argv[1]) == "-w")) {
      displayWarranty();
      return 0;
    }
    else if ((std::string(argv[1]) == "license") || (std::string(argv[1]) == "--license") || (std::string(argv[1]) == "-l")) {
      bsd();
      return 0;
    }
    else if ((std::string(argv[1]) == "call")) {
      return chromShatter(argc-1,argv+1);
    }
    // else if ((std::string(argv[1]) == "lr")) {
    //   return tegua(argc-1,argv+1);
    // }
    // else if ((std::string(argv[1]) == "asm")) {
    //   return asmode(argc-1,argv+1);
    // }
    // else if ((std::string(argv[1]) == "cnv")) {
    //   return coral(argc-1,argv+1);
    // }
    // else if ((std::string(argv[1]) == "classify")) {
    //   return classify(argc-1,argv+1);
    // }
    else if ((std::string(argv[1]) == "filter")) {
      return filter(argc-1,argv+1);
    }
    // else if ((std::string(argv[1]) == "merge")) {
    //   return merge(argc-1,argv+1);
    // }
    std::cerr << "Unrecognized command " << std::string(argv[1]) << std::endl;
    return 1;
}

