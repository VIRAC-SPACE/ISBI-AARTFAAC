#include "Common/Config.h"

#include "Common/Exceptions/AddressTranslator.h"
#include "Common/Exceptions/Backtrace.h"
#include "Common/Exceptions/SymbolTable.h"

#include <map>
#include <cstdlib>
#include <cstring>
#include <boost/shared_ptr.hpp>

#ifdef HAVE_BFD
# ifdef __GNUG__
#  include <cxxabi.h>
# endif
#endif


#ifdef HAVE_BFD
  namespace
  {
    // Map of symbol tables.
    // Use the load address of the shared object or executable as key.
    typedef std::map< bfd_vma, boost::shared_ptr<SymbolTable> > SymbolTableMap;

    // The map of symbol tables is implemented as a Meyers singleton.
    // Use a lock to make access thread-safe.
    SymbolTableMap& theSymbolTableMap()
    {
      static SymbolTableMap symTabMap;
      return symTabMap;
    }
  }
#endif

  AddressTranslator::AddressTranslator()
  {
  }

  AddressTranslator::~AddressTranslator()
  {
  }

  void AddressTranslator::operator()(std::vector<Backtrace::TraceLine>& trace,
                                     void* const* addr, int size) 
  {
    // Initialize \a size elements of vector \a trace to avoid reallocations.
    trace.resize(size);
  
#ifdef HAVE_BFD
    std::lock_guard<std::mutex> lock(mutex);

    for (int i = 0; i < size; i++) {

      pc = reinterpret_cast<bfd_vma>(addr[i]);

      // Walk through list of shared objects (ref. man dl_iterate_phdr(3)).
      // The callback function #find_matching_file() will set #bfdFile and
      // #base_addr.
      dl_iterate_phdr(find_matching_file, this);

      // Lookup the SymbolTable using #base_addr as key. Create a new entry
      // if the symbol table of #bfdFile is not yet present in the map.
      SymbolTableMap::iterator it = theSymbolTableMap().find(base_addr);
      if(it == theSymbolTableMap().end()) {
        it = theSymbolTableMap().insert
          (std::make_pair(base_addr, boost::shared_ptr<SymbolTable>(new SymbolTable(bfdFile)))).first;
      }

      // Get the BFD-handle of the matching SymbolTable object.
      bfd* abfd = it->second->getBfd();
      if (!abfd) continue;

      // Get the handle to the symbols of the matching SymbolTable.
      syms = it->second->getSyms();
      if (!syms) continue;

      // Calculate offset address inside the matching shared object.
      pc -= base_addr;
      found = false;

      // Call the function #find_address_in_section() for each section
      // attached to the BFD \a abfd. This function will set #filename,
      // #functionname, #line and #found.
      bfd_map_over_sections(abfd, find_address_in_section, this);

      if (found) {
        // Unwind inlined functions to get the source information of the first
        // non-inlined function. In general, this will improve the usefulness
        // of the backtrace information.
        while(bfd_find_inliner_info(abfd, &filename, &functionname, &line));
        
        if (functionname && *functionname) {
# ifdef __GNUG__
          char* realname = abi::__cxa_demangle(functionname, 0, 0, 0);
          if (realname) {
            trace[i].function = realname;
            free(realname);
          }
          else {
            trace[i].function = functionname;
          }
# else
          trace[i].function = functionname;
# endif
        }
        if (filename) {
          const char* h = strrchr(filename,'/');
          if (h) {
            filename = h+1;
          }
          trace[i].file = filename;
        }
        trace[i].line = line;
      }
    }
#else
    (void) addr;  // suppress `unused parameter' warning
#endif
    return;
  }

  //##----  P r i v a t e   f u n c t i o n s  ----##//

#ifdef HAVE_BFD
  std::mutex AddressTranslator::mutex;

  int AddressTranslator::find_matching_file(dl_phdr_info* info,
                                            size_t      /*size*/,
                                            void*         data)
  {
    AddressTranslator* obj = static_cast<AddressTranslator*>(data);
    return obj->do_find_matching_file(info);
  }

  int AddressTranslator::do_find_matching_file(dl_phdr_info* info)
  {
    // This code is based on util/backtrace-symbols.c from Cairo
    // (http://cairographics.org).
    const ElfW(Phdr) *phdr = info->dlpi_phdr;
    for (int n = info->dlpi_phnum; --n >= 0; phdr++) {
      if (phdr->p_type == PT_LOAD) {
        ElfW(Addr) vaddr = phdr->p_vaddr + info->dlpi_addr;
        if (pc >= vaddr && pc < vaddr + phdr->p_memsz) {
          // We found a match
          if(info->dlpi_name && *info->dlpi_name) {
            bfdFile = info->dlpi_name;
          } else {
            bfdFile = "/proc/self/exe";
          }
          base_addr = info->dlpi_addr;
          return 1;
        }
      }
    }
    return 0;
  }
    

  void AddressTranslator::find_address_in_section(bfd*      abfd,
						  asection* section,
						  void*     data)
  {
    AddressTranslator* obj = static_cast<AddressTranslator*>(data);
    obj->do_find_address_in_section(abfd, section);
  }

  void AddressTranslator::do_find_address_in_section(bfd*       abfd, 
						     asection*  section)
  {
    if (found)
      return;

    if ((bfd_get_section_flags (abfd, section) & SEC_ALLOC) == 0)
      return;

    bfd_vma vma = bfd_get_section_vma (abfd, section);
    if (pc < vma)
      return;

    bfd_size_type size = bfd_get_section_size (section);
    if (pc >= vma + size)
      return;

    found = bfd_find_nearest_line (abfd, section, syms, pc - vma, 
				   &filename, &functionname, &line);
  }

#endif
