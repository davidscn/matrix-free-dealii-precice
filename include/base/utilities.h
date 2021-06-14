#pragma once

#include <deal.II/base/config.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/utilities.h>

#include <base/version.h>
#include <sys/stat.h>

DEAL_II_NAMESPACE_OPEN

namespace Utilities
{
  int
  create_directory(std::string  pathname,
                   const mode_t mode = (S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH |
                                        S_IXOTH))
  {
    // force trailing / so we can handle everything in loop
    if (pathname[pathname.size() - 1] != '/')
      {
        pathname += '/';
      }

    size_t pre = 0;
    size_t pos;

    while ((pos = pathname.find_first_of('/', pre)) != std::string::npos)
      {
        const std::string subdir = pathname.substr(0, pos++);
        pre                      = pos;

        // if leading '/', first string is 0 length
        if (subdir.size() == 0)
          continue;

        int mkdir_return_value;
        if ((mkdir_return_value = mkdir(subdir.c_str(), mode)) &&
            (errno != EEXIST))
          return mkdir_return_value;
      }

    return 0;
  }


  void
  print_configuration(ConditionalOStream &stream)
  {
    const int n_tasks = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    const int n_threads               = dealii::MultithreadInfo::n_threads();
    const unsigned int n_vect_doubles = VectorizedArray<double>::size();
    const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;

    stream
      << "-----------------------------------------------------------------------------"
      << std::endl
#ifdef DEBUG
      << "--     . running in DEBUG mode" << std::endl
#else
      << "--     . running in OPTIMIZED mode" << std::endl
#endif
      << "--     . running with " << n_tasks << " MPI process"
      << (n_tasks == 1 ? "" : "es") << std::endl;

    if (n_threads > 1)
      stream << "--     . using " << n_threads << " threads "
             << (n_tasks == 1 ? "" : "each") << std::endl;

    stream << "--     . vectorization over " << n_vect_doubles
           << " doubles = " << n_vect_bits << " bits (";

    stream << System::get_current_vectorization_level();

    stream << ")" << std::endl;
    stream << "--     . version " << GIT_TAG << " (revision " << GIT_SHORTREV
           << " on branch " << GIT_BRANCH << ")" << std::endl;
    stream << "--     . deal.II " << DEAL_II_PACKAGE_VERSION << " (revision "
           << DEAL_II_GIT_SHORTREV << " on branch " << DEAL_II_GIT_BRANCH << ")"
           << std::endl;
    stream
      << "-----------------------------------------------------------------------------"
      << std::endl
      << std::endl;
  }
} // namespace Utilities


DEAL_II_NAMESPACE_CLOSE
