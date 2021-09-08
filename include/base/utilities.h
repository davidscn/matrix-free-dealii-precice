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
  /**
   * @brief Create a directory on the filesystem if the permission allows it
   *
   * @param[in] pathname Name of the path where the directory should be created
   * @param[in] mode Mode used to create the directory
   *
   * @return int An integer which indicates if the creation was successfull (returns 0)
   */
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



  /**
   * @brief Prints the configuration information the codes run with such as the
   *        number of MPI ranks and the build mode
   *
   * @param[in] stream The output stream to which the information should be printed
   */
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



  /**
   * @brief Rounds a given number up to a defined precision.
   *        Example:
   *        round_to_precision(9.99941, 3) would return 10 whereas
   *        round_to_precision(9.99941, 4) would return 9.9994
   *
   * @tparam Number Type of the number to be rounded
   *
   * @param[in] number The number to be rounded
   * @param[in] precision The precision up to which to round the number
   *
   * @return The rounded number
   */
  template <typename Number>
  constexpr Number
  round_to_precision(const Number number, const int precision)
  {
    Assert(precision > 0, ExcInternalError());
    const std::size_t factor         = std::pow(10, precision);
    const Number      rounded_number = std::round(number * factor) / factor;
    return rounded_number;
  }
} // namespace Utilities


DEAL_II_NAMESPACE_CLOSE
