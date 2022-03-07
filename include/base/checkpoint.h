#pragma once

#include <deal.II/base/utilities.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/tria.h>

// for implementing a check-pointing and restart mechanism.
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

DEAL_II_NAMESPACE_OPEN

namespace Utilities
{
  /**
   * @brief Create a checkpoint consisting of the triangulation, vectors, and some metadata
   *
   * @tparam dim
   * @tparam VectorType
   * @param triangulation
   * @param dof_handler
   * @param vectors
   * @param name
   * @param t
   */
  template <int dim, typename VectorType>
  void
  create_checkpoint(
    const dealii::parallel::distributed::Triangulation<dim> &triangulation,
    const dealii::DoFHandler<dim> &                          dof_handler,
    const std::vector<const VectorType *> &                  vectors,
    const std::string &                                      name,
    const double                                             t)
  {
    dealii::parallel::distributed::SolutionTransfer<dim, VectorType>
      solution_transfer(dof_handler);

    solution_transfer.prepare_for_serialization(vectors);

    triangulation.save(name + "-checkpoint.mesh");

    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::ofstream file(name + "-checkpoint.metadata", std::ios::binary);
        boost::archive::binary_oarchive oa(file);
        oa << t;
      }
  }


  /**
   * @brief Load a checkpoint previously stored using create_checkpoint
   *
   * @tparam dim
   * @tparam VectorType
   * @param triangulation
   * @param dof_handler
   * @param vectors
   * @param name
   * @param t
   */
  // template <int dim, typename VectorType>
  // void
  // load_checkpoint(
  //   const dealii::parallel::distributed::Triangulation<dim> &triangulation,
  //   const dealii::DoFHandler<dim> &                          dof_handler,
  //   const std::vector<VectorType *> &                        vectors,
  //   const std::string &                                      name,
  //   const double                                             t)
  // {
  //   parallel::distributed::SolutionTransfer<dim, VectorType> solution_transfer(
  //     dof_handler);

  //   solution_transfer.deserialize(vectors);

  //   for (auto &it : vectors)
  //     it.update_ghost_values();

  //   std::ifstream file(name + "-checkpoint.metadata", std::ios::binary);

  //   boost::archive::binary_iarchive ia(file);
  //   ia >> t;
  // }
} // namespace Utilities

DEAL_II_NAMESPACE_CLOSE
