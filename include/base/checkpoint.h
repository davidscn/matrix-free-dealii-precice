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
    const dealii::DoFHandler<dim>                           &dof_handler,
    const std::vector<const VectorType *>                   &vectors,
    const std::string                                       &name,
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
  template <int dim, typename VectorType>
  void
  load_checkpoint(const dealii::DoFHandler<dim> &dof_handler,
                  std::vector<VectorType *>     &vectors,
                  const std::string             &name,
                  double                        &t)
  {
    auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      dof_handler.locally_owned_dofs(),
      DoFTools::extract_locally_relevant_dofs(dof_handler),
      MPI_COMM_WORLD);

    // We need something temporarily because the reloading doesn't allow for
    // ghosted vectors
    std::vector<VectorType>   tmp_vectors(vectors.size());
    std::vector<VectorType *> tmp_vectors_ptr;

    for (auto &vec : tmp_vectors)
      {
        vec.reinit(partitioner);
        tmp_vectors_ptr.emplace_back(&vec);
      }

    // triangulation.load(name + "-checkpoint.mesh");
    dealii::parallel::distributed::SolutionTransfer<dim, VectorType>
      solution_transfer(dof_handler);

    solution_transfer.deserialize(tmp_vectors_ptr);

    for (auto &it : tmp_vectors_ptr)
      it->update_ghost_values();

    std::ifstream file(name + "-checkpoint.metadata", std::ios::binary);

    boost::archive::binary_iarchive ia(file);
    ia >> t;


    for (unsigned int i = 0; i < tmp_vectors_ptr.size(); ++i)
      {
        vectors[i]->copy_locally_owned_data_from(*(tmp_vectors_ptr[i]));
        vectors[i]->update_ghost_values();
      }
  }
} // namespace Utilities

DEAL_II_NAMESPACE_CLOSE
