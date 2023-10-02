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

    // To load a checkpoint, we need write access to ghost entries
    // Thus, we might need temporary vectors to comply with the required
    // partitioner layout (fully distributed vector with write access to ghost
    // entries). We use lazy allocation, if required.
    std::vector<VectorType>   tmp_vectors(vectors.size());
    std::vector<VectorType *> tmp_vectors_ptr;

    for (std::size_t i = 0; i < vectors.size(); ++i)
      {
        // With LA:d:V, we can grant write access using zero_out_ghost_values
        // (this wouldn't work with distributed PETSc vectors or similar)
        if (vectors[i]->has_ghost_elements())
          {
            vectors[i]->zero_out_ghost_values();
            tmp_vectors_ptr.emplace_back(vectors[i]);
          }
        else
          {
            // in case our vector carries anyway only local data, i.e., no ghost
            // values at all, we need to create a temporary vector
            tmp_vectors[i].reinit(partitioner);
            tmp_vectors_ptr.emplace_back(&tmp_vectors[i]);
          }
      }

    // triangulation.load(name + "-checkpoint.mesh");
    dealii::parallel::distributed::SolutionTransfer<dim, VectorType>
      solution_transfer(dof_handler);
    solution_transfer.deserialize(tmp_vectors_ptr);

    for (unsigned int i = 0; i < tmp_vectors_ptr.size(); ++i)
      {
        // Copy over data from the temporary vectors, if necessary
        if (!vectors[i]->has_ghost_elements())
          vectors[i]->copy_locally_owned_data_from(*(tmp_vectors_ptr[i]));
        // ... and update all ghost values
        vectors[i]->update_ghost_values();
      }

    // Last but not least, retrieve the time-stamp data
    std::ifstream file(name + "-checkpoint.metadata", std::ios::binary);
    boost::archive::binary_iarchive ia(file);
    ia >> t;
  }
} // namespace Utilities

DEAL_II_NAMESPACE_CLOSE
