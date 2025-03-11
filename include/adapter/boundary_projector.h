#pragma once

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <adapter/boundary_mass_operator.h>

using namespace dealii;

/**
 * This class is essentially a re-implementation of
 * VectorTools::project_boundary_values. However, as opposed to the VectorTools
 * function, this class works in parallel and uses a MatrixFree implementation.
 * Although the problem itself is just an interface problem, a global system (at
 * least considering the size of the vectors) is solved. Irrelevant DoFs are,
 * set to zero and not touched when solving the system, i.e., the operators work
 * solely on the relevant part of the problem.
 */
template <int dim,
          int n_components    = 1,
          typename VectorType = LinearAlgebra::distributed::Vector<double>,
          typename VectorizedArrayType =
            VectorizedArray<typename VectorType::value_type>>
class BoundaryProjector
{
public:
  /**
   * The operator for the boundary
   */
  using MatrixType = MatrixFreeOperators::
    BoundaryMassOperator<dim, n_components, VectorType, VectorizedArrayType>;

  /**
   * @brief Constructor which initializes the class
   *
   * @param[in] interface_id boundary_id of the interface to project the
   *            residual onto
   * @param[in] data The underlying MatrixFree object. Note that MatrixFree
   *            cannot handle inhomogenous constraints and in case the interface
   *            values are constrained (due to the initialization of MatrixFree
   *            itself) the values are set to zero. Therefore, make sure to pass
   *            in an appropriate MatrixFree object here
   */
  BoundaryProjector(
    const types::boundary_id interface_id,
    std::shared_ptr<const MatrixFree<dim, double, VectorizedArrayType>> data);

  /**
   * @brief project Perform the actual projection step
   *
   * @param[in/out] residual The global dof vector holding the data to project.
   *                The result is passed into the same vector again so that the
   *                relevant DoFs hold the respective projection
   */
  void
  project(VectorType &residual);

private:
  const types::boundary_id interface_id;
  std::shared_ptr<const MatrixFree<dim, double, VectorizedArrayType>> data;
  IndexSet   boundary_indices;
  VectorType rhs;
};



template <int dim,
          int n_components,
          typename VectorType,
          typename VectorizedArrayType>
BoundaryProjector<dim, n_components, VectorType, VectorizedArrayType>::
  BoundaryProjector(
    const types::boundary_id interface_id,
    std::shared_ptr<const MatrixFree<dim, double, VectorizedArrayType>> data)
  : interface_id(interface_id)
  , data(data)
{
  // Compute relevant boundary indices
  boundary_indices =
    (DoFTools::extract_boundary_dofs(data->get_dof_handler(),
                                     ComponentMask(),
                                     std::set<types::boundary_id>{
                                       interface_id}) &
     data->get_dof_handler().locally_owned_dofs());

  // Initialize RHS vector
  data->initialize_dof_vector(rhs);
}



template <int dim,
          int n_components,
          typename VectorType,
          typename VectorizedArrayType>
void
BoundaryProjector<dim, n_components, VectorType, VectorizedArrayType>::project(
  VectorType &residual)
{
  // Setup the matrix
  MatrixType b_mass_matrix(interface_id);
  b_mass_matrix.initialize(data);
  b_mass_matrix.compute_diagonal();
  // Copy the relevant RHS entries into a zeroed vector
  rhs = 0;
  for (const auto &i : boundary_indices)
    rhs[i] = residual[i];

  // And solve the system
  ReductionControl               control(rhs.size(), 0., 1e-14, false, false);
  SolverCG<VectorType>           cg(control);
  PreconditionJacobi<MatrixType> preconditioner;
  preconditioner.initialize(b_mass_matrix, 0.8);
  cg.solve(b_mass_matrix, residual, rhs, preconditioner);
  residual.update_ghost_values();
}
