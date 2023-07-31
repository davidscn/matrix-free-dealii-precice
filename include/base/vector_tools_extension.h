#pragma once

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

DEAL_II_NAMESPACE_OPEN

namespace VectorTools
{
  /**
   * Similar to project_boundary_values, but instead of taking a function
   * describing the boundary, a global vector is passed into this function.
   * Furthermore, the projected boundary is expected to not contain any
   * constraints, since we want to evaluate the numerical flux across the
   * coupling boundary and do not want to pass the values into the actual
   * system. However, the function works only for serial Triangulations. For a
   * parallel version, have a look at the BoundaryProjector class.
   *
   * The rhs_vector is assumed to contain the nodal function values. The result
   * of this operation is again filled into the rhs_vector.
   */
  template <int dim, int spacedim, typename VectorType>
  void
  project_boundary_values(const Mapping<dim, spacedim>    &mapping,
                          const DoFHandler<dim, spacedim> &dof_handler,
                          const types::boundary_id         boundary_id,
                          VectorType                      &rhs_vector,
                          const Quadrature<dim - 1>       &quadrature)
  {
    Assert((dynamic_cast<const parallel::TriangulationBase<dim> *>(
              &(dof_handler.get_triangulation())) == nullptr),
           ExcNotImplemented());
    // Do the actual projection step from the global vector to the boundary
    // A dummy Function  object and the respective boundary_function variable
    // required for the MatrixCreator::create_boundary_mass_matrix function
    const Functions::ZeroFunction<dim>                  func(1);
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    boundary_functions[boundary_id] = &func;

    // a vector of size global dofs. However, all irrelevant DoFs (not located
    // at the interface) are set to 'invalid_dof_index' and all other dofs are
    // enumerated consecutively so that we obtain a mapping from global dofs,
    // i.e., the iterator running through the std::vector, and the dof numbering
    // in the local sub-problem dof_to_boundary_mapping[i]. Have a look at
    // map_dof_to_boundary_indices for more information
    std::vector<types::global_dof_index> dof_to_boundary_mapping;
    // a set keeping the interface boundary_id
    std::set<types::boundary_id> selected_boundary_components;
    for (typename std::map<types::boundary_id,
                           const Function<dim> *>::const_iterator i =
           boundary_functions.begin();
         i != boundary_functions.end();
         ++i)
      selected_boundary_components.insert(i->first);

    DoFTools::map_dof_to_boundary_indices(dof_handler,
                                          selected_boundary_components,
                                          dof_to_boundary_mapping);

    // Set up the sparsity pattern
    DynamicSparsityPattern dsp(dof_handler.n_boundary_dofs(boundary_functions),
                               dof_handler.n_boundary_dofs(boundary_functions));
    DoFTools::make_boundary_sparsity_pattern(dof_handler,
                                             boundary_functions,
                                             dof_to_boundary_mapping,
                                             dsp);
    SparsityPattern sparsity;
    sparsity.copy_from(dsp);
    sparsity.compress();

    // make mass matrix and right hand side. The RHS is just a dummy RHS and
    // replaced down the line
    SparseMatrix<double> mass_matrix(sparsity);
    Vector<double>       rhs(sparsity.n_rows());

    MatrixCreator::create_boundary_mass_matrix(
      mapping,
      dof_handler,
      quadrature,
      mass_matrix,
      boundary_functions,
      rhs,
      dof_to_boundary_mapping,
      static_cast<const Function<dim, double> *>(nullptr));

    // Fill in the RHS with the residual values
    for (unsigned int i = 0; i < dof_to_boundary_mapping.size(); ++i)
      if (dof_to_boundary_mapping[i] != numbers::invalid_dof_index)
        rhs(dof_to_boundary_mapping[i]) = rhs_vector(i);

    // Solve the actual system using a CG solver and a SSOR preconditioner
    Vector<double>   boundary_projection(rhs.size());
    ReductionControl control(5 * rhs.size(), 0., 1e-12, false, false);
    GrowingVectorMemory<Vector<double>>    memory;
    SolverCG<Vector<double>>               cg(control, memory);
    PreconditionSSOR<SparseMatrix<double>> prec;
    prec.initialize(mass_matrix, 1.2);
    cg.solve(mass_matrix, boundary_projection, rhs, prec);

    // Scatter the computed boundary projection back into the global flux vector
    for (unsigned int i = 0; i < dof_to_boundary_mapping.size(); ++i)
      if (dof_to_boundary_mapping[i] != numbers::invalid_dof_index)
        rhs_vector(i) = boundary_projection(dof_to_boundary_mapping[i]);
  }
} // namespace VectorTools

DEAL_II_NAMESPACE_CLOSE
