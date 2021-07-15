#pragma once

#include <deal.II/base/mg_level_object.h>

#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/vector_tools.h>

DEAL_II_NAMESPACE_OPEN

struct MultigridParameters
{
  struct
  {
    std::string  type            = "cg_with_amg";
    unsigned int maxiter         = 10000;
    double       abstol          = 1e-20;
    double       reltol          = 1e-4;
    unsigned int smoother_sweeps = 1;
    unsigned int n_cycles        = 1;
    std::string  smoother_type   = "Chebyshev";
  } coarse_solver;

  struct
  {
    std::string  type                = "chebyshev";
    double       smoothing_range     = 20;
    unsigned int degree              = 5;
    unsigned int eig_cg_n_iterations = 20;
  } smoother;

  struct
  {
    MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType
      p_sequence = MGTransferGlobalCoarseningTools::
        PolynomialCoarseningSequenceType::decrease_by_one;
    bool perform_h_transfer = true;
  } transfer;
};


template <typename VectorType,
          int dim,
          typename SystemMatrixType,
          typename LevelMatrixType,
          typename MGTransferType>
static void
mg_solve(SolverControl &                                        solver_control,
         VectorType &                                           dst,
         const VectorType &                                     src,
         const MultigridParameters &                            mg_data,
         const DoFHandler<dim> &                                dof,
         const SystemMatrixType &                               fine_matrix,
         const MGLevelObject<std::unique_ptr<LevelMatrixType>> &mg_matrices,
         const MGTransferType &                                 mg_transfer)
{
  AssertThrow(mg_data.coarse_solver.type == "cg_with_amg", ExcNotImplemented());
  AssertThrow(mg_data.smoother.type == "chebyshev", ExcNotImplemented());

  const unsigned int min_level = mg_matrices.min_level();
  const unsigned int max_level = mg_matrices.max_level();

  using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
  using SmootherType               = PreconditionChebyshev<LevelMatrixType,
                                             VectorType,
                                             SmootherPreconditionerType>;
  using PreconditionerType = PreconditionMG<dim, VectorType, MGTransferType>;

  // We initialize level operators and Chebyshev smoothers here.
  mg::Matrix<VectorType> mg_matrix(mg_matrices);

  MGLevelObject<typename SmootherType::AdditionalData> smoother_data(min_level,
                                                                     max_level);

  for (unsigned int level = min_level; level <= max_level; level++)
    {
      smoother_data[level].preconditioner =
        std::make_shared<SmootherPreconditionerType>();

      mg_matrices[level]->compute_diagonal();
      smoother_data[level].preconditioner =
        mg_matrices[level]->get_matrix_diagonal_inverse();

      smoother_data[level].smoothing_range = mg_data.smoother.smoothing_range;
      smoother_data[level].degree          = mg_data.smoother.degree;
      smoother_data[level].eig_cg_n_iterations =
        mg_data.smoother.eig_cg_n_iterations;
    }

  MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType> mg_smoother;
  mg_smoother.initialize(mg_matrices, smoother_data);

  // Next, we initialize the coarse-grid solver. We use conjugate-gradient
  // method with AMG as preconditioner.
  ReductionControl     coarse_grid_solver_control(mg_data.coarse_solver.maxiter,
                                              mg_data.coarse_solver.abstol,
                                              mg_data.coarse_solver.reltol,
                                              false,
                                              false);
  SolverCG<VectorType> coarse_grid_solver(coarse_grid_solver_control);

  std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;


  PETScWrappers::PreconditionBoomerAMG                 precondition_amg;
  PETScWrappers::PreconditionBoomerAMG::AdditionalData amg_data;

  amg_data.n_sweeps_coarse = mg_data.coarse_solver.smoother_sweeps;
  amg_data.max_iter        = mg_data.coarse_solver.n_cycles;

  amg_data.relaxation_type_down = PETScWrappers::PreconditionBoomerAMG::
    AdditionalData::RelaxationType::Chebyshev;
  amg_data.relaxation_type_up = PETScWrappers::PreconditionBoomerAMG::
    AdditionalData::RelaxationType::Chebyshev;
  amg_data.relaxation_type_coarse = PETScWrappers::PreconditionBoomerAMG::
    AdditionalData::RelaxationType::Chebyshev;

  precondition_amg.initialize(mg_matrices[min_level]->get_system_matrix(),
                              amg_data);


  mg_coarse =
    std::make_unique<MGCoarseGridIterativeSolver<VectorType,
                                                 SolverCG<VectorType>,
                                                 LevelMatrixType,
                                                 decltype(precondition_amg)>>(
      coarse_grid_solver, *mg_matrices[min_level], precondition_amg);

  // Finally, we create the Multigrid object, convert it to a
  preconditioner,
    // and use it inside of a conjugate-gradient solver to solve the linear
    // system of equations.
    Multigrid<VectorType> mg(
      mg_matrix, *mg_coarse, mg_transfer, mg_smoother, mg_smoother);

  PreconditionerType preconditioner(dof, mg, mg_transfer);

  SolverCG<VectorType>(solver_control)
    .solve(fine_matrix, dst, src, preconditioner);
}


// @sect4{Hybrid polynomial/geometric-global-coarsening multigrid
// preconditioner}

// The above function deals with the actual solution for a given sequence of
// multigrid objects. This functions creates the actual multigrid levels, in
// particular the operators, and the transfer operator as a
// MGTransferGlobalCoarsening object.
template <typename VectorType, typename OperatorType, int dim>
void
solve_with_gmg(SolverControl &            solver_control,
               const OperatorType &       system_matrix,
               VectorType &               dst,
               const VectorType &         src,
               const MultigridParameters &mg_data,
               const Mapping<dim> &       mapping,
               const DoFHandler<dim> &    dof_handler)
{
  // Create a DoFHandler and operator for each multigrid level,
  // as well as, create transfer operators. To be able to
  // set up the operators, we need a set of DoFHandler that we create
  // via global coarsening of p or h. For latter, we need also a sequence
  // of Triangulation objects that are obtained by
  // Triangulation::coarsen_global().
  //
  // In case no h-transfer is requested, we provide an empty deleter for the
  // `emplace_back()` function, since the Triangulation of our DoFHandler is
  // an external field and its destructor is called somewhere else.
  MGLevelObject<DoFHandler<dim>>                     dof_handlers;
  MGLevelObject<std::unique_ptr<OperatorType>>       operators;
  MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;

  std::vector<std::shared_ptr<const Triangulation<dim>>>
    coarse_grid_triangulations;
  if (mg_data.transfer.perform_h_transfer)
    coarse_grid_triangulations =
      MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
        dof_handler.get_triangulation());
  else
    coarse_grid_triangulations.emplace_back(const_cast<Triangulation<dim> *>(&(
                                              dof_handler.get_triangulation())),
                                            [](auto &) {});

  // Determine the total number of levels for the multigrid operation and
  // allocate sufficient memory for all levels.
  const unsigned int n_h_levels = coarse_grid_triangulations.size() - 1;

  const auto get_max_active_fe_degree = [&](const auto &dof_handler) {
    unsigned int max = 0;

    for (auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        max = std::max(max, dof_handler.get_fe(cell->active_fe_index()).degree);

    return Utilities::MPI::max(max, MPI_COMM_WORLD);
  };

  const unsigned int n_p_levels =
    MGTransferGlobalCoarseningTools::create_polynomial_coarsening_sequence(
      get_max_active_fe_degree(dof_handler), mg_data.transfer.p_sequence)
      .size();

  std::map<unsigned int, unsigned int> fe_index_for_degree;
  for (unsigned int i = 0; i < dof_handler.get_fe_collection().size(); ++i)
    {
      const unsigned int degree = dof_handler.get_fe(i).degree;
      Assert(fe_index_for_degree.find(degree) == fe_index_for_degree.end(),
             ExcMessage("FECollection does not contain unique degrees."));
      fe_index_for_degree[degree] = i;
    }

  unsigned int minlevel   = 0;
  unsigned int minlevel_p = n_h_levels;
  unsigned int maxlevel   = n_h_levels + n_p_levels - 1;

  dof_handlers.resize(minlevel, maxlevel);
  operators.resize(minlevel, maxlevel);
  transfers.resize(minlevel, maxlevel);

  // Loop from the minimum (coarsest) to the maximum (finest) level and set up
  // DoFHandler accordingly. We start with the h-levels, where we distribute
  // on increasingly finer meshes linear elements.
  for (unsigned int l = 0; l < n_h_levels; ++l)
    {
      dof_handlers[l].reinit(*coarse_grid_triangulations[l]);
      dof_handlers[l].distribute_dofs(dof_handler.get_fe_collection());
    }

  // After we reached the finest mesh, we will adjust the polynomial degrees
  // on each level. We reverse iterate over our data structure and start at
  // the finest mesh that contains all information about the active FE
  // indices. We then lower the polynomial degree of each cell level by level.
  for (unsigned int i = 0, l = maxlevel; i < n_p_levels; ++i, --l)
    {
      dof_handlers[l].reinit(dof_handler.get_triangulation());

      if (l == maxlevel) // finest level
        {
          auto &dof_handler_mg = dof_handlers[l];

          auto cell_other = dof_handler.begin_active();
          for (auto &cell : dof_handler_mg.active_cell_iterators())
            {
              if (cell->is_locally_owned())
                cell->set_active_fe_index(cell_other->active_fe_index());
              cell_other++;
            }
        }
      else // coarse level
        {
          auto &dof_handler_fine   = dof_handlers[l + 1];
          auto &dof_handler_coarse = dof_handlers[l + 0];

          auto cell_other = dof_handler_fine.begin_active();
          for (auto &cell : dof_handler_coarse.active_cell_iterators())
            {
              if (cell->is_locally_owned())
                {
                  const unsigned int next_degree =
                    MGTransferGlobalCoarseningTools::
                      create_next_polynomial_coarsening_degree(
                        cell_other->get_fe().degree,
                        mg_data.transfer.p_sequence);
                  Assert(fe_index_for_degree.find(next_degree) !=
                           fe_index_for_degree.end(),
                         ExcMessage("Next polynomial degree in sequence "
                                    "does not exist in FECollection."));

                  cell->set_active_fe_index(fe_index_for_degree[next_degree]);
                }
              cell_other++;
            }
        }

      dof_handlers[l].distribute_dofs(dof_handler.get_fe_collection());
    }

  // Next, we will create all data structures additionally needed on each
  // multigrid level. This involves determining constraints with homogeneous
  // Dirichlet boundary conditions, and building the operator just like on the
  // active level.
  MGLevelObject<AffineConstraints<typename VectorType::value_type>> constraints(
    minlevel, maxlevel);

  for (unsigned int level = minlevel; level <= maxlevel; ++level)
    {
      const auto &dof_handler = dof_handlers[level];
      auto &      constraint  = constraints[level];

      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);
      constraint.reinit(locally_relevant_dofs);

      const types::boundary_id constrained_id = 1;
      DoFTools::make_hanging_node_constraints(dof_handler, constraint);
      VectorTools::interpolate_boundary_values(mapping,
                                               dof_handler,
                                               constrained_id,
                                               Functions::ZeroFunction<dim>(),
                                               constraint);
      constraint.close();

      operators[level] = std::make_unique<OperatorType>();
    }

  // Set up intergrid operators and collect transfer operators within a single
  // operator as needed by the Multigrid solver class.
  for (unsigned int level = minlevel; level < minlevel_p; ++level)
    transfers[level + 1].reinit_geometric_transfer(dof_handlers[level + 1],
                                                   dof_handlers[level],
                                                   constraints[level + 1],
                                                   constraints[level]);

  for (unsigned int level = minlevel_p; level < maxlevel; ++level)
    transfers[level + 1].reinit_polynomial_transfer(dof_handlers[level + 1],
                                                    dof_handlers[level],
                                                    constraints[level + 1],
                                                    constraints[level]);

  MGTransferGlobalCoarsening<dim, VectorType> transfer(
    transfers,
    [&](const auto l, auto &vec) { operators[l]->initialize_dof_vector(vec); });

  // Finally, proceed to solve the problem with multigrid.
  mg_solve(solver_control,
           dst,
           src,
           mg_data,
           dof_handler,
           system_matrix,
           operators,
           transfer);
}

DEAL_II_NAMESPACE_CLOSE
