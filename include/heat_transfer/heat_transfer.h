#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <adapter/boundary_projector.h>
#include <adapter/precice_adapter.h>
#include <base/fe_integrator.h>
#include <base/time_handler.h>
#include <base/utilities.h>
#include <cases/case_selector.h>
#include <heat_transfer/cuda_laplace_operator.h>
#include <heat_transfer/laplace_operator.h>
#include <parameter/parameter_handling.h>

#include <fstream>
#include <iostream>


namespace Heat_Transfer
{
  using namespace dealii;

  /**
   *The coefficient is constant equal to one, at the moment. The class is kept
   *nevertheless in order to easily test more complex coefficients.
   */
  template <int dim>
  class Coefficient : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;

    template <typename number>
    number
    value(const Point<dim, number> &p, const unsigned int component = 0) const;
  };

  template <int dim>
  template <typename number>
  number
  Coefficient<dim>::value(const Point<dim, number> & /*p*/,
                          const unsigned int /*component*/) const
  {
    return 1.;
  }
  template <int dim>
  double
  Coefficient<dim>::value(const Point<dim> & p,
                          const unsigned int component) const
  {
    return value<double>(p, component);
  }

  // Forward declarations in case we don't have any CUDA support
#ifndef DEAL_II_COMPILER_CUDA_AWARE
  template <int dim, int fe_degree, typename number>
  class CUDALaplaceOperator;

  namespace CUDAWrappers
  {
    template <int dim, typename number>
    class MatrixFree;
  }
#endif

  template <int dim, bool use_cuda = false>
  class LaplaceProblem
  {
  public:
    using FECellIntegrator =
      typename LaplaceOperator<dim, double>::FECellIntegrator;
    using FEFaceIntegrator =
      typename LaplaceOperator<dim, double>::FEFaceIntegrator;

    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using LevelVectorType = LinearAlgebra::distributed::Vector<float>;

    // TODO: Find a solution for the Fe degree
    using CPUSystemMatrixType  = LaplaceOperator<dim, double>;
    using CUDASystemMatrixType = CUDALaplaceOperator<dim, 2, double>;

    using SystemMatrixType =
      std::conditional_t<use_cuda, CUDASystemMatrixType, CPUSystemMatrixType>;

    using DeviceVector =
      LinearAlgebra::distributed::Vector<double, ::dealii::MemorySpace::CUDA>;

    LaplaceProblem(const Parameters::HeatParameters<dim> &parameters);

    void
    run(std::shared_ptr<TestCases::TestCaseBase<dim>> testcase_);

  private:
    /// @brief make_grid Creates the grid and the boundary conditions of the problem
    void
    make_grid();

    /// @brief setup_system Initialize required data structures (MG and MF)
    void
    setup_system();

    /// @brief setup the gmg data structures
    void
    setup_gmg();

    /// @brief assemble_rhs Assemble the RHS vector in each time step.
    void
    assemble_rhs();

    /**
     * @brief apply_boundary_condition Fill the constraint object in order to apply
     *        Dirichlet boundary condition.
     *
     * @param initialize Boolean in order to set if we set up the constraint object
     *        for MatrixFree during initialization (see comment in the function
     *        for more detials).
     */
    void
    apply_boundary_condition(const bool initialize);


    /// @brief solve Solve the assembled system.
    void
    solve();

    /// @brief solve using no preconditioner and cuda, returns the number of CG iterations
    unsigned int
    cuda_solve();

    /// @brief solve using the usual CPU options, returns the number of CG iterations
    unsigned int
    cpu_solve();

    /// @brief solve using the gmg preconditioner (CPU), returns the number of CG iterations
    unsigned int
    solve_gmg_preconditioner();

    /**
     * @brief compute_error
     * @return error
     */
    double
    compute_error(
      const Function<dim> &                             function,
      const LinearAlgebra::distributed::Vector<double> &solution) const;
    /**
     * @brief output_results Output the generated results.
     *
     * @param result_number Number of the result for the name of the parameter file.
     */
    void
    output_results(const unsigned int result_number) const;

    void
    evaluate_boundary_flux();

    const Parameters::HeatParameters<dim> &parameters;

    parallel::distributed::Triangulation<dim> triangulation;

    // In order to hold the copy
    std::shared_ptr<TestCases::TestCaseBase<dim>> testcase;

    FE_Q<dim>       fe;
    const QGauss<1> quadrature_1d;
    DoFHandler<dim> dof_handler;

    MappingQ1<dim> mapping;

    AffineConstraints<double> constraints;

    SystemMatrixType system_matrix;
    // In order to operate with the non-homogenous Dirichlet BCs
    // TODO: use one operator with different constraint objects instead
    CPUSystemMatrixType inhomogeneous_operator;

    MGConstrainedDoFs mg_constrained_dofs;
    using LevelMatrixType = LaplaceOperator<dim, float>;
    MGLevelObject<LevelMatrixType> mg_matrices;

    VectorType solution;
    VectorType solution_old;
    VectorType solution_update;
    VectorType system_rhs;
    std::unique_ptr<
      Adapter::Adapter<dim, 1, VectorType, VectorizedArray<double>>>
      precice_adapter;

    ConditionalOStream  pcout;
    mutable TimerOutput timer;
    unsigned long int   total_n_cg_iterations;
    unsigned int        total_n_cg_solve;

    // Valid options are none, jacobi and gmg
    std::string preconditioner_type = "none";

    Time time;
  };



  template <int dim, bool use_cuda>
  LaplaceProblem<dim, use_cuda>::LaplaceProblem(
    const Parameters::HeatParameters<dim> &parameters)
    : parameters(parameters)
    , triangulation(MPI_COMM_WORLD,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<
                      dim>::construct_multigrid_hierarchy)
    , fe(parameters.poly_degree)
    , quadrature_1d(parameters.quad_order)
    , dof_handler(triangulation)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
    , total_n_cg_iterations(0)
    , total_n_cg_solve(0)
    , time(parameters.end_time, parameters.delta_t)

  {
    const int ierr = Utilities::create_directory(parameters.output_folder);
    (void)ierr;
    Assert(ierr == 0, ExcMessage("can't create: " + parameters.output_folder));
    Utilities::print_configuration(pcout);
  }



  template <int dim, bool use_cuda>
  void
  LaplaceProblem<dim, use_cuda>::make_grid()
  {
    Assert(testcase.get() != nullptr, ExcInternalError());
    testcase->make_coarse_grid_and_bcs(triangulation);
    triangulation.refine_global(parameters.n_global_refinement);
  }



  template <int dim, bool use_cuda>
  void
  LaplaceProblem<dim, use_cuda>::setup_system()
  {
    system_matrix.clear();
    dof_handler.distribute_dofs(fe);

    std::locale s = pcout.get_stream().getloc();
    pcout.get_stream().imbue(std::locale(""));
    pcout << "--     . dim       = " << dim << "\n"
          << "--     . fe_degree = " << fe.degree << "\n"
          << "--     . 1d_quad   = " << quadrature_1d.size() << "\n"
          << "--     . Number of active cells: "
          << triangulation.n_global_active_cells() << "\n"
          << "--     . Number of degrees of freedom: " << dof_handler.n_dofs()
          << "\n"
          << std::endl;
    pcout.get_stream().imbue(s);

    // Fill the constraint object with the specified dirichlet boundary
    // conditions. Initially, set up the constraint with Zero values on the
    // interface
    apply_boundary_condition(true);
    {
      using MatrixFreeStorage =
        std::conditional_t<use_cuda,
                           CUDAWrappers::MatrixFree<dim, double>,
                           MatrixFree<dim, double>>;

      // Set up two matrix-free objects: one for the homogenous part and one for
      // the inhomogenous part (without any constraints)
      typename MatrixFree<dim, double>::AdditionalData additional_data_cpu;
      additional_data_cpu.tasks_parallel_scheme =
        MatrixFree<dim, double>::AdditionalData::none;
      additional_data_cpu.mapping_update_flags =
        (update_values | update_gradients | update_JxW_values |
         update_quadrature_points);

      // FIXME: The boundary face flag is only for the RHS assembly required in
      // order to apply the coupling data. Here, only the second MatrixFree
      // object is required. In order to ensure compatibility of the global
      // vectors, we initialize both MatrixFree objects with the same
      // AdditionalData
      // For Dirichlet: write gradients (maybe normal gradients)
      // For Neumann: write values and integrate them during assembly
      // In both cases: define interface using FEEvaluation::quadrature_point()
      if (testcase->is_dirichlet)
        additional_data_cpu.mapping_update_flags_boundary_faces =
          (update_values | update_JxW_values | update_gradients |
           update_normal_vectors | update_quadrature_points);
      else
        additional_data_cpu.mapping_update_flags_boundary_faces =
          (update_values | update_JxW_values | update_quadrature_points);


      typename MatrixFreeStorage::AdditionalData additional_data;

      additional_data.mapping_update_flags =
        additional_data_cpu.mapping_update_flags;
      std::shared_ptr<MatrixFreeStorage> system_mf_storage(
        new MatrixFreeStorage);

      system_mf_storage->reinit(
        mapping, dof_handler, constraints, quadrature_1d, additional_data);

      system_matrix.initialize(system_mf_storage);
      system_matrix.evaluate_coefficient();
      system_matrix.set_delta_t(time.get_delta_t());

      // ... the second matrix-free operator for inhomogenous BCs
      AffineConstraints<double> no_constraints;
      no_constraints.close();
      std::shared_ptr<MatrixFree<dim, double>> matrix_free(
        new MatrixFree<dim, double>());
      matrix_free->reinit(mapping,
                          dof_handler,
                          no_constraints,
                          quadrature_1d,
                          additional_data_cpu);

      inhomogeneous_operator.initialize(matrix_free);
      inhomogeneous_operator.evaluate_coefficient();
      inhomogeneous_operator.set_delta_t(time.get_delta_t());
    }

    if constexpr (use_cuda)
      inhomogeneous_operator.initialize_dof_vector(solution);
    else
      system_matrix.initialize_dof_vector(solution);

    solution_old.reinit(solution);
    solution_update.reinit(solution);
    system_rhs.reinit(solution);

    if (preconditioner_type == "gmg")
      setup_gmg();
  }



  template <int dim, bool use_cuda>
  void
  LaplaceProblem<dim, use_cuda>::setup_gmg()
  {
    mg_matrices.clear_elements();
    dof_handler.distribute_mg_dofs();

    const unsigned int nlevels = triangulation.n_global_levels();
    mg_matrices.resize(0, nlevels - 1);
    mg_constrained_dofs.initialize(dof_handler);

    // Set constraints on Dirichlet boundaries
    for (const auto &el : testcase->dirichlet)
      {
        const auto &mask = testcase->dirichlet_mask.find(el.first);
        Assert(mask != testcase->dirichlet_mask.end(),
               ExcMessage("Could not find component mask for ID " +
                          std::to_string(el.first)));

        mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                           {el.first},
                                                           mask->second);
        // Add the Dirichlet boundary condition in case we read Dirichlet
        // boundary values
        if (testcase->is_dirichlet)
          mg_constrained_dofs.make_zero_boundary_constraints(
            dof_handler,
            {int(TestCases::TestCaseBase<dim>::interface_id)},
            ComponentMask());
      }

    for (unsigned int level = 0; level < nlevels; ++level)
      {
        IndexSet relevant_dofs;
        DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                      level,
                                                      relevant_dofs);
        AffineConstraints<double> level_constraints;
        level_constraints.reinit(relevant_dofs);
        level_constraints.add_lines(
          mg_constrained_dofs.get_boundary_indices(level));
        level_constraints.close();

        typename MatrixFree<dim, float>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme =
          MatrixFree<dim, float>::AdditionalData::none;
        additional_data.mapping_update_flags =
          (update_gradients | update_JxW_values | update_quadrature_points);
        additional_data.mg_level = level;
        std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(
          new MatrixFree<dim, float>());
        mg_mf_storage_level->reinit(mapping,
                                    dof_handler,
                                    level_constraints,
                                    quadrature_1d,
                                    additional_data);

        mg_matrices[level].initialize(mg_mf_storage_level,
                                      mg_constrained_dofs,
                                      level);
        mg_matrices[level].evaluate_coefficient();
        mg_matrices[level].set_delta_t(time.get_delta_t());
      }
  }



  template <int dim, bool use_cuda>
  void
  LaplaceProblem<dim, use_cuda>::assemble_rhs()
  {
    TimerOutput::Scope t(timer, "assemble rhs");

    apply_boundary_condition(false);

    //... and fill in the constraints into the solution vector (non-homogenous
    // part)
    solution = 0;
    constraints.distribute(solution);

    // Compute effect of the non-homogenous boundary condition on the system_rhs
    system_rhs = 0;
    inhomogeneous_operator.vmult(system_rhs, solution);
    system_rhs *= -1.0;

    const auto data = inhomogeneous_operator.get_matrix_free();
    // Do not reset system_rhs here as it holds the non-homogenous boundary
    // condition
    FECellIntegrator phi(*data);
    Assert(testcase->heat_transfer_rhs.get() != nullptr, ExcNotInitialized());

    solution_old.update_ghost_values();
    const auto dt = make_vectorized_array<double>(time.get_delta_t());
    for (unsigned int cell = 0; cell < data->n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(solution_old, EvaluationFlags::values);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_value(phi.get_value(q) +
                             (dt * evaluate_function<dim, double>(
                                     *(testcase->heat_transfer_rhs),
                                     phi.quadrature_point(q))),
                           q);
        phi.integrate_scatter(EvaluationFlags::values, system_rhs);
      }

    FEFaceIntegrator phi_face(*data);
    Assert(
      phi_face.fast_evaluation_supported(fe.degree, quadrature_1d.size()),
      ExcMessage(
        "The given combination of the polynomial degree and quadrature order is not supported by default."));
    unsigned int q_index = 0;

    if (!testcase->is_dirichlet)
      for (unsigned int face = data->n_inner_face_batches();
           face <
           data->n_boundary_face_batches() + data->n_inner_face_batches();
           ++face)
        {
          const auto boundary_id = data->get_boundary_id(face);

          // Only interfaces
          if (boundary_id != int(TestCases::TestCaseBase<dim>::interface_id))
            continue;

          // Read out the total displacment
          phi_face.reinit(face);

          // Number of active faces
          const auto active_faces = data->n_active_entries_per_face_batch(face);

          for (unsigned int q = 0; q < phi_face.n_q_points; ++q)
            {
              // Get the value from preCICE
              const auto heat_flux =
                precice_adapter->read_on_quadrature_point(q_index,
                                                          active_faces);
              phi_face.submit_value(-heat_flux * dt, q);
              ++q_index;
            }
          // Integrate the result and write into the rhs vector
          phi_face.integrate_scatter(EvaluationFlags::values, system_rhs);
        }
    system_rhs.compress(VectorOperation::add);
  }



  template <int dim, bool use_cuda>
  void
  LaplaceProblem<dim, use_cuda>::apply_boundary_condition(const bool initialize)
  {
    // Update the constraints object
    constraints.clear();
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    constraints.reinit(locally_relevant_dofs);

    // First, fill the constraint object with the constraints at the interface
    if (testcase->is_dirichlet)
      {
        Assert((precice_adapter.get() != nullptr) || initialize,
               ExcNotInitialized());
        // That's a bit odd: we need to fill the constraint object at the
        // coupling interface initially in order to tell MatrixFree during the
        // initialization that these constraints are actually constrained.
        // However, the adapter object is not yet initialized, since we need an
        // initialized MatrixFree object in order to set up the Adapter. As a
        // remedy, we fill the interface Dirichlet constraints initially with a
        // dummy zero boundary condition, which is only relevant for the
        // initialization of MatrixFree. Afterwards, we use the adapter as
        // usual.
        initialize ? VectorTools::interpolate_boundary_values(
                       mapping,
                       dof_handler,
                       int(TestCases::TestCaseBase<dim>::interface_id),
                       Functions::ZeroFunction<dim>(),
                       constraints) :
                     precice_adapter->apply_dirichlet_bcs(constraints);
      }

    for (const auto &el : testcase->dirichlet)
      {
        const auto &mask = testcase->dirichlet_mask.find(el.first);
        Assert(mask != testcase->dirichlet_mask.end(),
               ExcMessage("Could not find component mask for ID " +
                          std::to_string(el.first)));

        // Assumption: number of constrained dofs stays the sam throughout the
        // simulation. Otherwise the MG constrained dofs would need to be
        // initialized here as well
        Function<dim> *func = el.second.get();
        // Set the time in the analytic solution
        func->set_time(time.current());
        VectorTools::interpolate_boundary_values(
          mapping, dof_handler, el.first, *func, constraints, mask->second);
      }
    constraints.close();
  }



  template <int dim, bool use_cuda>
  unsigned int
  LaplaceProblem<dim, use_cuda>::cpu_solve()
  {
    unsigned int n_iterations = 0;

    if (preconditioner_type == "jacobi")
      {
        // FIXME: Leads currently to more iterations compared to "none"
        SolverControl                        solver_control(system_rhs.size(),
                                     1e-12 * system_rhs.l2_norm());
        SolverCG<VectorType>                 cg(solver_control);
        PreconditionJacobi<SystemMatrixType> preconditioner;
        double                               relaxation = .7;
        preconditioner.initialize(system_matrix, relaxation);
        system_matrix.compute_diagonal();
        cg.solve(system_matrix, solution_update, system_rhs, preconditioner);
        n_iterations += solver_control.last_step();
      }
    else if (preconditioner_type == "none")
      {
        constraints.set_zero(system_rhs);
        SolverControl        solver_control(system_rhs.size(),
                                     1e-12 * system_rhs.l2_norm());
        SolverCG<VectorType> cg(solver_control);
        cg.solve(system_matrix,
                 solution_update,
                 system_rhs,
                 PreconditionIdentity());
        n_iterations += solver_control.last_step();
      }
    else if (preconditioner_type == "gmg")
      {
        n_iterations += solve_gmg_preconditioner();
      }
    else
      AssertThrow(false, ExcNotImplemented());

    return n_iterations;
  }


  // The code here is not compiled in case we don't have GPU support due to the
  // constexpr below
  template <int dim, bool use_cuda>
  unsigned int
  LaplaceProblem<dim, use_cuda>::cuda_solve()
  {
    unsigned int n_iterations = 0;
    if (preconditioner_type == "none")
      {
        // Required here as the constraints are not ignored
        constraints.set_zero(system_rhs);
        SolverControl solver_control(system_rhs.size(),
                                     1e-12 * system_rhs.l2_norm());

        // Create vectos on the device
        DeviceVector update, rhs;
        system_matrix.initialize_dof_vector(update);
        system_matrix.initialize_dof_vector(rhs);
        update = 0;
        TimerOutput::Scope t1(timer, "memory transfer CPU to GPU");
        // transfer the vectors to the device
        LinearAlgebra::ReadWriteVector<double> rw_vector(
          dof_handler.locally_owned_dofs());
        rw_vector.import(system_rhs, VectorOperation::insert);
        rhs.import(rw_vector, VectorOperation::insert);
        t1.stop();
        // Solve
        SolverCG<DeviceVector> cg(solver_control);
        cg.solve(system_matrix, update, rhs, PreconditionIdentity());

        TimerOutput::Scope t2(timer, "memory transfer GPU to CPU");
        // transfer the solution back, using VectorOperation::add to the
        // solution could be faster/save the addition later on
        rw_vector.import(update, VectorOperation::insert);
        solution_update.import(rw_vector, VectorOperation::insert);
        t2.stop();
        n_iterations = solver_control.last_step();
      }
    else
      AssertThrow(false, ExcNotImplemented());

    return n_iterations;
  }



  template <int dim, bool use_cuda>
  void
  LaplaceProblem<dim, use_cuda>::solve()
  {
    TimerOutput::Scope t(timer, "solve system");

    if constexpr (use_cuda)
      total_n_cg_iterations += cuda_solve();
    else
      total_n_cg_iterations += cpu_solve();

    // More or less a safety operation, as the constraints have already been
    // enforced
    constraints.set_zero(solution_update);
    // and update the complete solution = non-homogenous part + homogenous part
    solution += solution_update;
    solution.update_ghost_values();
    ++total_n_cg_solve;
  }


  template <int dim, bool use_cuda>
  unsigned int
  LaplaceProblem<dim, use_cuda>::solve_gmg_preconditioner()
  {
    SolverControl solver_control(100, 1e-12 * system_rhs.l2_norm());

    MGTransferMatrixFree<dim, float> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    using SmootherType =
      PreconditionChebyshev<LevelMatrixType, LevelVectorType>;
    mg::SmootherRelaxation<SmootherType, LevelVectorType> mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData>  smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels();
         ++level)
      {
        if (level > 0)
          {
            smoother_data[level].smoothing_range     = 15.;
            smoother_data[level].degree              = 5;
            smoother_data[level].eig_cg_n_iterations = 10;
          }
        else
          {
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree          = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
          }
        mg_matrices[level].compute_diagonal();
        smoother_data[level].preconditioner =
          mg_matrices[level].get_matrix_diagonal_inverse();
      }
    mg_smoother.initialize(mg_matrices, smoother_data);

    MGCoarseGridApplySmoother<LevelVectorType> mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<LevelVectorType> mg_matrix(mg_matrices);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>
      mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels();
         ++level)
      mg_interface_matrices[level].initialize(mg_matrices[level]);
    mg::Matrix<LevelVectorType> mg_interface(mg_interface_matrices);

    Multigrid<LevelVectorType> mg(
      mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dim, LevelVectorType, MGTransferMatrixFree<dim, float>>
      preconditioner(dof_handler, mg, mg_transfer);

    SolverCG<VectorType> cg(solver_control);
    cg.solve(system_matrix, solution_update, system_rhs, preconditioner);

    return solver_control.last_step();
  }


  template <int dim, bool use_cuda>
  double
  LaplaceProblem<dim, use_cuda>::compute_error(
    const Function<dim> &                             function,
    const LinearAlgebra::distributed::Vector<double> &solution) const
  {
    TimerOutput::Scope t(timer, "compute errors");
    double             errors_squared = 0;
    const auto         data = inhomogeneous_operator.get_matrix_free();
    FECellIntegrator   phi(*data);

    for (unsigned int cell = 0; cell < data->n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(solution, EvaluationFlags::values);
        VectorizedArray<double> local_errors_squared = 0;
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto error =
              evaluate_function(function, phi.quadrature_point(q)) -
              phi.get_value(q);
            const auto JxW = phi.JxW(q);

            local_errors_squared += error * error * JxW;
          }
        for (unsigned int v = 0;
             v < data->n_active_entries_per_cell_batch(cell);
             ++v)
          errors_squared += local_errors_squared[v];
      }
    const double global_error =
      Utilities::MPI::sum(errors_squared, MPI_COMM_WORLD);
    return std::sqrt(global_error);
  }



  template <int dim, bool use_cuda>
  void
  LaplaceProblem<dim, use_cuda>::output_results(
    const unsigned int result_number) const
  {
    TimerOutput::Scope t(timer, "output");

    DataOut<dim> data_out;

    DataOutBase::VtkFlags flags;
#if DEAL_II_VERSION_GTE(9, 5, 0)
    flags.compression_level = DataOutBase::CompressionLevel::best_speed;
#else
    flags.compression_level = DataOutBase::VtkFlags::best_speed;
#endif
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    solution.update_ghost_values();
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");

    Vector<double> mpi_owner(triangulation.n_active_cells());
    mpi_owner =
      Utilities::MPI::this_mpi_process(triangulation.get_communicator());
    data_out.add_data_vector(mpi_owner, "owner");

    data_out.build_patches(mapping,
                           fe.degree,
                           DataOut<dim>::curved_inner_cells);

    const std::string filename = parameters.output_folder + "solution_" +
                                 Utilities::int_to_string(result_number, 3) +
                                 ".vtu";

    data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);

    pcout << "Output @ " << time.current() << "s written to " << filename
          << std::endl;
    // Compute error
    testcase->initial_condition->set_time(time.current());
    const auto error = compute_error(*(testcase->initial_condition), solution);
    pcout << "Error: " << error << "\n" << std::endl;
  }



  template <int dim, bool use_cuda>
  void
  LaplaceProblem<dim, use_cuda>::evaluate_boundary_flux()
  {
    VectorType &residual = system_rhs;
    // 1. Compute the residual
    // Contribution of the linear operator on the current solution, ignoring
    // constraints. Note that the sign is switched compared to the RHS assembly
    inhomogeneous_operator.vmult(residual, solution);

    Assert(testcase->heat_transfer_rhs.get() != nullptr, ExcNotInitialized());
    const auto       data = inhomogeneous_operator.get_matrix_free();
    FECellIntegrator phi(*data);
    const auto       dt = make_vectorized_array<double>(time.get_delta_t());

    solution_old.update_ghost_values();
    // Now loop over all cells
    for (unsigned int cell = 0; cell < data->n_cell_batches(); ++cell)
      {
        // Read in the solution of the old timestep as this is the relevant
        // data, again ignoring constraints here as they are inhomogeneous
        phi.reinit(cell);
        phi.gather_evaluate(solution_old, EvaluationFlags::values);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            // Same assembly as in the RHS assembly, again with signs switched
            phi.submit_value(-(phi.get_value(q) +
                               (dt * evaluate_function<dim, double>(
                                       *(testcase->heat_transfer_rhs),
                                       phi.quadrature_point(q)))),
                             q);
          }
        phi.integrate_scatter(EvaluationFlags::values, residual);
      }
    residual.compress(VectorOperation::add);
    residual /= time.get_delta_t();

    BoundaryProjector<dim> projector(
      int(TestCases::TestCaseBase<dim>::interface_id),
      inhomogeneous_operator.get_matrix_free());

    projector.project(residual);
  }


  template <int dim, bool use_cuda>
  void
  LaplaceProblem<dim, use_cuda>::run(
    std::shared_ptr<TestCases::TestCaseBase<dim>> testcase_)
  {
    testcase = testcase_;
    make_grid();

    setup_system();
    Assert(testcase->initial_condition.get() != nullptr, ExcNotInitialized());
    testcase->initial_condition->set_time(0.0);
    VectorTools::interpolate(dof_handler,
                             *(testcase->initial_condition),
                             solution);
    solution_old = solution;
    output_results(0);
    time.increment();

    precice_adapter = std::make_unique<
      Adapter::Adapter<dim, 1, VectorType, VectorizedArray<double>>>(
      parameters,
      int(TestCases::TestCaseBase<dim>::interface_id),
      inhomogeneous_operator.get_matrix_free(),
      0 /*MF dof index*/,
      0 /*MF quad index*/,
      testcase->is_dirichlet);
    // TODO: The if block here is ugly and it is actually repeated furhter down,
    // replace it
    {
      TimerOutput::Scope t(timer, "initialize preCICE");

      if (testcase->is_dirichlet)
        {
          // We misuse the system_rhs in the flux evaluation
          // evaluate_boundary_flux();
          precice_adapter->initialize(solution);
        }
      else
        precice_adapter->initialize(solution);
    }
    while (precice_adapter->is_coupling_ongoing())
      {
        precice_adapter->save_current_state_if_required([&]() {});

        assemble_rhs();
        solve();
        {
          TimerOutput::Scope t(timer, "advance preCICE");
          if (testcase->is_dirichlet)
            {
              // We misuse the system_rhs in the flux evaluation
              // evaluate_boundary_flux();
              precice_adapter->advance(solution, time.get_delta_t());
            }
          else
            precice_adapter->advance(solution, time.get_delta_t());
        }
        precice_adapter->reload_old_state_if_required(
          [&]() { solution = solution_old; });

        if (precice_adapter->is_time_window_complete())
          {
            if (static_cast<int>(Utilities::round_to_precision(
                  time.current() / parameters.output_tick, 12)) !=
                  static_cast<int>(Utilities::round_to_precision(
                    (time.current() - time.get_delta_t()) /
                      parameters.output_tick,
                    12)) ||
                time.current() >= time.end() - 1e-12)
              output_results(static_cast<unsigned int>(
                std::round(time.current() / parameters.output_tick)));

            // Increment and update
            solution_old = solution;
            time.increment();
          }
      }
    pcout << std::endl
          << "Average CG iter = " << (total_n_cg_iterations / total_n_cg_solve)
          << std::endl
          << "Total CG iter = " << total_n_cg_iterations << std::endl
          << "Total CG solve = " << total_n_cg_solve << std::endl;
    timer.print_wall_time_statistics(MPI_COMM_WORLD);
    pcout << std::endl;
  }
} // namespace Heat_Transfer
