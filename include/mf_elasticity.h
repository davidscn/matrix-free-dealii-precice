#pragma once

/**
 * level of output to deallog, can be:
 * 0 - none
 * 1 - norms of vectors
 * 2 - CG iterations
 * 3 - GMG iterations
 */
static const unsigned int debug_level = 0;

// We start by including all the necessary deal.II header files and some C++
// related ones. They have been discussed in detail in previous tutorial
// programs, so you need only refer to past tutorials for details.
#include <deal.II/base/config.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <deal.II/physics/transformations.h>

#include <cases/case_base.h>
#include <material.h>
#include <mf_nh_operator.h>
#include <parameter/parameter_handling.h>
#include <precice_adapter.h>
#include <q_equidistant.h>
#include <sys/stat.h>
#include <version.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// We then stick everything that relates to this tutorial program into a
// namespace of its own, and import all the deal.II function and class names
// into it:
namespace FSI
{
  using namespace dealii;

  // @sect3{Time class}

  // A simple class to store time data. Its functioning is transparent so no
  // discussion is necessary. For simplicity we assume a constant time step
  // size.
  class Time
  {
  public:
    Time(const double time_end, const double delta_t)
      : timestep(0)
      , time_current(0.0)
      , time_end(time_end)
      , delta_t(delta_t)
    {}

    virtual ~Time() = default;

    double
    current() const
    {
      return time_current;
    }
    double
    end() const
    {
      return time_end;
    }
    double
    get_delta_t() const
    {
      return delta_t;
    }
    unsigned int
    get_timestep() const
    {
      return timestep;
    }
    void
    increment()
    {
      time_current += delta_t;
      ++timestep;
    }

  private:
    unsigned int timestep;
    double       time_current;
    const double time_end;
    const double delta_t;
  };

  // The Solid class is the central class.
  template <int dim, int degree, int n_q_points_1d, typename Number>
  class Solid
  {
  public:
    using LevelNumber         = float;
    using LevelVectorType     = LinearAlgebra::distributed::Vector<LevelNumber>;
    using VectorType          = LinearAlgebra::distributed::Vector<Number>;
    using VectorizedArrayType = VectorizedArray<Number>;
    using LevelVectorizedArrayType = VectorizedArray<LevelNumber>;

    Solid(const Parameters::AllParameters<dim> &parameters);

    virtual ~Solid();

    void
    run(std::shared_ptr<TestCases::TestCaseBase<dim>> testcase_);

  private:
    // We start the collection of member functions with one that builds the
    // grid:
    void
    make_grid();

    // Set up the finite element system to be solved:
    void
    system_setup();

    // Set up the matrix-free objects
    void
    setup_matrix_free();

    // Update matrix-free within a nonlinear iteration
    void
    update_matrix_free(const int &it_nr);

    // Function to assemble the right hand side vector.
    void
    assemble_residual(const int it_nr);

    // Apply Dirichlet boundary conditions on the displacement field
    void
    make_constraints(const int &it_nr, const bool print = true);

    bool
    check_convergence(const unsigned int newton_iteration);

    // Solve for the displacement using a Newton-Raphson method. We break this
    // function into the nonlinear loop and the function that solves the
    // linearized Newton-Raphson step:
    void
    solve_nonlinear_timestep();

    // solve linear system and return a tuple consisting of number of
    // iterations, residual and condition number estiamte.
    std::tuple<unsigned int, double, double>
    solve_linear_system(VectorType &newton_update) const;

    void
    output_results(const unsigned int result_number) const;

    // Set up an Additional data object
    template <typename AdditionalData>
    void
    setup_mf_additional_data(AdditionalData &   data,
                             const unsigned int level,
                             const bool         initialize_indices) const;

    void
    reinit_mg_transfer();

    void
    setup_mg_interpolation(const unsigned int max_level,
                           const bool         reset_mg_transfer);

    template <typename AdditionalData>
    void
    reinit_matrix_free(const AdditionalData &data,
                       const bool            reinit_mf_current,
                       const bool            update_mapping_current,
                       const bool            reinit_mf_reference);

    template <typename AdditionalData>
    void
    reinit_multi_grid_matrix_free(const AdditionalData &data,
                                  const bool            reinit_mf_current,
                                  const bool            update_mapping_current,
                                  const bool            reinit_mf_reference,
                                  const unsigned int    level);

    void
    adjust_ghost_range(const unsigned int level);

    template <typename Operator>
    void
    setup_operator_cache(Operator &nh_operator, const unsigned int level);

    void
    setup_gmg(const bool initialize_all);

    void
    update_acceleration(VectorType &displacement_delta);

    void
    update_velocity(VectorType &displacement_delta);

    // MPI communicator
    MPI_Comm mpi_communicator;

    // Terminal output on root MPI process
    ConditionalOStream pcout;

    // Finally, some member variables that describe the current state: A
    // collection of the parameters used to describe the problem setup...
    const Parameters::AllParameters<dim> &parameters;

    // ...the volume of the reference and current configurations...
    double vol_reference;
    double vol_current;

    // ...and description of the geometry on which the problem is solved:
    parallel::distributed::Triangulation<dim> triangulation;

    // In order to hold the copy
    std::shared_ptr<TestCases::TestCaseBase<dim>> testcase;

    // Also, keep track of the current time and the time spent evaluating
    // certain functions
    Time                time;
    std::ofstream       timer_output_file;
    ConditionalOStream  timer_out;
    mutable TimerOutput timer;
    std::ofstream       deallogfile;
    std::ofstream       blessed_output_file;
    ConditionalOStream  bcout;

    // A description of the finite-element system including the displacement
    // polynomial degree, the degree-of-freedom handler, number of DoFs per
    // cell and the extractor objects used to retrieve information from the
    // solution vectors:
    const FESystem<dim>              fe;
    DoFHandler<dim>                  dof_handler;
    const unsigned int               dofs_per_cell;
    const FEValuesExtractors::Vector u_fe;

    IndexSet locally_owned_dofs, locally_relevant_dofs;

    static constexpr double beta  = 0.25;
    static constexpr double gamma = 0.5;

    const double alpha_1 = 1. / (beta * std::pow(parameters.delta_t, 2));
    const double alpha_2 = 1. / (beta * parameters.delta_t);
    const double alpha_3 = (1 - (2 * beta)) / (2 * beta);
    const double alpha_4 = gamma / (beta * parameters.delta_t);
    const double alpha_5 = 1 - (gamma / beta);
    const double alpha_6 = (1 - (gamma / (2 * beta))) * parameters.delta_t;

    // matrix material
    std::shared_ptr<Material_Compressible_Neo_Hook_One_Field<dim, Number>>
      material;

    std::shared_ptr<
      Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArrayType>>
      material_vec;
    std::shared_ptr<
      Material_Compressible_Neo_Hook_One_Field<dim, LevelVectorizedArrayType>>
      material_level;

    // inclusion material
    std::shared_ptr<Material_Compressible_Neo_Hook_One_Field<dim, Number>>
      material_inclusion;

    std::shared_ptr<
      Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArrayType>>
      material_inclusion_vec;

    std::shared_ptr<
      Material_Compressible_Neo_Hook_One_Field<dim, LevelVectorizedArrayType>>
      material_inclusion_level;

    std::unique_ptr<
      Adapter::Adapter<dim, degree, VectorType, VectorizedArrayType>>
      precice_adapter;

    static const unsigned int n_components      = dim;
    static const unsigned int first_u_component = 0;

    enum
    {
      u_dof = 0
    };

    // Rules for Gauss-quadrature on both the cell and faces. The number of
    // quadrature points on both cells and faces is recorded.
    const QGauss<dim>     qf_cell;
    const QGauss<dim - 1> qf_face;
    const unsigned int    n_q_points;
    const unsigned int    n_q_points_f;

    // Objects that store the converged solution and right-hand side vectors,
    // as well as the tangent matrix. There is a AffineConstraints object used
    // to keep track of constraints.
    AffineConstraints<double> constraints;

    // the residual
    VectorType system_rhs;

    // solution at the previous time-step
    VectorType old_displacement;

    // current value of increment solution
    VectorType delta_displacement;

    // current total solution:  solution_total = solution_n + solution_delta
    VectorType total_displacement;

    VectorType newton_update;

    VectorType acceleration;
    VectorType velocity;
    VectorType acceleration_old;
    VectorType velocity_old;

    MGLevelObject<LevelVectorType> mg_total_displacement;


    // Then define a number of variables to store norms and update norms and
    // normalisation factors.
    struct Errors
    {
      Errors()
        : e(1.0)
      {}

      void
      reset()
      {
        e = 1.0;
      }
      void
      normalise(const Errors &rhs)
      {
        if (rhs.e != 0.0)
          e /= rhs.e;
      }

      double e;
    };

    mutable Errors error_residual, error_residual_0, error_residual_norm,
      error_update, error_update_0, error_update_norm;

    // Print information to screen in a pleasing way...
    void
    print_conv_header();

    void
    print_conv_footer();

    void
    print_solution_watchpoint();

    std::shared_ptr<MappingQEulerian<dim, VectorType>> eulerian_mapping;
    std::shared_ptr<MatrixFree<dim, double>>           mf_data_current;
    std::shared_ptr<MatrixFree<dim, double>>           mf_data_reference;


    std::vector<std::shared_ptr<MappingQEulerian<dim, LevelVectorType>>>
                                                         mg_eulerian_mapping;
    std::vector<std::shared_ptr<MatrixFree<dim, float>>> mg_mf_data_current;
    std::vector<std::shared_ptr<MatrixFree<dim, float>>> mg_mf_data_reference;

    NeoHookOperator<dim, degree, n_q_points_1d, double> mf_nh_operator;

    using LevelMatrixType = NeoHookOperator<dim, degree, n_q_points_1d, float>;

    MGLevelObject<LevelMatrixType> mg_mf_nh_operator;

    std::shared_ptr<MGTransferMatrixFree<dim, float>> mg_transfer;

    using SmootherChebyshev =
      PreconditionChebyshev<LevelMatrixType, LevelVectorType>;

    // MGSmootherPrecondition<LevelMatrixType, SmootherChebyshev,
    // LevelVectorType> mg_smoother_chebyshev;
    mg::SmootherRelaxation<SmootherChebyshev, LevelVectorType>
      mg_smoother_chebyshev;

    MGCoarseGridApplySmoother<LevelVectorType> mg_coarse_chebyshev;

    std::shared_ptr<SolverControl>             coarse_solver_control;
    std::shared_ptr<SolverCG<LevelVectorType>> coarse_solver;

    MGCoarseGridIterativeSolver<LevelVectorType,
                                SolverCG<LevelVectorType>,
                                LevelMatrixType,
                                SmootherChebyshev>
      mg_coarse_iterative;

    mg::Matrix<LevelVectorType> mg_operator_wrapper;

    std::shared_ptr<Multigrid<LevelVectorType>> multigrid;

    std::shared_ptr<
      PreconditionMG<dim, LevelVectorType, MGTransferMatrixFree<dim, float>>>
      multigrid_preconditioner;

    MGConstrainedDoFs mg_constrained_dofs;

    bool print_mf_memory;

    unsigned long int total_n_cg_iterations;
    unsigned int      total_n_cg_solve;
  };

  // @sect3{Implementation of the <code>Solid</code> class}

  // @sect4{Public interface}

  int
  create_directory(std::string pathname, const mode_t mode)
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

  // We initialise the Solid class using data extracted from the parameter file.
  template <int dim, int degree, int n_q_points_1d, typename Number>
  Solid<dim, degree, n_q_points_1d, Number>::Solid(
    const Parameters::AllParameters<dim> &parameters)
    : mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    , parameters(parameters)
    , vol_reference(0.0)
    , vol_current(0.0)
    , triangulation(
        mpi_communicator,
        typename Triangulation<dim>::MeshSmoothing(
          // guarantee that the mesh also does not change by more than
          // refinement level across vertices that might connect two cells:
          Triangulation<dim>::limit_level_difference_at_vertices),
        typename parallel::distributed::Triangulation<dim>::Settings(
          // needed for GMG:
          parallel::distributed::Triangulation<
            dim>::construct_multigrid_hierarchy))
    , time(parameters.end_time, parameters.delta_t)
    , timer_out(timer_output_file,
                Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    , timer(mpi_communicator,
            timer_out,
            TimerOutput::summary,
            TimerOutput::wall_times)
    , bcout(blessed_output_file,
            Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    ,
    // The Finite Element System is composed of dim continuous displacement
    // DOFs.
    fe(FE_Q<dim>(degree), dim)
    , // displacement
    dof_handler(triangulation)
    , dofs_per_cell(fe.dofs_per_cell)
    , u_fe(first_u_component)
    , material(
        std::make_shared<Material_Compressible_Neo_Hook_One_Field<dim, Number>>(
          parameters.mu,
          parameters.nu,
          parameters.rho,
          alpha_1,
          parameters.material_formulation))
    , material_vec(
        std::make_shared<
          Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArrayType>>(
          parameters.mu,
          parameters.nu,
          parameters.rho,
          alpha_1,
          parameters.material_formulation))
    , material_level(
        std::make_shared<
          Material_Compressible_Neo_Hook_One_Field<dim,
                                                   LevelVectorizedArrayType>>(
          parameters.mu,
          parameters.nu,
          parameters.rho,
          alpha_1,
          parameters.material_formulation))
    , material_inclusion(
        std::make_shared<Material_Compressible_Neo_Hook_One_Field<dim, Number>>(
          parameters.mu * 100.,
          parameters.nu,
          parameters.rho,
          alpha_1,
          parameters.material_formulation))
    , material_inclusion_vec(
        std::make_shared<
          Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArrayType>>(
          parameters.mu * 100.,
          parameters.nu,
          parameters.rho,
          alpha_1,
          parameters.material_formulation))
    , material_inclusion_level(
        std::make_shared<
          Material_Compressible_Neo_Hook_One_Field<dim,
                                                   LevelVectorizedArrayType>>(
          parameters.mu * 100.,
          parameters.nu,
          parameters.rho,
          alpha_1,
          parameters.material_formulation))
    , qf_cell(n_q_points_1d)
    , qf_face(n_q_points_1d)
    , n_q_points(qf_cell.size())
    , n_q_points_f(qf_face.size())
    , print_mf_memory(true)
    , total_n_cg_iterations(0)
    , total_n_cg_solve(0)
  {
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        const int ierr =
          create_directory(parameters.output_folder,
                           S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
        (void)ierr;
        Assert(ierr == 0,
               ExcMessage("can't create: " + parameters.output_folder));

        deallogfile.open(parameters.output_folder + "deallog.txt");
        deallog.attach(deallogfile);

        timer_output_file.open(parameters.output_folder + "timings.txt");
        blessed_output_file.open(parameters.output_folder + "output");
      }

    mf_nh_operator.set_material(material_vec, material_inclusion_vec);

    // print some data about how we run:
    auto print = [&](ConditionalOStream &stream) {
      const int n_tasks =
        dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
      const int          n_threads      = dealii::MultithreadInfo::n_threads();
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

      if (n_vect_bits == 64)
        stream << "disabled";
      else if (n_vect_bits == 128)
        stream << "SSE2";
      else if (n_vect_bits == 256)
        stream << "AVX";
      else if (n_vect_bits == 512)
        stream << "AVX512";
      else
        AssertThrow(false, ExcNotImplemented());

      stream << ")" << std::endl;
      stream << "--     . version " << GIT_TAG << " (revision " << GIT_SHORTREV
             << " on branch " << GIT_BRANCH << ")" << std::endl;
      stream << "--     . deal.II " << DEAL_II_PACKAGE_VERSION << " (revision "
             << DEAL_II_GIT_SHORTREV << " on branch " << DEAL_II_GIT_BRANCH
             << ")" << std::endl;
      stream
        << "-----------------------------------------------------------------------------"
        << std::endl
        << std::endl;
    };
    print(timer_out);
    print(pcout);
  }

  // The class destructor simply clears the data held by the DOFHandler
  template <int dim, int degree, int n_q_points_1d, typename Number>
  Solid<dim, degree, n_q_points_1d, Number>::~Solid()
  {
    mf_nh_operator.clear();

    mf_data_current.reset();
    mf_data_reference.reset();
    eulerian_mapping.reset();

    dof_handler.clear();

    multigrid_preconditioner.reset();
    multigrid.reset();
    mg_coarse_chebyshev.clear();
    mg_smoother_chebyshev.clear();
    mg_operator_wrapper.reset();
    mg_mf_nh_operator.clear_elements();
    mg_transfer.reset();
  }


  // We start the function with preprocessing, and then output the initial grid
  // before starting the simulation proper with the first time (and loading)
  // increment.
  //
  template <int dim, int degree, int n_q_points_1d, typename Number>
  void
  Solid<dim, degree, n_q_points_1d, Number>::run(
    std::shared_ptr<TestCases::TestCaseBase<dim>> testcase_)
  {
    testcase = testcase_;
    make_grid();
    system_setup();
    output_results(0);
    time.increment();

    // We then declare the incremental solution update $\varDelta
    // \mathbf{\Xi}:= \{\varDelta \mathbf{u}\}$ and start the loop over the
    // time domain.
    //
    setup_matrix_free();
    precice_adapter = std::make_unique<
      Adapter::Adapter<dim, degree, VectorType, VectorizedArrayType>>(
      parameters,
      int(TestCases::TestCaseBase<dim>::interface_id),
      false,
      mf_data_reference);
    precice_adapter->initialize(total_displacement);

    // At the beginning, we reset the solution update for this time step...
    while (precice_adapter->is_coupling_ongoing())
      {
        // preCICE complains otherwise
        precice_adapter->save_current_state_if_required([&]() {});

        delta_displacement = 0.0;
        solve_nonlinear_timestep();

        precice_adapter->advance(total_displacement, time.get_delta_t());

        precice_adapter->reload_old_state_if_required([&]() {
          acceleration       = acceleration_old;
          total_displacement = old_displacement;
        });

        // ...and plot the results before moving on happily to the next time
        // step:
        if (precice_adapter->is_time_window_complete())
          {
            // TODO: Work with vectors without ghost entris
            old_displacement = total_displacement;
            // Acceleration update is performed within the Newton loop
            update_velocity(delta_displacement);

            if (static_cast<int>(time.current() / parameters.output_tick) !=
                  static_cast<int>((time.current() - time.get_delta_t()) /
                                   parameters.output_tick) ||
                time.current() >= time.end() - 1e-12)
              output_results(static_cast<unsigned int>(
                std::round(time.current() / parameters.output_tick)));
            print_solution_watchpoint();
            time.increment();
          }
      }

    // for post-processing, print average CG iterations over the whole run:
    timer_out << std::endl
              << "Average CG iter = "
              << (total_n_cg_iterations / total_n_cg_solve) << std::endl
              << "Total CG iter = " << total_n_cg_iterations << std::endl
              << "Total CG solve = " << total_n_cg_solve << std::endl;
    // for post-processing, print average CG iterations over the whole run:
    bcout << std::endl
          << "Average CG iter = " << (total_n_cg_iterations / total_n_cg_solve)
          << std::endl;
  }



  template <int dim, int degree, int n_q_points_1d, typename Number>
  void
  Solid<dim, degree, n_q_points_1d, Number>::make_grid()
  {
    Assert(testcase.get() != nullptr, ExcInternalError());
    testcase->make_coarse_grid_and_bcs(triangulation);
    triangulation.refine_global(parameters.n_global_refinement);

    vol_reference = GridTools::volume(triangulation);
    vol_current   = vol_reference;
    pcout << "--     . Reference volume: " << vol_reference << std::endl;
    bcout << "--     . Reference volume: " << vol_reference << std::endl;
  }


  // @sect4{Solid::system_setup}

  // Next we describe how the FE system is setup.  We first determine the number
  // of components per block. Since the displacement is a vector component, the
  // first dim components belong to it.
  template <int dim, int degree, int n_q_points_1d, typename Number>
  void
  Solid<dim, degree, n_q_points_1d, Number>::system_setup()
  {
    timer.enter_subsection("Setup system");

    // The DOF handler is then initialised and we renumber the grid in an
    // efficient manner. We also record the number of DOFs per block.
    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();
    DoFRenumbering::Cuthill_McKee(dof_handler);

    auto print = [&](ConditionalOStream &stream) {
      stream << "--     . dim       = " << dim << "\n"
             << "--     . fe_degree = " << degree << "\n"
             << "--     . 1d_quad   = " << n_q_points_1d << "\n"
             << "--     . Number of active cells: "
             << triangulation.n_global_active_cells() << "\n"
             << "--     . Number of degrees of freedom: "
             << dof_handler.n_dofs() << "\n"
             << std::endl;
    };
    std::locale s = pcout.get_stream().getloc();
    pcout.get_stream().imbue(std::locale(""));
    print(pcout);
    pcout.get_stream().imbue(s);
    print(bcout);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs.clear();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);


    // We then set up storage vectors
    system_rhs.reinit(locally_owned_dofs,
                      locally_relevant_dofs,
                      mpi_communicator);
    // TODO: Switch to vectors without ghosts
    old_displacement.reinit(system_rhs);
    delta_displacement.reinit(system_rhs);
    total_displacement.reinit(system_rhs);
    newton_update.reinit(system_rhs);
    acceleration.reinit(system_rhs);
    velocity.reinit(system_rhs);
    acceleration_old.reinit(system_rhs);
    velocity_old.reinit(system_rhs);

    // switch to ghost mode:
    delta_displacement.update_ghost_values();
    total_displacement.update_ghost_values();
    acceleration.update_ghost_values();

    timer.leave_subsection();

    // print some info
    timer_out << "dim   = " << dim << std::endl
              << "p     = " << degree << std::endl
              << "q     = " << n_q_points_1d << std::endl
              << "cells = " << triangulation.n_global_active_cells()
              << std::endl
              << "dofs  = " << dof_handler.n_dofs() << std::endl
              << std::endl;
  }



  template <int dim, int degree, int n_q_points_1d, typename Number>
  void
  Solid<dim, degree, n_q_points_1d, Number>::setup_matrix_free()
  {
    make_constraints(0, false);
    // Setup MF dditional data
    typename MatrixFree<dim, double>::AdditionalData data;
    setup_mf_additional_data(data,
                             numbers::invalid_unsigned_int,
                             true /*indices*/);

    mf_data_current   = std::make_shared<MatrixFree<dim, double>>();
    mf_data_reference = std::make_shared<MatrixFree<dim, double>>();

    eulerian_mapping =
      std::make_shared<MappingQEulerian<dim, VectorType>>(degree,
                                                          dof_handler,
                                                          total_displacement);
    reinit_matrix_free(data,
                       true /*current*/,
                       false /*mapping*/,
                       true /*reference*/);

    adjust_ghost_range(numbers::invalid_unsigned_int);
    setup_operator_cache(mf_nh_operator, numbers::invalid_unsigned_int);

    // The gmg part
    if (parameters.preconditioner_type != "gmg")
      return;

    const unsigned int max_level = triangulation.n_global_levels() - 1;
    // Setup the MG transfer matrices
    reinit_mg_transfer();

    // Setup mg interpolation
    setup_mg_interpolation(max_level, true);

    // Setup MG additional data
    std::vector<typename MatrixFree<dim, float>::AdditionalData>
      mg_additional_data(max_level + 1);

    mg_eulerian_mapping.clear();
    mg_eulerian_mapping.resize(0);

    mg_mf_data_current.resize(triangulation.n_global_levels());
    mg_mf_data_reference.resize(triangulation.n_global_levels());
    mg_mf_nh_operator.resize(0, max_level);

    for (unsigned int level = 0; level <= max_level; ++level)
      {
        mg_mf_nh_operator[level].set_material(material_level,
                                              material_inclusion_level);
        // The MF objects
        mg_mf_data_current[level] = std::make_shared<MatrixFree<dim, float>>();
        mg_mf_data_reference[level] =
          std::make_shared<MatrixFree<dim, float>>();

        // Additional data
        setup_mf_additional_data(mg_additional_data[level], level, true);

        // Eulerian mapping
        std::shared_ptr<MappingQEulerian<dim, LevelVectorType>> euler_level =
          std::make_shared<MappingQEulerian<dim, LevelVectorType>>(
            degree, dof_handler, mg_total_displacement[level], level);
        mg_eulerian_mapping.push_back(euler_level);

        // Reinit
        reinit_multi_grid_matrix_free(mg_additional_data[level],
                                      true /*current*/,
                                      false /*mapping*/,
                                      true /*reference*/,
                                      level);

        adjust_ghost_range(level);
        setup_operator_cache(mg_mf_nh_operator[level], level);
      }
    setup_gmg(true);
  }



  template <int dim, int degree, int n_q_points_1d, typename Number>
  template <typename AdditionalData>
  void
  Solid<dim, degree, n_q_points_1d, Number>::setup_mf_additional_data(
    AdditionalData &   data,
    const unsigned int level,
    const bool         initialize_indices) const
  {
    timer.enter_subsection("Setup: AdditionalData");

    // The constraints in Newton-Raphson are different for it_nr=0 and 1,
    // and then they are the same so we only need to re-init the data
    // according to the updated displacement/mapping

    data.tasks_parallel_scheme = AdditionalData::none;
    data.mapping_update_flags =
      update_values | update_gradients | update_JxW_values;
    data.initialize_indices = initialize_indices;
    // make sure materials with different ID end up in different SIMD blocks:
    data.cell_vectorization_categories_strict = true;

    // Setup MF additional data
    if (level == numbers::invalid_unsigned_int)
      {
        // TODO: Sets up the structure for inner face batches as well, maybe
        // another option way is possible. Also, this option is only required
        // for the referential MatrixFree, but consistency between referential
        // and Eulerian MatrixFree is required
        data.mapping_update_flags_boundary_faces =
          update_values | update_gradients | update_quadrature_points |
          update_JxW_values;
        data.cell_vectorization_category.resize(triangulation.n_active_cells());
        for (const auto &cell : triangulation.active_cell_iterators())
          if (cell->is_locally_owned())
            data.cell_vectorization_category[cell->active_cell_index()] =
              cell->material_id();
      }
    // Setup MG additinal data
    else
      {
        data.cell_vectorization_category.resize(triangulation.n_cells(level));
        for (const auto &cell : triangulation.cell_iterators_on_level(level))
          if (cell->is_locally_owned_on_level())
            data.cell_vectorization_category[cell->index()] =
              cell->material_id();

        data.mg_level = level;
      }
    timer.leave_subsection();
  }



  template <int dim, int degree, int n_q_points_1d, typename Number>
  void
  Solid<dim, degree, n_q_points_1d, Number>::reinit_mg_transfer()
  {
    timer.enter_subsection("Setup MF: MGTransferMatrixFree");

    // and clean up transfer which is also initialized with mg_matrices:
    mg_transfer.reset();

    mg_transfer = std::make_shared<MGTransferMatrixFree<dim, LevelNumber>>(
      mg_constrained_dofs);
    mg_transfer->build(dof_handler);

    timer.leave_subsection();
  }



  template <int dim, int degree, int n_q_points_1d, typename Number>
  void
  Solid<dim, degree, n_q_points_1d, Number>::setup_mg_interpolation(
    const unsigned int max_level,
    const bool         reset_mg_transfer)
  {
    timer.enter_subsection("Setup MF: interpolate_to_mg");

    if (reset_mg_transfer)
      mg_total_displacement.resize(0, max_level);

    // transfer displacement to MG levels:
    LevelVectorType solution_total_transfer;
    solution_total_transfer.reinit(total_displacement);
    solution_total_transfer = total_displacement;
    mg_transfer->interpolate_to_mg(dof_handler,
                                   mg_total_displacement,
                                   solution_total_transfer);

    timer.leave_subsection();
  }



  template <int dim, int degree, int n_q_points_1d, typename Number>
  template <typename AdditionalData>
  void
  Solid<dim, degree, n_q_points_1d, Number>::reinit_matrix_free(
    const AdditionalData &data,
    const bool            reinit_mf_current,
    const bool            update_mapping_current,
    const bool            reinit_mf_reference)
  {
    if (!reinit_mf_current && !reinit_mf_reference && !update_mapping_current)
      return;

    timer.enter_subsection("Setup MF: Reinit matrix-free");

    // TODO: Parametrize with function below
    const QGauss<1> quad(n_q_points_1d);

    // solution_total is the point around which we linearize
    if (reinit_mf_current)
      mf_data_current->reinit(
        *eulerian_mapping, dof_handler, constraints, quad, data);

    if (update_mapping_current)
      mf_data_current->update_mapping(*eulerian_mapping);

    // TODO: Parametrize mapping
    if (reinit_mf_reference)
      {
        const std::vector<const AffineConstraints<double> *> constr = {
          &constraints};
        const std::vector<Quadrature<1>> quadratures = {
          quad, QEquidistant<1>(n_q_points_1d)};
        mf_data_reference->reinit(StaticMappingQ1<dim>::mapping,
                                  {&dof_handler},
                                  constr,
                                  quadratures,
                                  data);

        // Only reinitialized in case the reference MF has changed, since all
        // data structures are reinitialized according to the reference object
        mf_nh_operator.initialize(mf_data_current,
                                  mf_data_reference,
                                  total_displacement,
                                  parameters.mf_caching);
      }

    // print memory consumption by MF
    if (print_mf_memory)
      {
        timer_out << "MF cache memory = "
                  << dealii::Utilities::MPI::sum(
                       mf_nh_operator.memory_consumption() / 1000000,
                       mpi_communicator)
                  << " Mb" << std::endl;
        print_mf_memory = false;
      }
    timer.leave_subsection();
  }



  template <int dim, int degree, int n_q_points_1d, typename Number>
  template <typename AdditionalData>
  void
  Solid<dim, degree, n_q_points_1d, Number>::reinit_multi_grid_matrix_free(
    const AdditionalData &data,
    const bool            reinit_mf_current,
    const bool            update_mf_current_mapping,
    const bool            reinit_mf_reference,
    const unsigned int    level)
  {
    // Assumption: only zero boundary conditions are used. Otheriwse, we
    // (probably) have to rebuild the level constraints
    if (!reinit_mf_current && !reinit_mf_reference &&
        !update_mf_current_mapping)
      return;

    timer.enter_subsection("Setup MF: Reinit multi-grid MF");

    const QGauss<1> quad(n_q_points_1d);

    AffineConstraints<double> level_constraints;
    IndexSet                  relevant_dofs;
    DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                  level,
                                                  relevant_dofs);

    // GMG MF operators do not support edge indices yet
    AssertThrow(
      mg_constrained_dofs.get_refinement_edge_indices(level).is_empty(),
      ExcNotImplemented());

    level_constraints.reinit(relevant_dofs);
    level_constraints.add_lines(
      mg_constrained_dofs.get_boundary_indices(level));
    level_constraints.close();

    if (reinit_mf_current)
      mg_mf_data_current[level]->reinit(*mg_eulerian_mapping[level],
                                        dof_handler,
                                        level_constraints,
                                        quad,
                                        data);
    if (update_mf_current_mapping)
      mg_mf_data_current[level]->update_mapping(*mg_eulerian_mapping[level]);

    if (reinit_mf_reference)
      {
        // TODO: Parametrize mapping
        mg_mf_data_reference[level]->reinit(StaticMappingQ1<dim>::mapping,
                                            dof_handler,
                                            level_constraints,
                                            quad,
                                            data);
        mg_mf_nh_operator[level].initialize(mg_mf_data_current[level],
                                            mg_mf_data_reference[level],
                                            mg_total_displacement[level],
                                            parameters.mf_caching);
      }

    timer.leave_subsection();
  }



  template <int dim, int degree, int n_q_points_1d, typename Number>
  void
  Solid<dim, degree, n_q_points_1d, Number>::adjust_ghost_range(
    const unsigned int level)
  {
    timer.enter_subsection("Setup MF: ghost range");

    // adjust ghost range if needed
    if (level == numbers::invalid_unsigned_int)
      {
        const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner =
          mf_data_current->get_vector_partitioner();

        Assert(partitioner->is_globally_compatible(
                 *mf_data_reference->get_vector_partitioner().get()),
               ExcInternalError());

        // TODO: Use initialize_dof_vector
        adjust_ghost_range_if_necessary(partitioner, newton_update);
        adjust_ghost_range_if_necessary(partitioner, system_rhs);
        adjust_ghost_range_if_necessary(partitioner, total_displacement);
        adjust_ghost_range_if_necessary(partitioner, acceleration);
        total_displacement.update_ghost_values();
        acceleration.update_ghost_values();
      }
    else
      // FIXME: interpolate_to_mg will resize MG vector, make sure it has the
      // right partition for MF
      {
        const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner =
          mg_mf_data_current[level]->get_vector_partitioner();

        Assert(partitioner->is_globally_compatible(
                 *mg_mf_data_reference[level]->get_vector_partitioner().get()),
               ExcInternalError());

        adjust_ghost_range_if_necessary(partitioner,
                                        mg_total_displacement[level]);
        mg_total_displacement[level].update_ghost_values();
      }

    timer.leave_subsection();
  }



  template <int dim, int degree, int n_q_points_1d, typename Number>
  template <typename Operator>
  void
  Solid<dim, degree, n_q_points_1d, Number>::setup_operator_cache(
    Operator &         nh_operator,
    const unsigned int level)
  {
    timer.enter_subsection("Setup MF: cache() and diagonal()");

    // need to cache prior to diagonal computations:
    nh_operator.cache();
    nh_operator.compute_diagonal();

    if (debug_level > 0)
      {
        deallog
          << "Number of constrained DoFs "
          << Utilities::MPI::sum(mf_data_current->get_constrained_dofs().size(),
                                 mpi_communicator)
          << std::endl;
        const std::string info = "on level " + std::to_string(level) + " :   ";
        deallog
          << "Diagonal "
          << (level == numbers::invalid_unsigned_int ? " :              " :
                                                       info)
          << nh_operator.get_matrix_diagonal_inverse()->get_vector().l2_norm()
          << std::endl;
        deallog << "Solution total (on level): "
                << (level == numbers::invalid_unsigned_int ?
                      total_displacement.l2_norm() :
                      mg_total_displacement[level].l2_norm())
                << std::endl;
      }
    timer.leave_subsection();
  }



  template <int dim, int degree, int n_q_points_1d, typename Number>
  void
  Solid<dim, degree, n_q_points_1d, Number>::setup_gmg(
    const bool initialize_all)
  {
    timer.enter_subsection("Setup MF: GMG setup");

    // setup GMG preconditioner
    const bool cheb_coarse = parameters.mf_coarse_chebyshev;
    {
      MGLevelObject<typename SmootherChebyshev::AdditionalData> smoother_data;
      smoother_data.resize(0, triangulation.n_global_levels() - 1);
      for (unsigned int level = 0; level < triangulation.n_global_levels();
           ++level)
        {
          if (cheb_coarse && level == 0)
            {
              smoother_data[level].smoothing_range =
                1e-3; // reduce residual by this relative tolerance
              smoother_data[level].degree =
                numbers::invalid_unsigned_int; // use as a solver
              smoother_data[level].eig_cg_n_iterations =
                (parameters.mf_coarse_chebyshev_accurate_eigenval ?
                   mg_mf_nh_operator[level].m() :
                   parameters.mf_chebyshev_n_cg_iterations);
            }
          else
            {
              // [1.2 \lambda_{max}/range, 1.2 \lambda_{max}]
              smoother_data[level].smoothing_range = 20;
              // With degree one, the Jacobi method with optimal damping
              // parameter is retrieved
              smoother_data[level].degree = 5;
              // number of CG iterataions to estimate the largest eigenvalue:
              smoother_data[level].eig_cg_n_iterations =
                parameters.mf_chebyshev_n_cg_iterations;
            }
          smoother_data[level].preconditioner =
            mg_mf_nh_operator[level].get_matrix_diagonal_inverse();
        }
      mg_smoother_chebyshev.initialize(mg_mf_nh_operator, smoother_data);
      mg_coarse_chebyshev.initialize(mg_smoother_chebyshev);
    }

    coarse_solver_control = std::make_shared<ReductionControl>(
      std::max(mg_mf_nh_operator[0].m(), static_cast<unsigned int>(100)),
      1e-10,
      1e-3,
      false,
      false);

    coarse_solver.reset();
    coarse_solver =
      std::make_shared<SolverCG<LevelVectorType>>(*coarse_solver_control);

    // Set all matrices to zero
    mg_coarse_iterative.clear();
    mg_coarse_iterative.initialize(*coarse_solver,
                                   mg_mf_nh_operator[0],
                                   mg_smoother_chebyshev[0]);

    if (initialize_all)
      { // wrap our level and interface matrices in an object having the
        // required
        // multiplication functions.
        mg_operator_wrapper.reset();
        mg_operator_wrapper.initialize(mg_mf_nh_operator);

        multigrid.reset();

        if (cheb_coarse)
          multigrid =
            std::make_shared<Multigrid<LevelVectorType>>(mg_operator_wrapper,
                                                         mg_coarse_chebyshev,
                                                         *mg_transfer,
                                                         mg_smoother_chebyshev,
                                                         mg_smoother_chebyshev,
                                                         /*min_level*/ 0);
        else
          multigrid =
            std::make_shared<Multigrid<LevelVectorType>>(mg_operator_wrapper,
                                                         mg_coarse_iterative,
                                                         *mg_transfer,
                                                         mg_smoother_chebyshev,
                                                         mg_smoother_chebyshev,
                                                         /*min_level*/ 0);


        multigrid->connect_coarse_solve(
          [&](const bool start, const unsigned int level) {
            if (start)
              timer.enter_subsection("Coarse solve level " +
                                     Utilities::int_to_string(level));
            else
              timer.leave_subsection();
          });

        multigrid->connect_restriction(
          [&](const bool start, const unsigned int level) {
            if (start)
              timer.enter_subsection("Coarse solve level " +
                                     Utilities::int_to_string(level));
            else
              timer.leave_subsection();
          });
        multigrid->connect_prolongation(
          [&](const bool start, const unsigned int level) {
            if (start)
              timer.enter_subsection("Prolongation level " +
                                     Utilities::int_to_string(level));
            else
              timer.leave_subsection();
          });
        multigrid->connect_pre_smoother_step(
          [&](const bool start, const unsigned int level) {
            if (start)
              timer.enter_subsection("Pre-smoothing level " +
                                     Utilities::int_to_string(level));
            else
              timer.leave_subsection();
          });

        multigrid->connect_post_smoother_step(
          [&](const bool start, const unsigned int level) {
            if (start)
              timer.enter_subsection("Post-smoothing level " +
                                     Utilities::int_to_string(level));
            else
              timer.leave_subsection();
          });

        // Reset mg preconditioner
        multigrid_preconditioner.reset();
        multigrid_preconditioner =
          std::make_shared<PreconditionMG<dim,
                                          LevelVectorType,
                                          MGTransferMatrixFree<dim, float>>>(
            dof_handler, *multigrid, *mg_transfer);
      }
    timer.leave_subsection();
  }



  // Note: Dirichlet boundary conditions are in structural mechanics usually
  // zero. Therefore, the constraints are not differently between Newton
  // iterations. We keep updating the constraints object anyway, i.e., the
  // assumption is not exploited in make_constraints. However, the MF and GMG
  // constraints are not updated accordingly and in case one wants to utilize
  // other contraints, the MF GMG needs to be reinitialized here.
  template <int dim, int degree, int n_q_points_1d, typename Number>
  void
  Solid<dim, degree, n_q_points_1d, Number>::update_matrix_free(const int &)
  {
    // We need to update the mapping in case we use a current dof handler, which
    // depends on selected caching strategy
    const bool update_mf_mapping =
      !(parameters.mf_caching == "scalar_referential" ||
        parameters.mf_caching == "tensor4_ns");
    // Currently hard-coded since the boundary conditions are static and we
    // don't need to reinit. Needs to be adjusted for time-dependent boundary
    // conditions or AMR.
    const bool reinit_mf = false;
    // For MF additional data
    const bool initialize_indices = false;

    // The non-gmg MF part
    // Setup MF dditional data
    // Use invalid unsigned int for the 'usual' non gmg MF level
    typename MatrixFree<dim, double>::AdditionalData data;
    if (reinit_mf)
      setup_mf_additional_data(data,
                               numbers::invalid_unsigned_int,
                               initialize_indices);
    // Recompute Eulerian mapping if necessary
    reinit_matrix_free(data,
                       reinit_mf /*current*/,
                       update_mf_mapping /*mapping*/,
                       reinit_mf /*reference*/);

    adjust_ghost_range(numbers::invalid_unsigned_int);
    setup_operator_cache(mf_nh_operator, numbers::invalid_unsigned_int);

    // The gmg part
    if (parameters.preconditioner_type != "gmg")
      return;

    const unsigned int max_level = triangulation.n_global_levels() - 1;

    // Setup mg interpolation
    setup_mg_interpolation(max_level, false);

    // Setup MG additional data
    std::vector<typename MatrixFree<dim, float>::AdditionalData>
      mg_additional_data(max_level + 1);

    for (unsigned int level = 0; level <= max_level; ++level)
      {
        // Additional data
        if (reinit_mf)
          setup_mf_additional_data(mg_additional_data[level],
                                   level,
                                   initialize_indices);

        // Reinit
        reinit_multi_grid_matrix_free(mg_additional_data[level],
                                      reinit_mf /*current*/,
                                      update_mf_mapping /*mapping*/,
                                      reinit_mf /*reference*/,
                                      level);
        adjust_ghost_range(level);
        setup_operator_cache(mg_mf_nh_operator[level], level);
      }
    setup_gmg(false);
  }



  template <int dim, int degree, int n_q_points_1d, typename Number>
  bool
  Solid<dim, degree, n_q_points_1d, Number>::check_convergence(
    const unsigned int newton_iteration)
  {
    if (newton_iteration == 0)
      error_residual_0 = error_residual;

    // We can now determine the normalised residual error and check for
    // solution convergence:
    error_residual_norm = error_residual;
    error_residual_norm.normalise(error_residual_0);

    bool converged = false;

    // do at least 3 NR iterations before converging,
    if (newton_iteration > 2)
      {
        // first check abosolute tolerance
        if (error_residual.e <= parameters.tol_f_abs ||
            error_update.e <= parameters.tol_u_abs)
          converged = true;


        if (error_residual_norm.e <= parameters.tol_f &&
            error_update_norm.e <= parameters.tol_u)
          converged = true;

        if (converged)
          {
            pcout << " CONVERGED! " << std::endl;
            print_conv_footer();

            bcout << "Converged in " << newton_iteration << " Newton iterations"
                  << std::endl;
          }
      }

    return converged;
  }

  // The next function is the driver method for the Newton-Raphson scheme. At
  // its top we create a new vector to store the current Newton update step,
  // reset the error storage objects and print solver header.
  template <int dim, int degree, int n_q_points_1d, typename Number>
  void
  Solid<dim, degree, n_q_points_1d, Number>::solve_nonlinear_timestep()
  {
    pcout << std::endl
          << "Timestep " << time.get_timestep() << " @ " << time.current()
          << "s" << std::endl;

    bcout << std::endl
          << "Timestep " << time.get_timestep() << " @ " << time.current()
          << "s" << std::endl;

    error_residual.reset();
    error_residual_0.reset();
    error_residual_norm.reset();
    error_update.reset();
    error_update_0.reset();
    error_update_norm.reset();

    print_conv_header();

    // We now perform a number of Newton iterations to iteratively solve the
    // nonlinear problem.  Since the problem is fully nonlinear and we are
    // using a full Newton method, the data stored in the tangent matrix and
    // right-hand side vector is not reusable and must be cleared at each
    // Newton step.  We then initially build the right-hand side vector to
    // check for convergence (and store this value in the first iteration).
    // The unconstrained DOFs of the rhs vector hold the out-of-balance
    // forces.
    unsigned int newton_iteration = 0;
    for (; newton_iteration < parameters.max_iterations_NR; ++newton_iteration)
      {
        pcout << " " << std::setw(2) << newton_iteration << " " << std::flush;

        // If we have decided that we want to continue with the iteration, we
        // assemble the tangent, make and impose the Dirichlet constraints,
        // and do the solve of the linearized system:
        make_constraints(newton_iteration);

        // update total solution prior to assembly

        // now ready to go-on and assmble linearized problem around the total
        // displacement
        // TODO: merge this function call with zeroing in main loop
        update_acceleration(delta_displacement);

        assemble_residual(newton_iteration);

        if (check_convergence(newton_iteration))
          break;

        // setup matrix-free part:
        update_matrix_free(newton_iteration);

        const std::tuple<unsigned int, double, double> lin_solver_output =
          solve_linear_system(newton_update);

        total_n_cg_iterations += std::get<0>(lin_solver_output);
        total_n_cg_solve++;

        if (newton_iteration == 0)
          error_update_0 = error_update;

        // We can now determine the normalised Newton update error, and
        // perform the actual update of the solution increment for the current
        // time step, update all quadrature point information pertaining to
        // this new displacement and stress state and continue iterating:
        error_update_norm = error_update;
        error_update_norm.normalise(error_update_0);

        if (newton_iteration != 0)
          delta_displacement += newton_update;
        else
          delta_displacement = newton_update;

        total_displacement += newton_update;


        pcout << " | " << std::fixed << std::setprecision(3) << std::setw(7)
              << std::scientific << std::get<0>(lin_solver_output) << "  "
              << std::get<1>(lin_solver_output) << "  "
              << std::get<2>(lin_solver_output) << "  " << error_residual_norm.e
              << "  " << error_residual.e << "  "
              << "  " << error_update_norm.e << "  " << error_update.e << "  "
              << std::endl;
      }

    // At the end, if it turns out that we have in fact done more iterations
    // than the parameter file allowed, we raise an exception.
    Assert(newton_iteration < parameters.max_iterations_NR,
           ExcMessage("No convergence in nonlinear solver!"));
  }


  // @sect4{Solid::print_conv_header, Solid::print_conv_footer and
  // Solid::print_solution}

  // This program prints out data in a nice table that is updated
  // on a per-iteration basis. The next two functions set up the table
  // header and footer:
  template <int dim, int degree, int n_q_points_1d, typename Number>
  void
  Solid<dim, degree, n_q_points_1d, Number>::print_conv_header()
  {
    static const unsigned int l_width = 98;

    for (unsigned int i = 0; i < l_width; ++i)
      pcout << "_";
    pcout << std::endl;

    pcout << "    SOLVER STEP    "
          << " |  LIN_IT    LIN_RES   COND_NUM   RES_NORM   "
          << "   RES_U      NU_NORM  "
          << "     NU_U" << std::endl;

    for (unsigned int i = 0; i < l_width; ++i)
      pcout << "_";
    pcout << std::endl;
  }



  template <int dim, int degree, int n_q_points_1d, typename Number>
  void
  Solid<dim, degree, n_q_points_1d, Number>::print_conv_footer()
  {
    static const unsigned int l_width = 98;

    for (unsigned int i = 0; i < l_width; ++i)
      pcout << "_";
    pcout << std::endl;

    pcout << "Relative errors:" << std::endl
          << "  Displacement: " << error_update_norm.e << std::endl
          << "  Force:        " << error_residual_norm.e << std::endl
          << "Absolute errors:" << std::endl
          << "  Displacement: " << error_update.e << std::endl
          << "  Force:        " << error_residual.e << std::endl
          << "Volume:         " << vol_current << " / " << vol_reference
          << std::endl;
  }



  // Print solution at a given point
  template <int dim, int degree, int n_q_points_1d, typename Number>
  void
  Solid<dim, degree, n_q_points_1d, Number>::print_solution_watchpoint()
  {
    if (!parameters.output_solution)
      return;

    static const unsigned int l_width = 87;

    for (unsigned int i = 0; i < l_width; ++i)
      pcout << "_";
    pcout << std::endl;

    for (const auto &soln_pt : parameters.output_points)
      {
        Tensor<1, dim> displacement;
        for (int d = 0; d < dim; ++d)
          displacement[d] = std::numeric_limits<double>::max();
        unsigned int found = 0;

        try
          {
            const MappingQ<dim> mapping(degree);
            const auto          cell_point =
              GridTools::find_active_cell_around_point(mapping,
                                                       dof_handler,
                                                       soln_pt);
            // we may find artifical cells here:
            if (cell_point.first->is_locally_owned())
              {
                found = 1;
                const Quadrature<dim> soln_qrule(cell_point.second);
                AssertThrow(soln_qrule.size() == 1, ExcInternalError());
                FEValues<dim> fe_values_soln(fe, soln_qrule, update_values);
                fe_values_soln.reinit(cell_point.first);

                std::vector<Tensor<1, dim>> soln_values(soln_qrule.size());
                fe_values_soln[u_fe].get_function_values(total_displacement,
                                                         soln_values);
                displacement = soln_values[0];
              }
          }
        catch (const GridTools::ExcPointNotFound<dim> &)
          {}

        for (unsigned int d = 0; d < dim; ++d)
          displacement[d] =
            Utilities::MPI::min(displacement[d], mpi_communicator);

        AssertThrow(Utilities::MPI::max(found, mpi_communicator) == 1,
                    ExcMessage("Found no cell with point inside!"));

        bcout << "Solution @ " << soln_pt << std::endl
              << "  displacement: " << displacement << std::endl;

      } // end loop over output points
  }



  // Update the acceleration according to Newmarks method
  template <int dim, int degree, int n_q_points_1d, typename Number>
  void
  Solid<dim, degree, n_q_points_1d, Number>::update_velocity(
    VectorType &displacement_delta)
  {
    velocity.equ(alpha_4, displacement_delta);
    velocity.add(alpha_5, velocity_old, alpha_6, acceleration_old);

    //      total_displacement_old = total_displacement;
    // TODO: maybe copy_locally_owned_data_from is sufficient here
    velocity_old     = velocity;
    acceleration_old = acceleration;
  }



  // Update the acceleration according to Newmarks method
  template <int dim, int degree, int n_q_points_1d, typename Number>
  void
  Solid<dim, degree, n_q_points_1d, Number>::update_acceleration(
    VectorType &displacement_delta)
  {
    acceleration.equ(alpha_1, displacement_delta);
    acceleration.add(-alpha_2, velocity_old, -alpha_3, acceleration_old);
  }


  // Note that we must ensure that
  // the matrix is reset before any assembly operations can occur.
  template <int dim, int degree, int n_q_points_1d, typename Number>
  void
  Solid<dim, degree, n_q_points_1d, Number>::assemble_residual(const int it_nr)
  {
    TimerOutput::Scope t(timer, "Assemble residual");
    pcout << " ASR " << std::flush;

    system_rhs = 0.0;

    // FIXME: The fast assembly (FEEValuation) fails sometimes to converge with
    // and stagnates shorty before the convergence limit as compared to the
    // FEValues assembly e.g. try one of the tests. Hence, we use it only for
    // the first five iterations and return to the more accurate assembly
    // afterwards. However, most of the cases will already be converged at this
    // stage.
    const bool assemble_fast = it_nr < 5;

    // The usual assembly strategy
    if (!assemble_fast)
      {
        Vector<double>                       cell_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::vector<Tensor<2, dim, Number>> solution_grads_u_total(
          qf_cell.size());
        std::vector<Tensor<1, dim, Number>> local_acceleration(qf_cell.size());

        // values at quadrature points:
        std::vector<Tensor<2, dim, Number>>          grad_Nx(dofs_per_cell);
        std::vector<SymmetricTensor<2, dim, Number>> symm_grad_Nx(
          dofs_per_cell);

        FEValues<dim> fe_values(
          fe, qf_cell, update_values | update_gradients | update_JxW_values);

        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const auto &cell_mat =
                (cell->material_id() == 2 ? material_inclusion : material);

              cell_rhs = 0.;
              fe_values.reinit(cell);
              cell->get_dof_indices(local_dof_indices);

              // We first need to find the solution gradients at quadrature
              // points inside the current cell and then we update each local QP
              // using the displacement gradient:
              fe_values[u_fe].get_function_gradients(total_displacement,
                                                     solution_grads_u_total);

              fe_values[u_fe].get_function_values(acceleration,
                                                  local_acceleration);

              // Now we build the residual. In doing so, we first extract some
              // configuration dependent variables from our QPH history objects
              // for the current quadrature point.
              for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                  const Tensor<2, dim, Number> &grad_u =
                    solution_grads_u_total[q_point];
                  const Tensor<2, dim, Number> F =
                    Physics::Elasticity::Kinematics::F(grad_u);
                  const SymmetricTensor<2, dim, Number> b =
                    Physics::Elasticity::Kinematics::b(F);

                  const Number det_F = determinant(F);
                  Assert(det_F > Number(0.0), ExcInternalError());
                  const Tensor<2, dim, Number> F_inv = invert(F);

                  // don't calculate b_bar if we don't need to:
                  const SymmetricTensor<2, dim, Number> b_bar =
                    cell_mat->formulation == 0 ?
                      Physics::Elasticity::Kinematics::b(
                        Physics::Elasticity::Kinematics::F_iso(F)) :
                      SymmetricTensor<2, dim, Number>();

                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                      grad_Nx[k] = fe_values[u_fe].gradient(k, q_point) * F_inv;
                      symm_grad_Nx[k] = symmetrize(grad_Nx[k]);
                    }

                  SymmetricTensor<2, dim, Number> tau;
                  cell_mat->get_tau(tau, det_F, b_bar, b);
                  const double JxW = fe_values.JxW(q_point);

                  // loop over j first to make caching a bit more
                  // straight-forward without recourse to symmetry
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      cell_rhs(j) -= (symm_grad_Nx[j] * tau) * JxW;
                      const unsigned int component_j =
                        fe.system_to_component_index(j).first;

                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        cell_rhs(j) -=
                          fe_values[u_fe].value(j, q_point) * cell_mat->rho *
                          fe_values[u_fe].value(i, q_point) *
                          local_acceleration[q_point][component_j] * JxW;
                    }

                } // end loop over quadrature points
              constraints.distribute_local_to_global(cell_rhs,
                                                     local_dof_indices,
                                                     system_rhs);
            }
      }
    else
      {
        FEEvaluation<dim, degree, n_q_points_1d, dim, Number> phi_reference(
          *mf_data_reference);
        // Copy constructor
        FEEvaluation<dim, degree, n_q_points_1d, dim, Number> phi_acc(
          phi_reference);
        const unsigned int n_cells = mf_data_reference->n_cell_batches();

        for (unsigned int cell = 0; cell < n_cells; ++cell)
          {
            const unsigned int material_id =
              mf_data_reference->get_cell_iterator(cell, 0)->material_id();
            const auto &cell_mat =
              (material_id == 0 ? material_vec : material_inclusion_vec);

            phi_reference.reinit(cell);
            phi_reference.read_dof_values_plain(total_displacement);
            phi_reference.evaluate(false, true, false);

            phi_acc.reinit(cell);
            phi_acc.read_dof_values_plain(acceleration);
            phi_acc.evaluate(true, false);


            // Now we build the residual. In doing so, we first extract some
            // configuration dependent variables from our QPH history objects
            // for the current quadrature point.
            for (unsigned int q_point = 0; q_point < phi_reference.n_q_points;
                 ++q_point)
              {
                const Tensor<2, dim, VectorizedArray<Number>> grad_u =
                  phi_reference.get_gradient(q_point);

                const Tensor<2, dim, VectorizedArray<Number>> F =
                  Physics::Elasticity::Kinematics::F(grad_u);

                const SymmetricTensor<2, dim, VectorizedArray<Number>> b =
                  Physics::Elasticity::Kinematics::b(F);

                const VectorizedArray<Number> det_F = determinant(F);

                Assert(*std::min_element(
                         det_F.begin(),
                         det_F.begin() +
                           mf_data_reference->n_active_entries_per_cell_batch(
                             cell)) > Number(0.0),
                       ExcInternalError());

                const Tensor<2, dim, VectorizedArray<Number>> F_inv = invert(F);

                // don't calculate b_bar if we don't need to:
                const SymmetricTensor<2, dim, VectorizedArray<Number>> b_bar =
                  cell_mat->formulation == 0 ?
                    Physics::Elasticity::Kinematics::b(
                      Physics::Elasticity::Kinematics::F_iso(F)) :
                    SymmetricTensor<2, dim, VectorizedArray<Number>>();

                SymmetricTensor<2, dim, VectorizedArray<Number>> tau;
                cell_mat->get_tau(tau, det_F, b_bar, b);

                const Tensor<2, dim, VectorizedArray<Number>> res =
                  Tensor<2, dim, VectorizedArray<Number>>(tau);

                phi_reference.submit_gradient(-res * transpose(F_inv), q_point);
                phi_acc.submit_value(-phi_acc.get_value(q_point) *
                                       cell_mat->rho,
                                     q_point);
              } // end loop over quadrature points

            phi_reference.integrate(false, true);
            phi_reference.distribute_local_to_global(system_rhs);
            phi_acc.integrate(true, false);
            phi_acc.distribute_local_to_global(system_rhs);
          }
      }



    FEFaceEvaluation<dim, degree, n_q_points_1d, dim, Number> phi(
      *mf_data_reference);
    auto q_index = precice_adapter->begin_interface_IDs();

    for (unsigned int face = mf_data_reference->n_inner_face_batches();
         face < mf_data_reference->n_boundary_face_batches() +
                  mf_data_reference->n_inner_face_batches();
         ++face)
      {
        const auto boundary_id = mf_data_reference->get_boundary_id(face);

        // Only interfaces
        if (boundary_id != int(TestCases::TestCaseBase<dim>::interface_id))
          continue;

        // Read out the total displacment
        phi.reinit(face);
        phi.gather_evaluate(total_displacement, false, true);
        // Number of active faces
        const auto active_faces =
          mf_data_reference->n_active_entries_per_face_batch(face);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            // Evaluate deformation gradient
            const auto F =
              Physics::Elasticity::Kinematics::F(phi.get_gradient(q));
            // Get the value from preCICE
            const auto traction =
              precice_adapter->read_on_quadrature_point(*q_index, active_faces);
            // Perform pull-back operation and submit value
            phi.submit_value(
              Physics::Transformations::Covariant::pull_back(traction, F), q);
            ++q_index;
          }
        // Integrate the result and write into the rhs vector
        phi.integrate_scatter(true, false, system_rhs);
      }

    system_rhs.compress(VectorOperation::add);

    // Determine the error in the residual for the unconstrained degrees of
    // freedom. Note that to do so, we need to ignore constrained DOFs by
    // setting the residual in these vector components to zero. That will not
    // affect the solution of linear system, though.
    constraints.set_zero(system_rhs);

    error_residual.e = system_rhs.l2_norm();
  }


  // @sect4{Solid::make_constraints}
  // The constraints for this problem are simple to describe.
  // However, since we are dealing with an iterative Newton method,
  // it should be noted that any displacement constraints should only
  // be specified at the zeroth iteration and subsequently no
  // additional contributions are to be made since the constraints
  // are already exactly satisfied.
  template <int dim, int degree, int n_q_points_1d, typename Number>
  void
  Solid<dim, degree, n_q_points_1d, Number>::make_constraints(const int &it_nr,
                                                              const bool print)
  {
    if (print)
      pcout << " CST " << std::flush;

    // Since the constraints are different at different Newton iterations, we
    // need to clear the constraints matrix and completely rebuild
    // it. However, after the first iteration, the constraints remain the same
    // and we can simply skip the rebuilding step if we do not clear it.

    // Note: Dirichlet boundary conditions are in structural mechanics usually
    // zero. Therefore, the constraints are not differently between Newton
    // iterations. We keep updating the constraints object anyway, i.e., the
    // assumption is not exploited here. However, the GMG constraints are not
    // adjusted accordingly and in case one wants to utilize other contraints,
    // the GMG needs to be reinitialized between different Newton iterations.
    if (it_nr > 1)
      return;

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);

    mg_constrained_dofs.clear();
    mg_constrained_dofs.initialize(dof_handler);

    const bool apply_dirichlet_bc = (it_nr == 0);

    // Use constraints functions and parameters as given from the input file
    // We also setup GMG constraints

    Functions::ZeroFunction<dim> zero(n_components);
    for (const auto &el : testcase->dirichlet)
      {
        const auto &mask = testcase->dirichlet_mask.find(el.first);
        Assert(mask != testcase->dirichlet_mask.end(),
               ExcMessage("Could not find component mask for ID " +
                          std::to_string(el.first)));

        mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                           {el.first},
                                                           mask->second);

        Function<dim> *func;
        if (apply_dirichlet_bc)
          func = el.second.get();
        else
          func = &zero;

        VectorTools::interpolate_boundary_values(
          dof_handler, el.first, *func, constraints, mask->second);
      }

    constraints.close();
  }



  // @sect4{Solid::solve_linear_system}
  // As the system is composed of a single block, defining a solution scheme
  // for the linear problem is straight-forward.
  template <int dim, int degree, int n_q_points_1d, typename Number>
  std::tuple<unsigned int, double, double>
  Solid<dim, degree, n_q_points_1d, Number>::solve_linear_system(
    VectorType &newton_update) const
  {
    unsigned int lin_it      = 0;
    double       lin_res     = 0.0;
    double       cond_number = 1.0;

    // reset solution vector each iteration
    newton_update = 0.;

    // We solve for the incremental displacement $d\mathbf{u}$.
    const double tol_sol = parameters.tol_lin * system_rhs.l2_norm();

    const bool estimate_condition = parameters.estimate_condition;

    // estimate condition number of matrix-free operator from dummy CG
    if (estimate_condition)
      {
        IterationNumberControl control_condition(
          parameters.cond_number_cg_iterations, tol_sol, false, false);
        SolverCG<VectorType> solver_condition(control_condition);

        solver_condition.connect_condition_number_slot(
          [&](const double number) { cond_number = number; });

        solver_condition.solve(mf_nh_operator,
                               newton_update,
                               system_rhs,
                               PreconditionIdentity());
        // reset back to zero
        newton_update = 0.;
      }

    timer.enter_subsection("Linear solver");
    const int solver_its = dof_handler.n_dofs() * parameters.max_iterations_lin;

    SolverControl solver_control(solver_its,
                                 tol_sol,
                                 (debug_level > 1 ? true : false),
                                 (debug_level > 0 ? true : false));

    pcout << " SLV " << std::flush;
    SolverCG<VectorType> solver_CG(solver_control);

    if (parameters.preconditioner_type == "jacobi")
      {
        PreconditionJacobi<NeoHookOperator<dim, degree, n_q_points_1d, double>>
          preconditioner;
        preconditioner.initialize(mf_nh_operator,
                                  parameters.preconditioner_relaxation);

        solver_CG.solve(mf_nh_operator,
                        newton_update,
                        system_rhs,
                        preconditioner);
      }
    else if (parameters.preconditioner_type == "none")
      {
        solver_CG.solve(mf_nh_operator,
                        newton_update,
                        system_rhs,
                        PreconditionIdentity());
      }
    else
      {
        AssertThrow(parameters.preconditioner_type == "gmg",
                    ExcMessage("Preconditioner type not implemented"));
        solver_CG.solve(mf_nh_operator,
                        newton_update,
                        system_rhs,
                        *multigrid_preconditioner);
      }

    lin_it  = solver_control.last_step();
    lin_res = solver_control.last_value();

    timer.leave_subsection();

    constraints.set_zero(newton_update);

    error_update.e = newton_update.l2_norm();

    // Now that we have the displacement update, distribute the constraints
    // back to the Newton update:
    // This function call does nothing special and is only for inhomogenous
    // constraints, which are not present in the current setup
    constraints.distribute(newton_update);

    return std::make_tuple(lin_it, lin_res, cond_number);
  }

  template <int dim, int degree, int n_q_points_1d, typename Number>
  void
  Solid<dim, degree, n_q_points_1d, Number>::output_results(
    const unsigned int result_number) const
  {
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;

    DataOut<dim> data_out;
    data_out.set_flags(flags);

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);

    std::vector<std::string> solution_name(dim, "displacement");

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(total_displacement,
                             solution_name,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    // Per-cell data (MPI subdomains):
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    Vector<float> material_id(triangulation.n_active_cells());
    Vector<float> manifold_id(triangulation.n_active_cells());
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          material_id[cell->active_cell_index()] = cell->material_id();
          manifold_id[cell->active_cell_index()] = cell->manifold_id();
        }

    data_out.add_data_vector(material_id, "material_id");
    data_out.add_data_vector(manifold_id, "manifold_id");

    // Visualize the displacements on a displaced grid
    // Recompute Eulerian mapping according to the current configuration
    MappingQEulerian<dim, VectorType> euler_mapping(degree,
                                                    dof_handler,
                                                    total_displacement);
    data_out.build_patches(euler_mapping,
                           degree,
                           DataOut<dim>::curved_inner_cells);

    const std::string filename = parameters.output_folder + "solution_" +
                                 Utilities::int_to_string(result_number, 3) +
                                 ".vtu";

    data_out.write_vtu_in_parallel(filename, mpi_communicator);

    // output MG mesh
    // Creates processor partition
    //    const std::string mg_mesh = parameters.output_folder + "mg_mesh";
    //    GridOut           grid_out;
    //    grid_out.write_mesh_per_processor_as_vtu(triangulation,
    //                                             mg_mesh,
    //                                             true,
    //                                             /*artificial*/ false);
  }
} // namespace FSI
