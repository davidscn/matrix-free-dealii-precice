#pragma once

#include <deal.II/base/parameter_handler.h>

using namespace dealii;

namespace FSI
{
  // @sect3{Run-time parameters}
  //
  // There are several parameters that can be set in the code so we set up a
  // ParameterHandler object to read in the choices at run-time.
  namespace Parameters
  {
    template <int dim>
    class Misc
    {
    public:
      std::string output_folder   = std::string("");
      bool        output_solution = true;

      std::vector<Point<dim>> output_points;

      void
      add_misc_parameters(ParameterHandler &prm);
    };

    template <int dim>
    void
    Misc<dim>::add_misc_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Misc");
      {
        prm.declare_entry("Output folder",
                          "",
                          Patterns::Anything(),
                          "Output folder (must exist)");

        prm.add_action("Output folder", [&](const std::string &value) {
          output_folder = value;
          if (!output_folder.empty() && output_folder.back() != '/')
            output_folder += "/";
        });

        prm.add_parameter("Output solution",
                          output_solution,
                          "Output solution and mesh",
                          Patterns::Bool());

        prm.add_parameter("Output points",
                          output_points,
                          "Points in undeformed configuration to "
                          "output unknown fields");
      }
      prm.leave_subsection();
    }

    template <int dim>
    class BoundaryConditions
    {
    public:
      std::map<types::boundary_id, std::unique_ptr<FunctionParser<dim>>>
                                                  dirichlet;
      std::map<types::boundary_id, ComponentMask> dirichlet_mask;

      std::map<types::boundary_id, std::unique_ptr<FunctionParser<dim>>>
        neumann;

      void
      add_bc_parameters(ParameterHandler &prm);
    };


    template <int dim>
    void
    BoundaryConditions<dim>::add_bc_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary conditions");
      prm.add_parameter("Dirichlet IDs and expressions",
                        dirichlet,
                        "Dirichlet functions for each boundary ID");

      prm.add_parameter("Dirichlet IDs and component mask",
                        dirichlet_mask,
                        "Dirichlet component mask for each boundary ID");

      prm.add_parameter("Neumann IDs and expressions",
                        neumann,
                        "Neumann functions for each boundary ID");

      prm.leave_subsection();
    }

    // @sect4{Finite Element system}

    // Here we specify the polynomial order used to approximate the solution.
    // The quadrature order should be adjusted accordingly.
    struct FESystem
    {
      unsigned int poly_degree = 2;
      unsigned int quad_order  = 3;

      void
      add_parameters(ParameterHandler &prm);
    };


    void
    FESystem::add_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.add_parameter("Polynomial degree",
                          poly_degree,
                          "Displacement system polynomial order",
                          Patterns::Integer(1));

        prm.add_parameter("Quadrature order",
                          quad_order,
                          "Gauss quadrature order",
                          Patterns::Integer(1));
      }
      prm.leave_subsection();
    }

    // @sect4{Geometry}

    // Make adjustments to the problem geometry and its discretisation.
    struct Geometry
    {
      unsigned int elements_per_edge   = 2;
      double       scale               = 1e-3;
      unsigned int dim                 = 2;
      unsigned int n_global_refinement = 0;
      unsigned int extrusion_slices    = 5;
      double       extrusion_height    = 1;
      std::string  type                = "CSM";

      void
      add_parameters(ParameterHandler &prm);
    };

    void
    Geometry::add_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        prm.add_parameter("Elements per edge",
                          elements_per_edge,
                          "Number of elements per long edge of the beam",
                          Patterns::Integer(0));

        prm.add_parameter("Global refinement",
                          n_global_refinement,
                          "Number of global refinements",
                          Patterns::Integer(0));

        prm.add_parameter("Grid scale",
                          scale,
                          "Global grid scaling factor",
                          Patterns::Double(0.0));

        prm.add_parameter("Extrusion height",
                          extrusion_height,
                          "Extrusion height",
                          Patterns::Double(0.0));

        prm.add_parameter("Extrusion slices",
                          extrusion_slices,
                          "Number of extrusion slices",
                          Patterns::Integer(0));

        prm.add_parameter("Dimension",
                          dim,
                          "Dimension of the problem",
                          Patterns::Integer(2, 3));

        prm.add_parameter("Type",
                          type,
                          "Type of the problem",
                          Patterns::Selection("CSM|Cook"));
      }
      prm.leave_subsection();
    }

    // @sect4{Materials}

    // We also need the shear modulus $ \mu $ and Poisson ration $ \nu $ for the
    // neo-Hookean material.
    struct Materials
    {
      double       nu                   = 0.4;
      double       mu                   = 0.5e6;
      double       rho                  = 1000;
      unsigned int material_formulation = 1;

      void
      add_parameters(ParameterHandler &prm);
    };

    void
    Materials::add_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        prm.add_parameter("Poisson's ratio",
                          nu,
                          "Poisson's ratio",
                          Patterns::Double(-1.0, 0.5));

        prm.add_parameter("Shear modulus",
                          mu,
                          "Shear modulus",
                          Patterns::Double(0.));

        prm.add_parameter("Density", rho, "Density", Patterns::Double(0.));

        prm.add_parameter("Formulation",
                          material_formulation,
                          "Formulation of the energy function",
                          Patterns::Integer(0, 1));
      }
      prm.leave_subsection();
    }

    // @sect4{Linear solver}

    // Next, we choose both solver and preconditioner settings.  The use of an
    // effective preconditioner is critical to ensure convergence when a large
    // nonlinear motion occurs within a Newton increment.
    struct LinearSolver
    {
      double       tol_lin                               = 1e-6;
      unsigned int max_iterations_lin                    = 1;
      std::string  preconditioner_type                   = "jacobi";
      double       preconditioner_relaxation             = 0.65;
      bool         estimate_condition                    = true;
      unsigned int cond_number_cg_iterations             = 20;
      std::string  mf_caching                            = "scalar";
      bool         mf_coarse_chebyshev                   = true;
      bool         mf_coarse_chebyshev_accurate_eigenval = true;
      unsigned int mf_chebyshev_n_cg_iterations          = 30;

      void
      add_parameters(ParameterHandler &prm);
    };

    void
    LinearSolver::add_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        prm.add_parameter("Residual",
                          tol_lin,
                          "Linear solver residual (scaled by residual norm)",
                          Patterns::Double(0.0));

        prm.add_parameter("Max iteration multiplier",
                          max_iterations_lin,
                          "Linear solver iterations "
                          "(multiples of the system matrix size)",
                          Patterns::Integer(1));

        prm.add_parameter("Preconditioner type",
                          preconditioner_type,
                          "Type of preconditioner",
                          Patterns::Selection("jacobi|gmg|none"));

        prm.add_parameter("Preconditioner relaxation",
                          preconditioner_relaxation,
                          "Preconditioner relaxation value",
                          Patterns::Double(0.0));

        prm.add_parameter("Estimate condition number",
                          estimate_condition,
                          "Enable or disable condition estimate",
                          Patterns::Bool());

        prm.add_parameter("Condition number CG iterations",
                          cond_number_cg_iterations,
                          "Number of CG iterations to "
                          "estimate condition number",
                          Patterns::Integer(1));

        prm.add_parameter(
          "MF caching",
          mf_caching,
          "Type of caching for matrix-free operator",
          Patterns::Selection(
            "scalar|scalar_referential|tensor2|tensor4|tensor4_ns"));

        prm.add_parameter(
          "MF Chebyshev number CG iterations",
          mf_chebyshev_n_cg_iterations,
          "Number of CG iterations to estiamte condition number "
          "for Chebyshev smoother",
          Patterns::Integer(2));

        prm.add_parameter("MF Chebyshev coarse",
                          mf_coarse_chebyshev,
                          "Use Chebyshev smoother as coarse level solver");

        prm.add_parameter(
          "MF Chebyshev coarse accurate eigenvalues",
          mf_coarse_chebyshev_accurate_eigenval,
          "Accurately estimate eigenvalues for coarse level Chebyshev"
          "solver");
      }
      prm.leave_subsection();
    }

    // @sect4{Nonlinear solver}

    // A Newton-Raphson scheme is used to solve the nonlinear system of
    // governing equations.  We now define the tolerances and the maximum number
    // of iterations for the Newton-Raphson nonlinear solver.
    struct NonlinearSolver
    {
      unsigned int max_iterations_NR = 10;
      double       tol_f             = 1e-9;
      double       tol_f_abs         = 0.;
      double       tol_u             = 1e-6;
      double       tol_u_abs         = 0;

      void
      add_parameters(ParameterHandler &prm);
    };

    void
    NonlinearSolver::add_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        prm.add_parameter("Max iterations Newton-Raphson",
                          max_iterations_NR,
                          "Number of Newton-Raphson iterations allowed",
                          Patterns::Integer(0));

        prm.add_parameter("Tolerance force",
                          tol_f,
                          "Force residual tolerance",
                          Patterns::Double(0.0));

        prm.add_parameter("Absolute tolerance force",
                          tol_f_abs,
                          "Force residual absolute tolerance",
                          Patterns::Double(0.0));

        prm.add_parameter("Absolute tolerance displacement",
                          tol_u_abs,
                          "Displacement update absolute tolerance",
                          Patterns::Double(0.0));

        prm.add_parameter("Tolerance displacement",
                          tol_u,
                          "Displacement error tolerance",
                          Patterns::Double(0.0));
      }
      prm.leave_subsection();
    }

    // @sect4{Time}

    // Set the timestep size $ \varDelta t $ and the simulation end-time.
    struct Time
    {
      double delta_t  = 0.1;
      double end_time = 1.;

      void
      add_parameters(ParameterHandler &prm);
    };

    void
    Time::add_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.add_parameter("End time", end_time, "End time", Patterns::Double());

        prm.add_parameter("Time step size",
                          delta_t,
                          "Time step size",
                          Patterns::Double(0.));
      }
      prm.leave_subsection();
    }

    // @sect4{All parameters}

    // Finally we consolidate all of the above structures into a single
    // container that holds all of our run-time selections.
    template <int dim>
    class AllParameters : public FESystem,
                          public Geometry,
                          public Materials,
                          public LinearSolver,
                          public NonlinearSolver,
                          public Time,
                          public Misc<dim>,
                          public BoundaryConditions<dim>

    {
    public:
      AllParameters(const std::string &input_file);

      void
      set_time(const double time) const;
    };

    template <int dim>
    void
    AllParameters<dim>::set_time(const double time) const
    {
      for (const auto &d : this->dirichlet)
        d.second->set_time(time);

      for (const auto &n : this->neumann)
        n.second->set_time(time);
    }

    template <int dim>
    AllParameters<dim>::AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;

      FESystem::add_parameters(prm);
      Geometry::add_parameters(prm);
      Materials::add_parameters(prm);
      LinearSolver::add_parameters(prm);
      NonlinearSolver::add_parameters(prm);
      Time::add_parameters(prm);

      this->add_misc_parameters(prm);
      this->add_bc_parameters(prm);

      prm.parse_input(input_file);

      AssertThrow((material_formulation == 0 && mf_caching == "scalar") ||
                    material_formulation == 1,
                  ExcInternalError());
      AssertDimension(dim, this->dim);
    }

  } // namespace Parameters
} // namespace FSI
