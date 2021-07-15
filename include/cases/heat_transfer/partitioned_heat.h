#pragma once

#include <cases/case_base.h>

namespace TestCases
{
  template <int dim>
  struct PartitionedHeat : public TestCaseBase<dim>
  {
  public:
    PartitionedHeat(const bool is_dirichlet);

    virtual void
    make_coarse_grid_and_bcs(Triangulation<dim> &triangulation) override;

  private:
    // Constants of the problem
    const double alpha = 3;
    const double beta  = 1.3;
  };

  // The analytic solution of the test case
  // giving the Dirichlet boundary conditions
  // 1 + x^2 + alpha y^2 + beta * t
  template <int dim>
  class AnalyticSolution : public Function<dim>
  {
  public:
    AnalyticSolution(const double alpha, const double beta)
      : Function<dim>()
      , alpha(alpha)
      , beta(beta)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const
    {
      (void)component;
      AssertIndexRange(component, 1);
      const double time = this->get_time();
      return 1 + (p[0] * p[0]) + (alpha * p[1] * p[1]) + (beta * time);
    }

  private:
    const double alpha;
    const double beta;
  };

  // The constant RHS function
  // beta - 2 - (2 * alpha)
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide(const double alpha, const double beta)
      : Function<dim>()
      , alpha(alpha)
      , beta(beta)
    {}
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }


    template <typename number>
    number
    value(const Point<dim, number> & /*p*/,
          const unsigned int component = 0) const
    {
      (void)component;
      AssertIndexRange(component, 1);
      return beta - 2 - (2 * alpha);
    }

  private:
    const double alpha;
    const double beta;
  };



  template <int dim>
  PartitionedHeat<dim>::PartitionedHeat(const bool is_dirichlet)
  {
    this->is_dirichlet = is_dirichlet;
  }



  template <int dim>
  void
  PartitionedHeat<dim>::make_coarse_grid_and_bcs(
    Triangulation<dim> &triangulation)
  {
    // The Dirichlet domain is located on the left, the Neumann domain is
    // located on the right
    const double root_location = this->is_dirichlet ? 0 : 1;
    Assert(dim == 2, ExcNotImplemented());

    const types::boundary_id dirichlet_id = 1;

    const unsigned int mpi_size =
      Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation_in_groups<dim, dim>(
        [&](Triangulation<dim> &tria) {
          Triangulation<dim> hex_tria;
          GridGenerator::hyper_rectangle(hex_tria,
                                         Point<dim>{0 + root_location, 0},
                                         Point<dim>{1 + root_location, 1},
                                         false);
          GridGenerator::convert_hypercube_to_simplex_mesh(hex_tria, tria);
          const double tol_boundary = 1e-6;
          for (const auto &cell : tria.active_cell_iterators())
            for (const auto &face : cell->face_iterators())
              if (face->at_boundary() == true)
                {
                  const auto center = face->center();
                  // Boundaries for the interface at x = 1
                  if (center[0] >= (1 - tol_boundary) &&
                      center[0] <= (1 + tol_boundary))
                    face->set_boundary_id(this->interface_id);
                  else
                    // Boundaries for the dirichlet boundary
                    face->set_boundary_id(dirichlet_id);
                }

          tria.refine_global(3);
        },
        [&](Triangulation<dim> &tria_serial,
            const MPI_Comm /*mpi_comm*/,
            const unsigned int /*group_size*/) {
          GridTools::partition_triangulation(mpi_size, tria_serial);
        },
        MPI_COMM_WORLD,
        1,
        Triangulation<dim>::limit_level_difference_at_vertices,
        TriangulationDescription::construct_multigrid_hierarchy);
    triangulation.create_triangulation(construction_data);


    this->dirichlet_mask[dirichlet_id] = ComponentMask(1, true);
    this->dirichlet[dirichlet_id] =
      std::make_unique<AnalyticSolution<dim>>(alpha, beta);
    this->dirichlet[dirichlet_id]->set_time(0.0);

    this->heat_transfer_rhs = std::make_unique<RightHandSide<dim>>(alpha, beta);
    this->initial_condition =
      std::make_unique<AnalyticSolution<dim>>(alpha, beta);
    this->initial_condition->set_time(0.0);
  }
} // namespace TestCases
