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
    const bool disable_precice = false;
    if (disable_precice)
      {
        Assert(dim == 2, ExcNotImplemented());
        GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                  std::vector<unsigned int>{2,
                                                                            1},
                                                  Point<dim>{0, 0},
                                                  Point<dim>{2, 1},
                                                  false);
      }
    else
      {
        const double root_location = this->is_dirichlet ? 0 : 1;
        Assert(dim == 2, ExcNotImplemented());
        GridGenerator::hyper_rectangle(triangulation,
                                       Point<dim>{0 + root_location, 0},
                                       Point<dim>{1 + root_location, 1},
                                       false);
      }

    const types::boundary_id dirichlet_id = 1;

    const double tol_boundary = 1e-6;
    for (const auto &cell : triangulation.active_cell_iterators())
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
