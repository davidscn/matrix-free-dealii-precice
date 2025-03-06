#pragma once

#include <cases/case_base.h>

namespace CaseUtilities
{

  template <int dim>
  double
  getMaxEdgeLengthAtBoundary(const Triangulation<dim> &tria,
                             const types::boundary_id  id)
  {
    double max_length = 0;
    for (const auto &cell : tria.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() && face->boundary_id() == id)
          {
            max_length = std::max(max_length, face->measure());
          }

    return max_length;
  }
} // namespace CaseUtilities

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
    const double alpha = 1;
    const double beta  = 2 * numbers::PI;
    const double gamma = 1;
  };

  // The analytic solution of the test case
  // giving the Dirichlet boundary conditions
  // 1 + x^2 + alpha y^2 + beta * t
  template <int dim>
  class AnalyticSolution : public Function<dim>
  {
  public:
    AnalyticSolution(const double alpha, const double beta, const double gamma)
      : Function<dim>()
      , alpha(alpha)
      , beta(beta)
      , gamma(gamma)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      (void)component;
      AssertIndexRange(component, 1);
      const double time = this->get_time();
      auto         x    = p[0];
      auto         y    = p[1];
      return std::cos(alpha * x) + y * std::cos(beta * y) + (gamma * time);
    }

  private:
    const double alpha;
    const double beta;
    const double gamma;
  };

  // The constant RHS function
  // beta - 2 - (2 * alpha)
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide(const double alpha, const double beta, const double gamma)
      : Function<dim>()
      , alpha(alpha)
      , beta(beta)
      , gamma(gamma)
    {}
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }


    template <typename number>
    number
    value(const Point<dim, number> &p, const unsigned int component = 0) const
    {
      (void)component;
      AssertIndexRange(component, 1);
      auto x = p[0];
      auto y = p[1];
      return Utilities::fixed_power<2>(alpha) * std::cos(alpha * x) +
             2 * beta * std::sin(beta * y) +
             Utilities::fixed_power<2>(beta) * y * std::cos(beta * y) + gamma;
    }

  private:
    const double alpha;
    const double beta;
    const double gamma;
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
    const double r_in = 0.35; // Radius of the hole/cylinder
    const double r_out =
      0.5; // Outer "radius" => bounding square from -0.5..0.5
    const unsigned int n_cells  = 4;
    const bool         colorize = false;
    // const double       tol      = 1e-10; // Tolerance for matching radii

    const bool         disable_precice = true;
    Triangulation<dim> tria1, tria2;
    if (disable_precice)
      {
        GridGenerator::hyper_ball_balanced(tria1,
                                           /*center=*/Point<dim>(0, 0),
                                           /*outer_radius=*/r_in);


        GridGenerator::hyper_cube_with_cylindrical_hole(
          tria2, r_in, r_out, n_cells, 1, colorize);
      }

    GridGenerator::merge_triangulations(
      tria1, tria2, triangulation, 1.0e-12, true, false);

    // By default, the manifold_id is set to 0 on the boundary faces, 1 on the
    // boundary cells, and numbers::flat_manifold_id on the central cell and on
    // internal faces.
    PolarManifold<dim> polar_manifold(Point<dim>(0, 0));
    triangulation.set_manifold(0, polar_manifold);
    // The Dirichlet domain is located on the left, the Neumann domain is
    // located on the right

    const types::boundary_id dirichlet_id = 1;

    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary())
          {
            face->set_boundary_id(dirichlet_id);
          }

    // const double tol_boundary = 1e-6;
    // for (const auto &cell : triangulation.active_cell_iterators())
    //   for (const auto &face : cell->face_iterators())
    //     if (face->at_boundary() == true)
    //       {
    //         const auto center = face->center();
    //         // Boundaries for the interface at x = 1
    //         if (center[0] >= (1 - tol_boundary) &&
    //             center[0] <= (1 + tol_boundary))
    //           face->set_boundary_id(this->interface_id);
    //         else
    //           // Boundaries for the dirichlet boundary
    //           face->set_boundary_id(dirichlet_id);
    // }

    this->dirichlet_mask[dirichlet_id] = ComponentMask(1, true);
    this->dirichlet[dirichlet_id] =
      std::make_unique<AnalyticSolution<dim>>(alpha, beta, gamma);
    this->dirichlet[dirichlet_id]->set_time(0.0);

    this->heat_transfer_rhs =
      std::make_unique<RightHandSide<dim>>(alpha, beta, gamma);
    this->initial_condition =
      std::make_unique<AnalyticSolution<dim>>(alpha, beta, gamma);
    this->initial_condition->set_time(0.0);
  }
} // namespace TestCases
