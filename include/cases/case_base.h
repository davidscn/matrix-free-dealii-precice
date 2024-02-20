#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>


using namespace dealii;

namespace TestCases
{
  template <int dim>
  struct TestCaseBase
  {
  public:
    static constexpr std::array<types::boundary_id, 2> interfaces{{11, 12}};

    static constexpr types::boundary_id interface_id = 11;
    // Optional TODO in case of more involved BCs: use FunctionParser
    std::map<types::boundary_id, std::unique_ptr<Function<dim>>> dirichlet;
    std::map<types::boundary_id, ComponentMask>                  dirichlet_mask;
    std::unique_ptr<Function<dim>>                               body_force;
    std::unique_ptr<Function<dim>> initial_condition;
    std::unique_ptr<Function<dim>> heat_transfer_rhs;

    bool body_force_is_spatially_constant = true;
    // Boolean for the simulation type (only used for heat transfer).
    // is_dirichlet = true reads and applys Dirichlet boundary condition,
    // is_dirichlet = false reads and applys Neumann boundary condition
    bool is_dirichlet = false;
    virtual void
    make_coarse_grid_and_bcs(Triangulation<dim> &triangulation) = 0;

    virtual void
    refine_triangulation_and_finish_bcs(
      Triangulation<dim> &triangulation,
      const unsigned int  n_refinement_levels) const;

    virtual ~TestCaseBase() = default;

  protected:
    void
    refine_boundary(Triangulation<dim>      &triangulation,
                    const types::boundary_id boundary_id) const;
  };



  template <int dim>
  void
  TestCaseBase<dim>::refine_boundary(Triangulation<dim>      &triangulation,
                                     const types::boundary_id boundary_id) const
  {
    for (const auto &cell : triangulation.active_cell_iterators())
      {
        for (auto f : GeometryInfo<dim>::face_indices())
          {
            const auto face = cell->face(f);

            if (face->at_boundary() && face->boundary_id() == boundary_id)
              cell->set_refine_flag();
          }
      }
    triangulation.prepare_coarsening_and_refinement();
    triangulation.execute_coarsening_and_refinement();
  }

  template <int dim>
  void
  TestCaseBase<dim>::refine_triangulation_and_finish_bcs(
    Triangulation<dim> &triangulation,
    const unsigned int  n_refinement_levels) const
  {
    triangulation.refine_global(n_refinement_levels);
  }
} // namespace TestCases
