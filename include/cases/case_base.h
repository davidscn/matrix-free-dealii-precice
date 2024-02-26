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
    static constexpr types::boundary_id interface_id = 11;
    // Optional TODO in case of more involved BCs: use FunctionParser
    std::map<types::boundary_id, std::unique_ptr<Function<dim>>> dirichlet;
    std::map<types::boundary_id, std::unique_ptr<Function<dim>>> neumann;
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

    virtual ~TestCaseBase() = default;

  protected:
    void
    refine_boundary(Triangulation<dim>      &triangulation,
                    const types::boundary_id boundary_id) const;
    void
    refine_in_direction(Triangulation<dim> &triangulation,
                                     int direction) const;
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
  TestCaseBase<dim>::refine_in_direction(Triangulation<dim>      &triangulation,
                                     int direction) const
  {
    for (const auto &cell : triangulation.active_cell_iterators())
      {
	cell->clear_refine_flag();
             if (direction == 0)
              cell->set_refine_flag(RefinementCase<dim>::cut_x);
             else if (direction == 1)
              cell->set_refine_flag(RefinementCase<dim>::cut_y);
             else if (direction == 2)
		{
		if constexpr (dim == 3)
              	  {  cell->set_refine_flag(RefinementCase<dim>::cut_z); }
		}
             else
	      AssertThrow(false, ExcNotImplemented());
      }
    triangulation.prepare_coarsening_and_refinement();
    triangulation.execute_coarsening_and_refinement();
  }

} // namespace TestCases
