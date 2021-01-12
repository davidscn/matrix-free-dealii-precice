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

namespace FSI
{
  namespace TestCases
  {
    template <int dim>
    struct TestCaseBase
    {
      static constexpr unsigned int interface_id = 11;

    public:
      virtual void
      make_coarse_grid(Triangulation<dim> &triangulation) const = 0;

    protected:
      void
      refine_boundary(Triangulation<dim> &     triangulation,
                      const types::boundary_id boundary_id) const;
    };



    template <int dim>
    void
    TestCaseBase<dim>::refine_boundary(
      Triangulation<dim> &     triangulation,
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
  } // namespace TestCases
} // namespace FSI
