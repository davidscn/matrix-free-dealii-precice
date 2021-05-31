/* ---------------------------------------------------------------------
 * Setup for a 2D thin beam attached to square pillar in a channel flow.
 * Details as well as comparison results can be found in "Fluid-structure
 * interaction based upon a stabilized (ALE) finite element method" by
 * Wall and Ramm (1998).
 * ---------------------------------------------------------------------
 */

#pragma once

#include <cases/case_base.h>


namespace TestCases
{
  template <int dim>
  struct WallBeam : public TestCaseBase<dim>
  {
  public:
    virtual void
    make_coarse_grid_and_bcs(Triangulation<dim> &triangulation) override;
  };



  template <int dim>
  void
  WallBeam<dim>::make_coarse_grid_and_bcs(Triangulation<dim> &triangulation)
  {
    AssertDimension(dim, 2);
    // Create the mesh consisting of the square and the flap or only the
    // square
    static constexpr bool include_square = false;


    // We need (multiples of) five subdivisions in x-direction
    const std::vector<unsigned int> repetitions =
      dim == 2 ? std::vector<unsigned int>({4 * 5, 1}) :
                 std::vector<unsigned int>({4 * 5, 1, 1});

    const double x_coord = include_square ? -1.0 : 0.0;

    // Create a beam through the square
    const Point<dim> bottom_left =
      (dim == 3 ? Point<dim>(x_coord, -0.03, -0.005) :
                  Point<dim>(x_coord, -0.03));
    const Point<dim> top_right =
      dim == 3 ? Point<dim>(4.0, 0.03, 0.005) : Point<dim>(4.0, 0.03);

    // Boundary indicators are lost during the subsequent merge
    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              repetitions,
                                              bottom_left,
                                              top_right,
                                              /*colorize*/ false);

    // upper and lower half of the square contain the square
    if (include_square)
      {
        Triangulation<dim> tria1, tria2;
        {
          // Create the bottom part of the square
          const Point<dim> bottom_square =
            (dim == 3 ? Point<dim>(-1.0, -0.5, -0.005) :
                        Point<dim>(-1.0, -0.5));
          const Point<dim> top_square =
            dim == 3 ? Point<dim>(0.0, -0.03, 0.005) : Point<dim>(0.0, -0.03);

          GridGenerator::hyper_rectangle(tria1, bottom_square, top_square);
        }
        {
          // Create the top part of the square
          const Point<dim> bottom_square =
            (dim == 3 ? Point<dim>(-1.0, 0.03, -0.005) :
                        Point<dim>(-1.0, 0.03));
          const Point<dim> top_square =
            dim == 3 ? Point<dim>(0.0, 0.5, 0.005) : Point<dim>(0.0, 0.5);

          GridGenerator::hyper_rectangle(tria2, bottom_square, top_square);
        }

        // Merge everything together
        GridGenerator::merge_triangulations({&triangulation, &tria1, &tria2},
                                            triangulation);
      }

    // Set boundary conditions
    const types::boundary_id clamped_mesh_id = 1;
    // Fix all boundary components
    this->dirichlet_mask[clamped_mesh_id] = ComponentMask(dim, true);
    this->dirichlet[clamped_mesh_id] =
      std::make_unique<Functions::ZeroFunction<dim>>(dim);

    const double tolerance = 1e-8;
    // Iterate over all cells and set the IDs
    for (const auto &cell : triangulation.active_cell_iterators())
      {
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary() == true)
            {
              const auto center = face->center();
              // Boundaries for the interface
              if (center[0] > (0.0 + tolerance))
                face->set_boundary_id(this->interface_id);
              // Boundaries clamped in all directions
              else
                face->set_boundary_id(clamped_mesh_id);
            }
        cell->set_material_id(0);
      }
  }
} // namespace TestCases
