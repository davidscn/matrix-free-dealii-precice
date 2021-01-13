#pragma once

#include <cases/case_base.h>

namespace FSI
{
  namespace TestCases
  {
    template <int dim>
    struct TurekHron : public TestCaseBase<dim>
    {
    public:
      virtual void
      make_coarse_grid_and_bcs(Triangulation<dim> &triangulation) override;
    };



    template <int dim>
    void
    TurekHron<dim>::make_coarse_grid_and_bcs(Triangulation<dim> &triangulation)
    {
      // Select initially a very coarse mesh
      uint n_x = 5;
      uint n_y = 1;
      uint n_z = 1;

      const std::vector<unsigned int> repetitions =
        dim == 2 ? std::vector<unsigned int>({n_x, n_y}) :
                   std::vector<unsigned int>({n_x, n_y, n_z});

      const Point<dim> bottom_left =
        (dim == 3 ? Point<dim>(0.24899, 0.19, -0.005) :
                    Point<dim>(0.24899, 0.19));
      const Point<dim> top_right =
        dim == 3 ? Point<dim>(0.6, 0.21, 0.005) : Point<dim>(0.6, 0.21);

      GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                repetitions,
                                                bottom_left,
                                                top_right,
                                                /*colorize*/ true);
      const types::boundary_id clamped_mesh_id              = 1;
      const types::boundary_id out_of_plane_clamped_mesh_id = 2;

      // Set boundary conditions
      // Fix all boundary components
      this->dirichlet_mask[clamped_mesh_id] = ComponentMask(dim, true);
      this->dirichlet[clamped_mesh_id] =
        std::make_unique<Functions::ZeroFunction<dim>>(dim);
      // In case we run a 3D case, only the z component is fixed
      if (dim == 3)
        {
          this->dirichlet_mask[out_of_plane_clamped_mesh_id] =
            ComponentMask({false /*x*/, false /*y*/, true /*z*/});
          this->dirichlet[out_of_plane_clamped_mesh_id] =
            std::make_unique<Functions::ZeroFunction<dim>>(dim);
        }

      // boundary IDs are obtained through colorized = true
      uint id_flap_long_bottom         = 2; // x direction
      uint id_flap_long_top            = 3;
      uint id_flap_short_bottom        = 0; // y direction
      uint id_flap_short_top           = 1;
      uint id_flap_out_of_plane_bottom = 4; // z direction
      uint id_flap_out_of_plane_top    = 5;

      // Iterate over all cells and set the IDs
      for (const auto &cell : triangulation.active_cell_iterators())
        {
          for (const auto &face : cell->face_iterators())
            if (face->at_boundary() == true)
              {
                // Boundaries for the interface
                if (face->boundary_id() == id_flap_short_top ||
                    face->boundary_id() == id_flap_long_bottom ||
                    face->boundary_id() == id_flap_long_top)
                  face->set_boundary_id(this->interface_id);
                // Boundaries clamped in all directions
                else if (face->boundary_id() == id_flap_short_bottom)
                  face->set_boundary_id(clamped_mesh_id);
                // Boundaries clamped out-of-plane (z) direction
                else if (face->boundary_id() == id_flap_out_of_plane_bottom ||
                         face->boundary_id() == id_flap_out_of_plane_top)
                  face->set_boundary_id(out_of_plane_clamped_mesh_id);
                // on the coarse mesh reset material ID
              }
          cell->set_material_id(0);
        }
    }
  } // namespace TestCases
} // namespace FSI
