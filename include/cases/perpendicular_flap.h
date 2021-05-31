#pragma once

#include <cases/case_base.h>


namespace TestCases
{
  template <int dim>
  struct PerpendicularFlap : public TestCaseBase<dim>
  {
  public:
    virtual void
    make_coarse_grid_and_bcs(Triangulation<dim> &triangulation) override;
  };



  template <int dim>
  void
  PerpendicularFlap<dim>::make_coarse_grid_and_bcs(
    Triangulation<dim> &triangulation)
  {
    // Select initially a very coarse mesh
    // Flap_perp case
    uint       n_x = 1;
    uint       n_y = 5;
    uint       n_z = 1;
    Point<dim> point_bottom =
      dim == 3 ? Point<dim>(-0.05, 0, 0) : Point<dim>(-0.05, 0);
    Point<dim> point_tip =
      dim == 3 ? Point<dim>(0.05, 1, 0.3) : Point<dim>(0.05, 1);

    const std::vector<unsigned int> repetitions =
      dim == 2 ? std::vector<unsigned int>({n_x, n_y}) :
                 std::vector<unsigned int>({n_x, n_y, n_z});


    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              repetitions,
                                              point_bottom,
                                              point_tip,
                                              /*colorize*/ true);

    // NOTE: we need to set IDs regardless of cell being locally owned or
    // not as in the global refinement cells will be repartitioned and faces
    // of their parents should have right IDs
    const unsigned int clamped_mesh_id              = 1;
    const unsigned int out_of_plane_clamped_mesh_id = 2;

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

    // boundary IDs are obtained through colorize = true

    // IDs for PF
    uint id_flap_long_bottom         = 0; // x direction
    uint id_flap_long_top            = 1;
    uint id_flap_short_bottom        = 2; // y direction
    uint id_flap_short_top           = 3;
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
