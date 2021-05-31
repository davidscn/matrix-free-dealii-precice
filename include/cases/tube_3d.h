#pragma once

#include <cases/case_base.h>

namespace TestCases
{
  template <int dim>
  struct Tube3D : public TestCaseBase<dim>
  {
  public:
    virtual void
    make_coarse_grid_and_bcs(Triangulation<dim> &triangulation) override;
  };



  template <int dim>
  void
  Tube3D<dim>::make_coarse_grid_and_bcs(Triangulation<dim> &triangulation)
  {
    Assert(dim == 3, ExcNotImplemented());
    // Select initially a very coarse mesh
    const double       length         = 5;
    const double       inner_radius   = .5;
    const double       outer_radius   = .6;
    const unsigned int n_radial_cells = 5;
    const unsigned int n_axial_cells  = 7;
    GridGenerator::cylinder_shell(triangulation,
                                  length,
                                  inner_radius,
                                  outer_radius,
                                  n_radial_cells,
                                  n_axial_cells);

    // NOTE: we need to set IDs regardless of cell being locally owned or
    // not as in the global refinement cells will be repartitioned and faces
    // of their parents should have right IDs
    const types::boundary_id clamped_mesh_id     = 1;
    const types::boundary_id do_nothing_boundary = 2;

    const double tol_boundary = 1e-6;
    // Set boundary conditions
    // Fix all boundary components
    this->dirichlet_mask[clamped_mesh_id] = ComponentMask(dim, true);
    this->dirichlet[clamped_mesh_id] =
      std::make_unique<Functions::ZeroFunction<dim>>(dim);

    // boundary IDs are obtained through colorize = true
    // Iterate over all cells and set the IDs
    for (const auto &cell : triangulation.active_cell_iterators())
      {
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary() == true)
            {
              const auto center = face->center(true);
              // Boundaries for the interface
              if ((std::pow(center[0], 2) + std::pow(center[1], 2)) <=
                  (std::pow(inner_radius, 2) + tol_boundary))
                face->set_boundary_id(this->interface_id);
              // Boundaries clamped in all directions
              else if ((std::abs(center[dim - 1]) < tol_boundary) ||
                       (std::abs(center[dim - 1]) > length - tol_boundary))
                face->set_boundary_id(clamped_mesh_id);
              // Boundaries clamped out-of-plane (z) direction
              else if ((std::pow(center[0], 2) + std::pow(center[1], 2)) >=
                       (std::pow(outer_radius, 2) - tol_boundary))
                face->set_boundary_id(do_nothing_boundary);
              else
                AssertThrow(false, ExcMessage("Unknown boundary condition."));
              // on the coarse mesh reset material ID
            }
        cell->set_material_id(0);
      }
    GridTools::scale(1e-2, triangulation);
  }
} // namespace TestCases
