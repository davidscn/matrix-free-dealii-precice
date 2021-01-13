#pragma once

#include <cases/case_base.h>

namespace FSI
{
  namespace TestCases
  {
    template <int dim>
    struct CookMembrane : public TestCaseBase<dim>
    {
    public:
      virtual void
      make_coarse_grid_and_bcs(Triangulation<dim> &triangulation) override;
    };



    template <int dim>
    void
    CookMembrane<dim>::make_coarse_grid_and_bcs(
      Triangulation<dim> &triangulation)
    {
      // Divide the beam, but only along the x- and y-coordinate directions
      std::vector<unsigned int> repetitions(dim, 2 /*elements_per_edge*/);
      // Only allow one element through the thickness
      // (modelling a plane strain condition)
      if (dim == 3)
        repetitions[dim - 1] = 1;

      const Point<dim> bottom_left =
        (dim == 3 ? Point<dim>(0.0, 0.0, -0.5) : Point<dim>(0.0, 0.0));
      const Point<dim> top_right =
        (dim == 3 ? Point<dim>(48.0, 44.0, 0.5) : Point<dim>(48.0, 44.0));

      GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                repetitions,
                                                bottom_left,
                                                top_right);

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

      // Since we wish to apply a Neumann BC to the right-hand surface, we
      // must find the cell faces in this part of the domain and mark them
      // with a distinct boundary ID number.  The faces we are looking for are
      // on the +x surface and will get boundary ID 11. Dirichlet boundaries
      // exist on the left-hand face of the beam (this fixed boundary will get
      // ID 1) and on the +Z and -Z faces (which correspond to ID 2 and we
      // will use to impose the plane strain condition)
      const double tol_boundary = 1e-6;
      // NOTE: we need to set IDs regardless of cell being locally owned or
      // not as in the global refinement cells will be repartitioned and faces
      // of their parents should have right IDs
      for (auto cell : triangulation.active_cell_iterators())
        {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            if (cell->face(face)->at_boundary() == true)
              {
                if (std::abs(cell->face(face)->center()[0] - 0.0) <
                    tol_boundary)
                  cell->face(face)->set_boundary_id(
                    clamped_mesh_id); // -X faces
                else if (std::abs(cell->face(face)->center()[0] - 48.0) <
                         tol_boundary)
                  cell->face(face)->set_boundary_id(
                    this->interface_id); // +X faces
                else if (std::abs(std::abs(cell->face(face)->center()[0]) -
                                  0.5) < tol_boundary)
                  cell->face(face)->set_boundary_id(
                    out_of_plane_clamped_mesh_id); // +Z and -Z faces
              }
          // on the coarse mesh reset material ID
          cell->set_material_id(0);
        }

      // Transform the hyper-rectangle into the beam shape
      GridTools::transform(
        [](auto &pt_in) {
          const double &x = pt_in[0];
          const double &y = pt_in[1];

          const double y_upper =
            44.0 + (16.0 / 48.0) * x; // Line defining upper edge of beam
          const double y_lower =
            0.0 + (44.0 / 48.0) * x; // Line defining lower edge of beam
          const double theta =
            y / 44.0; // Fraction of height along left side of beam
          const double y_transform =
            (1 - theta) * y_lower + theta * y_upper; // Final transformation

          Point<dim> pt_out = pt_in;
          pt_out[1]         = y_transform;

          return pt_out;
        },
        triangulation);
      GridTools::scale(1e-3, triangulation);
    }
  } // namespace TestCases
} // namespace FSI
