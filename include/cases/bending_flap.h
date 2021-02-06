#pragma once

#include <cases/case_base.h>

namespace FSI
{
  namespace TestCases
  {
    template <int dim>
    struct BendingFlap : public TestCaseBase<dim>
    {
    public:
      virtual void
      make_coarse_grid_and_bcs(Triangulation<dim> &triangulation) override;
    };



    template <int dim>
    void
    BendingFlap<dim>::make_coarse_grid_and_bcs(
      Triangulation<dim> &triangulation)
    {
      Assert(dim == 3, ExcNotImplemented());

      // Select initially a very coarse mesh
      uint       n_x = 1;
      uint       n_y = 2;
      uint       n_z = 3;
      Point<dim> point_bottom =
        dim == 3 ? Point<dim>(0.49, 0.0, -0.3) : Point<dim>(0.49, 0);
      Point<dim> point_tip =
        dim == 3 ? Point<dim>(0.56, 0.35, 0.3) : Point<dim>(0.56, 0.35);

      const std::vector<unsigned int> repetitions =
        dim == 3 ? std::vector<unsigned int>({n_x, n_y, n_z}) :
                   std::vector<unsigned int>({n_x, n_y});


      GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                repetitions,
                                                point_bottom,
                                                point_tip,
                                                /*colorize*/ false);

      // NOTE: we need to set IDs regardless of cell being locally owned or
      // not as in the global refinement cells will be repartitioned and faces
      // of their parents should have right IDs
      const unsigned int clamped_mesh_id = 1;

      // Set boundary conditions
      // Fix all boundary components
      this->dirichlet_mask[clamped_mesh_id] = ComponentMask(dim, true);
      this->dirichlet[clamped_mesh_id] =
        std::make_unique<Functions::ZeroFunction<dim>>(dim);


      // Iterate over all cells and set the IDs
      for (const auto &cell : triangulation.active_cell_iterators())
        {
          for (const auto &face : cell->face_iterators())
            if (face->at_boundary() == true)
              {
                // Boundaries clamped in all directions, bottom y
                if (face->center()[1] < 1e-6)
                  face->set_boundary_id(clamped_mesh_id);
                // Boundaries for the interface: x, z and top y
                else if (face->center()[1] > 1e-6)
                  face->set_boundary_id(this->interface_id);
                else
                  AssertThrow(false, ExcMessage("Unknown boundary condition."));

                // on the coarse mesh reset material ID
              }
          cell->set_material_id(0);
        }
    }
  } // namespace TestCases
} // namespace FSI
