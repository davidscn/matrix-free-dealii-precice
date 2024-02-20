/* ---------------------------------------------------------------------
 * OpenDiHu muscle tendon coupling case
 * ---------------------------------------------------------------------
 */

#pragma once
#include <cases/case_base.h>


namespace TestCases
{
  template <int dim>
  struct Tendon : public TestCaseBase<dim>
  {
  public:
    virtual void
    make_coarse_grid_and_bcs(Triangulation<dim> &triangulation) override;

    virtual void
    refine_triangulation_and_finish_bcs(
      Triangulation<dim> &triangulation,
      const unsigned int  n_refinement_levels) const override;
  };



  template <int dim>
  void
  Tendon<dim>::make_coarse_grid_and_bcs(Triangulation<dim> &triangulation)
  {
    Assert(dim == 3, ExcNotImplemented());

    // Select initially a very coarse mesh
    uint       n_x = 1;
    uint       n_y = 1;
    uint       n_z = 1;
    Point<dim> point_bottom =
      dim == 3 ? Point<dim>(0.0, 0.0, 15) : Point<dim>(0, 0);
    Point<dim> point_tip =
      dim == 3 ? Point<dim>(3.0, 3.0, 20) : Point<dim>(0.0, 0.0);

    const std::vector<unsigned int> repetitions =
      dim == 3 ? std::vector<unsigned int>({n_x, n_y, n_z}) :
                 std::vector<unsigned int>({n_x, n_y});


    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              repetitions,
                                              point_bottom,
                                              point_tip,
                                              /*colorize*/ false);

    // NOTE: we need to set IDs regardless of cell being locally owned or
    // not as in the g lobal refinement cells will be repartitioned and faces
    // of their parents should have right IDs
    const unsigned int clamped_mesh_id = 1;
    const unsigned int do_nothing_id   = 2;

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
              // The left coupling interface
              if (std::abs(face->center()[2] - point_bottom[2]) < 1e-6)
                face->set_boundary_id(this->interfaces[0]);
              // The right coupling interface
              else if (std::abs(face->center()[2] - point_tip[2]) < 1e-6)
                face->set_boundary_id(this->interfaces[1]);
              else
                face->set_boundary_id(do_nothing_id);

              // AssertThrow(false, ExcMessage("Unknown boundary condition."));

              // on the coarse mesh reset material ID
            }
        cell->set_material_id(0);
      }
  }

  template <int dim>
  void
  Tendon<dim>::refine_triangulation_and_finish_bcs(
    Triangulation<dim> &triangulation,
    const unsigned int  n_refinement_levels) const
  {
    triangulation.refine_global(n_refinement_levels);
    const double z_height = 5;
    const double z_height_per_cell =
      z_height / std::pow(2, n_refinement_levels);

    // TODO: Make this dependent on the refinement level
    // you can use the information from the triangulation to do so
    const double       lower_x = (z_height / 2) - (z_height_per_cell / 2);
    const double       upper_x = (z_height / 2) + (z_height_per_cell / 2);
    const unsigned int clamped_mesh_id = 1;

    bool any_face_is_clamped = false;
    // Iterate over all cells and set the IDs
    for (const auto &cell : triangulation.active_cell_iterators())
      {
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary() == true)
            {
              // The left coupling interface
              if (face->center()[2] - lower_x > 1e-6 &&
                  face->center()[2] - upper_x < 1e-6)
                {
                  face->set_boundary_id(clamped_mesh_id);
                  any_face_is_clamped = true;
                }
              // on the coarse mesh reset material ID
            }
      }
    AssertThrow(any_face_is_clamped,
                ExcMessage("No Face was clamped, the problem is ill-posed."));
  }
} // namespace TestCases
