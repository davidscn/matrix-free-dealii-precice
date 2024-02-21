#pragma once
/*
To use this class, it requires ascii vtk data, which can be obtaines by
meshio convert example_in_file.vtu -o vtk42 inFile.vtk -a
Afterwards, change the header of the vtk line from version 4.2. to 3.0 manually
(should be compatible) Other workflows might work as well.
*/
#include <deal.II/grid/grid_in.h>

#include <cases/case_base.h>

#include <filesystem>

namespace TestCases
{
  template <int dim>
  struct MeshIn : public TestCaseBase<dim>
  {
  public:
    virtual void
    make_coarse_grid_and_bcs(Triangulation<dim> &triangulation) override;
  };



  template <int dim>
  void
  MeshIn<dim>::make_coarse_grid_and_bcs(Triangulation<dim> &triangulation)
  {
    dealii::GridIn<dim> gridIn;

    gridIn.attach_triangulation(triangulation);
    std::string inFile = "inFile.vtk";

    std::string extension = std::filesystem::path(inFile).extension();

    AssertThrow(!extension.empty(),
                dealii::ExcMessage("No valid file extension!"));

    // erase the leading "."
    extension.erase(0, 1);

    typename dealii::GridIn<dim>::Format format;

    if (extension == "e" || extension == "exo")
      format = dealii::GridIn<dim>::Format::exodusii;
    else
      format = gridIn.parse_format(extension);

    typename dealii::GridIn<dim>::ExodusIIData exodusIIData;

    if (format == dealii::GridIn<dim>::Format::exodusii)
      exodusIIData = gridIn.read_exodusii("inFile");
    else
      gridIn.read(inFile, format);

    // Implement boundary conditions
    // const types::boundary_id clamped_mesh_id     = 1;
    // const types::boundary_id do_nothing_boundary = 2;

    // const double tol_boundary = 1e-6;
    // Set boundary conditions
    // Fix all boundary components
    // this->dirichlet_mask[clamped_mesh_id] = ComponentMask(dim, true);
    // this->dirichlet[clamped_mesh_id] =
    //   std::make_unique<Functions::ZeroFunction<dim>>(dim);

    // Iterate over all cells and set the IDs
    // for (const auto &cell : triangulation.active_cell_iterators())
    //   {
    //     for (const auto &face : cell->face_iterators())
    //       if (face->at_boundary() == true)
    //         {
    //           // Boundaries clamped in all directions, bottom y
    //           if (face->center()[1] < 1e-6)
    //             face->set_boundary_id(clamped_mesh_id);
    //           // Boundaries for the interface: x, z and top y
    //           else if (face->center()[1] > 1e-6)
    //             face->set_boundary_id(this->interface_id);
    //           else
    //             AssertThrow(false, ExcMessage("Unknown boundary
    //             condition."));

    //           // on the coarse mesh reset material ID
    //         }
    //     cell->set_material_id(0);
    //   }
  }
} // namespace TestCases
