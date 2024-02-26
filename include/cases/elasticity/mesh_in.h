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
  class StretchRamp : public Function<dim>
  {
  public:
    StretchRamp(const double load, const double ramp_end_time)
      : Function<dim>()
      , load(load)
      , ramp_end_time(ramp_end_time)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component) const override
    {
      (void)p;
      AssertIndexRange(component, dim);
      const double time = this->get_time();
      if (component == dim - 1)
        return load * (std::min(time, ramp_end_time) / ramp_end_time);
      else
        return 0;
    }

  private:
    const double load;
    const double ramp_end_time;
  };


  template <int dim>
  void
  MeshIn<dim>::make_coarse_grid_and_bcs(Triangulation<dim> &triangulation)
  {
    dealii::GridIn<dim> gridIn;

    gridIn.attach_triangulation(triangulation);
    std::string inFile = this->mesh_in_filename;

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
      exodusIIData = gridIn.read_exodusii(inFile);
    else
      gridIn.read(inFile, format);

    // bottom tendon: -62.4 > z --> apply traction
    // bottom tendon: -56.5 < z --> coupling interface
    // the fixed traction: surface area: 0.4907550026758
    // 100 N /0.4907550026758 = 203.76766299835657
    // let's assume 0.5 --> 200
    // ramping it up to this value in 100 ms
    this->body_force = std::make_unique<Functions::ConstantFunction<dim>>(
      std::vector<double>{0, 0, -9.81e-4});


    // Implement boundary conditions
    const types::boundary_id clamped_mesh_id     = 1;
    const types::boundary_id stretched_mesh_id   = 5;
    const types::boundary_id do_nothing_boundary = 2;

    const double upper_tendon_upper_limit = -33.5;
    const double upper_tendon_lower_limit = -41.4;

    const double bottom_tendon_upper_limit = -56.5;
    const double bottom_tendon_lower_limit = -62.4;
    // Set boundary conditions
    // Fix all boundary components
    this->dirichlet_mask[clamped_mesh_id] = ComponentMask(dim, true);
    this->dirichlet[clamped_mesh_id] =
      std::make_unique<Functions::ZeroFunction<dim>>(dim);
    this->neumann[stretched_mesh_id] =
      std::make_unique<StretchRamp<dim>>(-100, 1000);
    this->neumann[stretched_mesh_id]->set_time(0.0);

    // Iterate over all cells and set the IDs
    for (const auto &cell : triangulation.active_cell_iterators())
      {
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary() == true)
            {
              // Boundaries clamped in all directions, bottom y
              if (face->center()[dim - 1] < bottom_tendon_lower_limit)
                face->set_boundary_id(stretched_mesh_id);
              // Boundaries for the interface: x, z and top y
              else if (face->center()[dim - 1] > bottom_tendon_upper_limit &&
                       face->center()[dim - 1] < upper_tendon_lower_limit)
                face->set_boundary_id(this->interface_id);
              else if (face->center()[dim - 1] > upper_tendon_upper_limit)
                face->set_boundary_id(clamped_mesh_id);
              else
                face->set_boundary_id(do_nothing_boundary);

              // on the coarse mesh reset material ID
            }
        cell->set_material_id(0);
      }
    // Not supported for fully distributed triangulations
    // this->refine_in_direction(triangulation, 2);
  }
} // namespace TestCases
