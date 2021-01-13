#pragma once

#include <cases/case_base.h>

namespace FSI
{
  namespace TestCases
  {
    template <int dim>
    struct MyCase : public TestCaseBase<dim>
    {
    public:
      virtual void
      make_coarse_grid_and_bcs(Triangulation<dim> &triangulation) override;
    };



    template <int dim>
    void
    MyCase<dim>::make_coarse_grid_and_bcs(Triangulation<dim> &triangulation)
    {
      // Create your coarse mesh here
      // make sure you use this->interface_id for the coupling interface when
      // distributing the boundary IDs.
      // Dirichlet boundary conditions are applied through this->dirichlet and
      // this->dirichlet_mask. Have a look at other cases.

      // material_id 2 is currently used for the inclusion material, which is
      // 100 x stiffer (mu) than the usual material. Other IDs are used for the
      // default material
    }
  } // namespace TestCases
} // namespace FSI
