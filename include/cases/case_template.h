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
      make_coarse_grid(Triangulation<dim> &triangulation) const override;
    };



    template <int dim>
    void
    MyCase<dim>::make_coarse_grid(Triangulation<dim> &triangulation) const
    {
      // Create your coarse mesh here
      // make sure you use this->interface_id for the coupling interface when
      // distributing the boundary IDs

      // material_id is used for the default material
      // material_id 2 is currently used for the inclusion material, which is
      // 100 x stiffer (mu) than the usual material
    }
  } // namespace TestCases
} // namespace FSI
