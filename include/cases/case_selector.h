#pragma once

#include <cases/case_base.h>
#include <cases/elasticity/Wall_beam.h>
#include <cases/elasticity/bending_flap.h>
#include <cases/elasticity/cook_membrane.h>
#include <cases/elasticity/perpendicular_flap.h>
#include <cases/elasticity/tendon.h>
#include <cases/elasticity/tube_3d.h>
#include <cases/elasticity/turek_hron.h>
#include <cases/heat_transfer/partitioned_heat.h>

namespace TestCases
{
  template <int dim>
  class CaseSelector
  {
  public:
    /**
     * @brief get_test_case Returns the specified test case
     * @param testcase_name Name of the test case as specified in the
     *        configuration file
     * @param simulation_type Simulation type: elasticity vs heat_transfer
     *
     * @return object describing the test case
     */
    static std::shared_ptr<TestCaseBase<dim>>
    get_test_case(const std::string &testcase_name,
                  const std::string &simulation_type)
    {
      Assert(simulation_type == "elasticity" ||
               simulation_type == "heat_transfer",
             ExcNotImplemented());
      if (simulation_type == "elasticity")
        {
          if (testcase_name == "turek_hron")
            return std::make_shared<TurekHron<dim>>();
          else if (testcase_name == "cook")
            return std::make_shared<CookMembrane<dim>>();
          else if (testcase_name == "tube3d")
            return std::make_shared<Tube3D<dim>>();
          else if (testcase_name == "bending_flap")
            return std::make_shared<BendingFlap<dim>>();
          else if (testcase_name == "Wall_beam")
            return std::make_shared<WallBeam<dim>>();
          else if (testcase_name == "perpendicular_flap")
            return std::make_shared<PerpendicularFlap<dim>>();
          else if (testcase_name == "tendon")
            return std::make_shared<Tendon<dim>>();
          // Add your case here
        }
      if (simulation_type == "heat_transfer")
        {
          if (testcase_name == "partitioned_heat_dirichlet")
            return std::make_shared<PartitionedHeat<dim>>(true);
          else if (testcase_name == "partitioned_heat_neumann")
            return std::make_shared<PartitionedHeat<dim>>(false);
        }
      AssertThrow(
        false,
        ExcMessage(
          "Unable to configure your case \"" + testcase_name +
          "\". Make sure you use the right executable for the selected case, "
          "namely the \"solid\" executable for FSI cases and the \"heat\" "
          "executable for heat transfer simulation."));
    }
  };
} // namespace TestCases
