#pragma once

#include <cases/case_base.h>
#include <cases/elasticity/Wall_beam.h>
#include <cases/elasticity/bending_flap.h>
#include <cases/elasticity/cook_membrane.h>
#include <cases/elasticity/perpendicular_flap.h>
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
      // Add your solid case to the list here
      static const std::map<std::string,
                            std::function<std::shared_ptr<TestCaseBase<dim>>()>>
        elasticity_cases{
          {"turek_hron", [] { return std::make_shared<TurekHron<dim>>(); }},
          {"cook", [] { return std::make_shared<CookMembrane<dim>>(); }},
          {"tube3d", [] { return std::make_shared<Tube3D<dim>>(); }},
          {"bending_flap", [] { return std::make_shared<BendingFlap<dim>>(); }},
          {"Wall_beam", [] { return std::make_shared<WallBeam<dim>>(); }},
          {"perpendicular_flap",
           [] { return std::make_shared<PerpendicularFlap<dim>>(); }}};

      // Add your heat case to the list here
      static const std::map<std::string,
                            std::function<std::shared_ptr<TestCaseBase<dim>>()>>
        heat_transfer_cases{
          {"partitioned_heat_dirichlet",
           [] { return std::make_shared<PartitionedHeat<dim>>(true); }},
          {"partitioned_heat_neumann",
           [] { return std::make_shared<PartitionedHeat<dim>>(false); }}};

      // Ensure the simulation type is valid
      AssertThrow(simulation_type == "elasticity" ||
                    simulation_type == "heat_transfer",
                  ExcNotImplemented());

      // Choose the appropriate map based on the simulation type
      const auto &cases = simulation_type == "elasticity" ? elasticity_cases :
                                                            heat_transfer_cases;

      // Attempt to find the test case in the map
      auto it = cases.find(testcase_name);
      if (it != cases.end())
        {
          return it->second(); // Execute the corresponding factory function
        }

      // Prepare a list of valid case names for the error message
      std::string available_options;
      for (const auto &pair : cases)
        {
          if (!available_options.empty())
            available_options += ", ";
          available_options += "\"" + pair.first + "\"";
        }

      // If the test case was not found, throw an exception with the available
      // options
      AssertThrow(
        false,
        ExcMessage(
          "Unable to configure your case \"" + testcase_name +
          "\". Make sure you use the right executable for the selected case, "
          "namely the \"solid\" executable for FSI cases and the \"heat\" "
          "executable for heat transfer simulation. Available options are: " +
          available_options + "."));
    }
  };
} // namespace TestCases
