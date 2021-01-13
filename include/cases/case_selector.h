#pragma once

#include <cases/case_base.h>
#include <cases/cook_membrane.h>
#include <cases/turek_hron.h>

namespace FSI
{
  namespace TestCases
  {
    template <int dim>
    class CaseSelector
    {
    public:
      static std::shared_ptr<TestCaseBase<dim>>
      get_test_case(const std::string testcase_name)
      {
        if (testcase_name == "turek_hron")
          return std::make_shared<TurekHron<dim>>();
        else if (testcase_name == "cook")
          return std::make_shared<CookMembrane<dim>>();
        // Add your case here
        else
          AssertThrow(false,
                      ExcMessage("Unable to configure your case " +
                                 testcase_name));
      }
    };
  } // namespace TestCases
} // namespace FSI
