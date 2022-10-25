#include <cases/case_selector.h>
#include <solid_mechanics/mf_elasticity.h>

// @sect3{Main function}
// Lastly we provide the main driver function which appears
// no different to the other tutorials.
int
main(int argc, char *argv[])
{
  using namespace dealii;
  using namespace FSI;

  try
    {
      deallog.depth_console(0);
      const std::string parameter_filename =
        argc > 1 ? argv[1] : "parameters.prm";

      ParameterHandler prm;

      Parameters::Geometry geometry;
      geometry.add_parameters(prm);

      Parameters::FESystem fesystem;
      fesystem.add_parameters(prm);

      prm.parse_input(parameter_filename, "", true);

      // Disable multi-threading
      Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, 1 /*dealii::numbers::invalid_unsigned_int*/);

      const unsigned int degree    = fesystem.poly_degree;
      const unsigned int dim       = geometry.dimension;
      const std::string  case_name = geometry.testcase;

      if (degree == 0)
        AssertThrow(degree > 0, ExcInternalError());

      if (dim == 2)
        {
          // query the testcase
          TestCases::CaseSelector<2> selector;
          auto testcase = selector.get_elasticity_test_case(case_name);
          Parameters::FSIParameters<2> parameters(parameter_filename);
          Solid<2, double>             solid_2d(parameters);
          solid_2d.run(testcase);
        }
      else if (dim == 3)
        {
          // query the testcase
          TestCases::CaseSelector<3> selector;
          auto testcase = selector.get_elasticity_test_case(case_name);
          Parameters::FSIParameters<3> parameters(parameter_filename);
          Solid<3, double>             solid_3d(parameters);
          solid_3d.run(testcase);
        }
      else
        {
          AssertThrow(false,
                      ExcMessage("The given dimension is not supported."));
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
