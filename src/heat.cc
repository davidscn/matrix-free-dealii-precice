#include <heat_transfer/heat_transfer.h>

int
main(int argc, char *argv[])
{
  using namespace dealii;
  using namespace Heat_Transfer;

  try
    {
      const std::string parameter_filename =
        argc > 1 ? argv[1] : "parameters.prm";

      ParameterHandler prm;

      FSI::Parameters::FESystem fesystem;
      fesystem.add_parameters(prm);
      FSI::Parameters::Geometry geometry;
      geometry.add_parameters(prm);

      prm.parse_input(parameter_filename, "", true);

      // Disable multi-threading
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      const unsigned int degree    = fesystem.poly_degree;
      const unsigned int dim       = geometry.dim;
      const std::string  case_name = geometry.testcase;


      if (degree == 0)
        AssertThrow(degree > 0, ExcInternalError());

      if (dim == 2)
        {
          // query the testcase
          TestCases::CaseSelector<2> selector;
          auto testcase = selector.get_test_case(case_name, "heat_transfer");
          FSI::Parameters::AllParameters<2> parameters(parameter_filename);
          LaplaceProblem<2>                 laplace_problem(parameters);
          laplace_problem.run(testcase);
        }
      else if (dim == 3)
        {
          // query the testcase
          TestCases::CaseSelector<3> selector;
          auto testcase = selector.get_test_case(case_name, "heat_transfer");
          FSI::Parameters::AllParameters<3> parameters(parameter_filename);
          LaplaceProblem<3>                 laplace_problem(parameters);
          laplace_problem.run(testcase);
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
