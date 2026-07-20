#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <base/solver_runners.h>
#include <parameter/parameter_handling.h>

#include <iostream>
#include <string>

int
main(int argc, char *argv[])
{
  using namespace dealii;

  try
    {
      const std::string parameter_filename =
        argc > 1 ? argv[1] : "parameters.prm";

      ParameterHandler prm;

      Parameters::FESystem fesystem;
      fesystem.add_parameters(prm);
      Parameters::Geometry geometry;
      geometry.add_parameters(prm);

      prm.parse_input(parameter_filename, "", true);

      // Disable multi-threading
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      const unsigned int degree    = fesystem.poly_degree;
      const unsigned int dim       = geometry.dimension;
      const std::string  case_name = geometry.testcase;


      if (degree == 0)
        AssertThrow(degree > 0, ExcInternalError());

      if (dim == 2)
        {
          SolverRunners::run_heat_2d(parameter_filename, case_name);
        }
      else if (dim == 3)
        {
          SolverRunners::run_heat_3d(parameter_filename, case_name);
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
