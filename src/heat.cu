#include <deal.II/base/cuda.h>

#include <heat_transfer/heat_transfer.h>

// By default, all the MPI ranks
// will try to access the device with number 0, which we assume to be
// the GPU device associated with the CPU on which a particular MPI
// rank runs. This works, but if we are running with MPI support it
// may be that multiple MPI processes are running on the same machine
// (for example, one per CPU core) and then they would all want to
// access the same GPU on that machine. If there is only one GPU in
// the machine, there is nothing we can do about it: All MPI ranks on
// that machine need to share it. But if there are more than one GPU,
// then it is better to address different graphic cards for different
// processes. The choice below is based on the MPI process id by
// assigning GPUs round robin to GPU ranks. (To work correctly, this
// scheme assumes that the MPI ranks on one machine are
// consecutive. If that were not the case, then the rank-GPU
// association may just not be optimal.) To make this work, MPI needs
// to be initialized before using this function.

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


      int         n_devices       = 0;
      cudaError_t cuda_error_code = cudaGetDeviceCount(&n_devices);
      AssertCuda(cuda_error_code);
      const unsigned int my_mpi_id =
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      const int device_id = my_mpi_id % n_devices;
      cuda_error_code     = cudaSetDevice(device_id);
      AssertCuda(cuda_error_code);

      if (degree == 0)
        AssertThrow(degree > 0, ExcInternalError());

      if (dim == 2)
        {
          // query the testcase
          TestCases::CaseSelector<2> selector;
          auto testcase = selector.get_heat_transfer_test_case(case_name);
          Parameters::HeatParameters<2> parameters(parameter_filename);
          LaplaceProblem<2, true>       laplace_problem(parameters);
          laplace_problem.run(testcase);
        }
      else if (dim == 3)
        {
          // query the testcase
          TestCases::CaseSelector<3> selector;
          auto testcase = selector.get_heat_transfer_test_case(case_name);
          Parameters::HeatParameters<3> parameters(parameter_filename);
          LaplaceProblem<3, true>       laplace_problem(parameters);
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
