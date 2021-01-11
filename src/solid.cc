// own headers
#include <boost/preprocessor/facilities/empty.hpp>
#include <boost/preprocessor/list/at.hpp>
#include <boost/preprocessor/list/for_each_product.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>

#include <mf_elasticity.h>

#define GET_D(L) BOOST_PP_TUPLE_ELEM(3, 0, BOOST_PP_TUPLE_ELEM(1, 0, L))
#define GET_Q(L) BOOST_PP_TUPLE_ELEM(3, 1, BOOST_PP_TUPLE_ELEM(1, 0, L))

//#define MF_DQ BOOST_PP_TUPLE_TO_LIST(8, ((1, 2), (2, 3), (3, 4), (4, 5),
//(5,6), (6,7), (7,8), (8,9)))
#define MF_DQ BOOST_PP_TUPLE_TO_LIST(1, ((1, 2)))

#define DOIF2(R, L)                                                      \
  else if ((degree == GET_D(L)) && (n_q_points == GET_Q(L)))             \
  {                                                                      \
    Parameters::AllParameters<2>         parameters(parameter_filename); \
    Solid<2, GET_D(L), GET_Q(L), double> solid_2d(parameters);           \
    solid_2d.run();                                                      \
  }


#define DOIF3(R, L)                                                      \
  else if ((degree == GET_D(L)) && (n_q_points == GET_Q(L)))             \
  {                                                                      \
    Parameters::AllParameters<3>         parameters(parameter_filename); \
    Solid<3, GET_D(L), GET_Q(L), double> solid_3d(parameters);           \
    solid_3d.run();                                                      \
  }


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

      {
        // Disable multi-threading
        Utilities::MPI::MPI_InitFinalize mpi_initialization(
          argc, argv, 1 /*dealii::numbers::invalid_unsigned_int*/);

        typedef double     NumberType;
        const unsigned int degree     = fesystem.poly_degree;
        const unsigned int n_q_points = fesystem.quad_order;
        const unsigned int dim        = geometry.dim;
        if (dim == 2)
          {
            if (degree == 0)
              {
                AssertThrow(false, ExcInternalError());
              }
            BOOST_PP_LIST_FOR_EACH_PRODUCT(DOIF2, 1, (MF_DQ))
            else
            {
              AssertThrow(false,
                          ExcMessage(
                            "Matrix-free calculations with degree=" +
                            std::to_string(degree) +
                            " and n_q_points_1d=" + std::to_string(n_q_points) +
                            " are not supported."));
            }
          }
        else if (dim == 3)
          {
            if (degree == 0)
              {
                AssertThrow(false, ExcInternalError());
              }
            BOOST_PP_LIST_FOR_EACH_PRODUCT(DOIF3, 1, (MF_DQ))
            else
            {
              AssertThrow(false,
                          ExcMessage(
                            "Matrix-free calculations with degree=" +
                            std::to_string(degree) +
                            " and n_q_points_1d=" + std::to_string(n_q_points) +
                            " are not supported."));
            }
          }
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
