#include <boost/preprocessor/facilities/empty.hpp>
#include <boost/preprocessor/list/at.hpp>
#include <boost/preprocessor/list/for_each_product.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>

#define GET_D(L) BOOST_PP_TUPLE_ELEM(3, 0, BOOST_PP_TUPLE_ELEM(1, 0, L))
#define GET_Q(L) BOOST_PP_TUPLE_ELEM(3, 1, BOOST_PP_TUPLE_ELEM(1, 0, L))

// Template parameter list for the polynomial degree and quadrature order
#define MF_DQ BOOST_PP_TUPLE_TO_LIST(4, ((1, 2), (2, 3), (3, 4), (4, 5)))
