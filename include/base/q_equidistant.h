#pragma once

#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>

using namespace dealii;

/**
 * Dummy quadrature formula used for value evaluation resulting in a well-posed
 * RBF system for the coupling. It cannot be used for actual quadrature.
 */
template <int dim>
class QEquidistant : public Quadrature<dim>
{
public:
  /**
   * Generate a formula with <tt>n</tt> quadrature points (in each space
   * direction).
   */
  QEquidistant(const unsigned int n);
};


template <>
QEquidistant<1>::QEquidistant(const unsigned int n)
  : Quadrature<1>(n)
{
  if (n == 0)
    return;

  // Compute distance between points
  const double point_distance = 1. / n;
  // Exclude boundaries
  double x_point = point_distance * 0.5;
  for (uint q = 0; q < n; ++q)
    {
      this->quadrature_points[q] = Point<1>(x_point);
      x_point += point_distance;
    }
}


template <int dim>
QEquidistant<dim>::QEquidistant(const unsigned int n)
  : Quadrature<dim>(QEquidistant<dim - 1>(n), QEquidistant<1>(n))
{}
