#pragma once

// deal.II
#include <deal.II/base/config.h>

#include <deal.II/matrix_free/fe_evaluation.h>

DEAL_II_NAMESPACE_OPEN

template <int dim,
          int n_components,
          typename Number,
          typename VectorizedArrayType>
using FECellIntegrators =
  FEEvaluation<dim, -1, 0, n_components, Number, VectorizedArrayType>;

template <int dim,
          int n_components,
          typename Number,
          typename VectorizedArrayType>
using FEFaceIntegrators =
  FEFaceEvaluation<dim, -1, 0, n_components, Number, VectorizedArrayType>;

DEAL_II_NAMESPACE_CLOSE
