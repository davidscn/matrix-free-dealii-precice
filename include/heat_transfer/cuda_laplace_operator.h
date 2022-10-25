#pragma once

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

#include <base/fe_integrator.h>
namespace Heat_Transfer
{
  using namespace dealii;

  template <int dim>
  class Coefficient;


  template <int dim, typename number, typename MemorySpace = MemorySpace::Host>
  class LaplaceOperator
    : public MatrixFreeOperators::
        Base<dim, LinearAlgebra::distributed::Vector<number, MemorySpace>>
  {
  public:
    using FECellIntegrator =
      FECellIntegrators<dim, 1, number, VectorizedArray<number>>;
    using FEFaceIntegrator =
      FEFaceIntegrators<dim, 1, number, VectorizedArray<number>>;
    using VectorType = LinearAlgebra::distributed::Vector<number, MemorySpace>;

    LaplaceOperator();

    void
    clear() override;

    void
    evaluate_coefficient(const Coefficient<dim> &coefficient_function);

    void
    set_delta_t(const double delta_t_)
    {
      delta_t = delta_t_;
    }

    virtual void
    compute_diagonal() override;

  private:
    virtual void
    apply_add(VectorType &dst, const VectorType &src) const override;

    void
    local_apply(const MatrixFree<dim, number> &              data,
                VectorType &                                 dst,
                const VectorType &                           src,
                const std::pair<unsigned int, unsigned int> &cell_range) const;

    void
    do_operation_on_cell(FECellIntegrator &phi) const;

    Table<2, VectorizedArray<number>> coefficient;
    double                            delta_t = 0;
  };



  template <int dim, typename number, typename MemorySpace>
  LaplaceOperator<dim, number, MemorySpace>::LaplaceOperator()
    : MatrixFreeOperators::Base<dim, VectorType>()
  {}



  template <int dim, typename number, typename MemorySpace>
  void
  LaplaceOperator<dim, number, MemorySpace>::clear()
  {
    coefficient.reinit(0, 0);
    MatrixFreeOperators::Base<dim, VectorType>::clear();
  }



  template <int dim, typename number, typename MemorySpace>
  void
  LaplaceOperator<dim, number, MemorySpace>::evaluate_coefficient(
    const Coefficient<dim> &coefficient_function)
  {
    const unsigned int n_cells = this->data->n_cell_batches();
    FECellIntegrator   phi(*this->data);

    coefficient.reinit(n_cells, phi.n_q_points);
    for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
        phi.reinit(cell);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          coefficient(cell, q) =
            coefficient_function.value(phi.quadrature_point(q), 0);
      }
  }



  template <int dim, typename number, typename MemorySpace>
  void
  LaplaceOperator<dim, number, MemorySpace>::local_apply(
    const MatrixFree<dim, number> &              data,
    VectorType &                                 dst,
    const VectorType &                           src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FECellIntegrator phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        AssertDimension(coefficient.size(0), data.n_cell_batches());
        AssertDimension(coefficient.size(1), phi.n_q_points);

        phi.reinit(cell);
        phi.read_dof_values(src);
        do_operation_on_cell(phi);
        phi.distribute_local_to_global(dst);
      }
  }



  template <int dim, typename number, typename MemorySpace>
  void
  LaplaceOperator<dim, number, MemorySpace>::apply_add(
    VectorType &      dst,
    const VectorType &src) const
  {
    this->data->cell_loop(&LaplaceOperator::local_apply, this, dst, src);
  }



  template <int dim, typename number, typename MemorySpace>
  void
  LaplaceOperator<dim, number, MemorySpace>::compute_diagonal()
  {
    this->inverse_diagonal_entries.reset(new DiagonalMatrix<VectorType>());
    VectorType &inverse_diagonal = this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);

    MatrixFreeTools::compute_diagonal(*(this->data),
                                      inverse_diagonal,
                                      &LaplaceOperator::do_operation_on_cell,
                                      this);

    this->set_constrained_entries_to_one(inverse_diagonal);

    for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
      {
        Assert(inverse_diagonal.local_element(i) > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        inverse_diagonal.local_element(i) =
          1. / inverse_diagonal.local_element(i);
      }
  }



  template <int dim, typename number, typename MemorySpace>
  void
  LaplaceOperator<dim, number, MemorySpace>::do_operation_on_cell(
    FECellIntegrator &phi) const
  {
    Assert(delta_t > 0, ExcNotInitialized());
    const unsigned int cell = phi.get_current_cell_index();
    phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
    for (unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        phi.submit_value(phi.get_value(q), q);
        phi.submit_gradient(coefficient(cell, q) * delta_t *
                              phi.get_gradient(q),
                            q);
      }
    phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }

  // Helper function in order to evaluate the vectorized point
  template <int dim, typename Number>
  VectorizedArray<Number>
  evaluate_function(const Function<dim> &                      function,
                    const Point<dim, VectorizedArray<Number>> &p_vectorized,
                    const unsigned int                         component = 0)
  {
    VectorizedArray<Number> result;
    for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
      {
        Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
          p[d] = p_vectorized[d][v];
        result[v] = function.value(p, component);
      }
    return result;
  }
} // namespace Heat_Transfer