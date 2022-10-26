#pragma once

#ifdef DEAL_II_COMPILER_CUDA_AWARE

#  include <deal.II/base/memory_space.h>
#  include <deal.II/base/table.h>
#  include <deal.II/base/utilities.h>

#  include <deal.II/lac/la_parallel_vector.h>

#  include <deal.II/matrix_free/cuda_fe_evaluation.h>
#  include <deal.II/matrix_free/cuda_matrix_free.h>
#  include <deal.II/matrix_free/fe_evaluation.h>
#  include <deal.II/matrix_free/matrix_free.h>
#  include <deal.II/matrix_free/operators.h>
#  include <deal.II/matrix_free/tools.h>

#  include <base/fe_integrator.h>
#  include <heat_transfer/laplace_operator.h>

namespace Heat_Transfer
{
  using namespace dealii;

  template <int dim>
  class Coefficient;


  template <int dim, int fe_degree>
  class VaryingCoefficientFunctor
  {
  public:
    VaryingCoefficientFunctor(double *coefficient)
      : coef(coefficient)
    {}

    __device__ void
    operator()(
      const unsigned int                                          cell,
      const typename CUDAWrappers::MatrixFree<dim, double>::Data *gpu_data);


    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = ::Utilities::pow(n_dofs_1d, dim);
    static const unsigned int n_q_points   = ::Utilities::pow(n_dofs_1d, dim);

  private:
    double *coef;
  };


  template <int dim, int fe_degree>
  __device__ void
  VaryingCoefficientFunctor<dim, fe_degree>::operator()(
    const unsigned int                                          cell,
    const typename CUDAWrappers::MatrixFree<dim, double>::Data *gpu_data)
  {
    const unsigned int pos = CUDAWrappers::local_q_point_id<dim, double>(
      cell, gpu_data, n_dofs_1d, n_q_points);

    coef[pos] = 1.;
  }

  template <int dim, int fe_degree>
  class LaplaceOperatorQuad
  {
  public:
    __device__
    LaplaceOperatorQuad(double coef, double delta_t)
      : coef(coef)
      , delta_t(delta_t)
    {}

    __device__ void
    operator()(
      CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double>
        *fe_eval) const;

  private:
    double coef;
    // TODO: Maybe remove from this class
    double delta_t;
  };



  template <int dim, int fe_degree>
  __device__ void
  LaplaceOperatorQuad<dim, fe_degree>::operator()(
    CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double>
      *fe_eval) const
  {
    fe_eval->submit_value(fe_eval->get_value());
    fe_eval->submit_gradient(coef * fe_eval->get_gradient() * delta_t);
  }

  template <int dim, int fe_degree>
  class LocalLaplaceOperator
  {
  public:
    LocalLaplaceOperator(double *coefficient, double delta_t)
      : coef(coefficient)
      , delta_t(delta_t)
    {}

    __device__ void
    operator()(
      const unsigned int                                          cell,
      const typename CUDAWrappers::MatrixFree<dim, double>::Data *gpu_data,
      CUDAWrappers::SharedData<dim, double>                      *shared_data,
      const double                                               *src,
      double                                                     *dst) const;

    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
    static const unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

  private:
    double *coef;
    double  delta_t;
  };


  template <int dim, int fe_degree>
  __device__ void
  LocalLaplaceOperator<dim, fe_degree>::operator()(
    const unsigned int                                          cell,
    const typename CUDAWrappers::MatrixFree<dim, double>::Data *gpu_data,
    CUDAWrappers::SharedData<dim, double>                      *shared_data,
    const double                                               *src,
    double                                                     *dst) const
  {
    const unsigned int pos = CUDAWrappers::local_q_point_id<dim, double>(
      cell, gpu_data, n_dofs_1d, n_q_points);

    CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double>
      fe_eval(cell, gpu_data, shared_data);

    fe_eval.read_dof_values(src);
    fe_eval.evaluate(true, true);
    fe_eval.apply_for_each_quad_point(
      LaplaceOperatorQuad<dim, fe_degree>(coef[pos], delta_t));
    fe_eval.integrate(true, true);

    fe_eval.distribute_local_to_global(dst);
  }

  template <int dim, typename number>
  class CUDALaplaceOperator
  {
  public:
    using VectorType =
      LinearAlgebra::distributed::Vector<number, MemorySpace::CUDA>;

    CUDALaplaceOperator();

    // and initialize the coefficient
    void
    initialize(const DoFHandler<dim>     &dof_handler,
               AffineConstraints<double> &constraints);

    void
    set_delta_t(double dt);

    void
    vmult(VectorType &dst, const VectorType &src) const;

    void
    initialize_dof_vector(VectorType &vec) const;

  private:
    CUDAWrappers::MatrixFree<dim, number>       mf_data;
    LinearAlgebra::CUDAWrappers::Vector<number> coef;
    double                                      delta_t;
  };



  template <int dim, typename number>
  CUDALaplaceOperator<dim, number>::CUDALaplaceOperator()
  {}



  template <int dim, typename number>
  void
  CUDALaplaceOperator<dim, number>::initialize(
    const DoFHandler<dim>     &dof_handler,
    AffineConstraints<double> &constraints)
  {
    const int     fe_degree = 1;
    MappingQ<dim> mapping(fe_degree);
    typename CUDAWrappers::MatrixFree<dim, number>::AdditionalData
      additional_data;

    additional_data.mapping_update_flags =
      (update_values | update_JxW_values | update_gradients |
       update_normal_vectors | update_quadrature_points);

    const QGauss<1> quad(fe_degree + 1);
    mf_data.reinit(mapping, dof_handler, constraints, quad, additional_data);


    const unsigned int n_owned_cells =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(
        &dof_handler.get_triangulation())
        ->n_locally_owned_active_cells();
    coef.reinit(Utilities::pow(fe_degree + 1, dim) * n_owned_cells);

    const VaryingCoefficientFunctor<dim, fe_degree> functor(coef.get_values());
    mf_data.evaluate_coefficients(functor);
  }


  template <int dim, typename number>
  void
  CUDALaplaceOperator<dim, number>::set_delta_t(double dt)
  {
    delta_t = dt;
  }


  template <int dim, typename number>
  void
  CUDALaplaceOperator<dim, number>::vmult(VectorType       &dst,
                                          const VectorType &src) const
  {
    dst                                            = 0.;
    const int                            fe_degree = 1;
    LocalLaplaceOperator<dim, fe_degree> local_operator(coef.get_values(),
                                                        delta_t);
    mf_data.cell_loop(local_operator, src, dst);
    // We handle here only homogeneous constraints, so the copy here can
    // probably ne omitted
    mf_data.copy_constrained_values(src, dst);
  }


  template <int dim, typename number>
  void
  CUDALaplaceOperator<dim, number>::initialize_dof_vector(VectorType &vec) const
  {
    mf_data.initialize_dof_vector(vec);
  }


  /**
   * Partial template specialization for CUDA
   */
  // template <int dim, typename number>
  // class LaplaceOperator<dim, number, MemorySpace::CUDA>
  //   : public MatrixFreeOperators::
  //       Base<dim, LinearAlgebra::distributed::Vector<number,
  //       MemorySpace::CUDA>>
  // {
  // public:
  //   using FECellIntegrator =
  //     FECellIntegrators<dim, 1, number, VectorizedArray<number>>;
  //   using FEFaceIntegrator =
  //     FEFaceIntegrators<dim, 1, number, VectorizedArray<number>>;
  //   using VectorType =
  //     LinearAlgebra::distributed::Vector<number, MemorySpace::CUDA>;

  //   LaplaceOperator();

  //   void
  //   clear() override;

  //   void
  //   evaluate_coefficient(const Coefficient<dim> &coefficient_function);

  //   void
  //   set_delta_t(const double delta_t_)
  //   {
  //     delta_t = delta_t_;
  //   }

  //   virtual void
  //   compute_diagonal() override;

  // private:
  //   virtual void
  //   apply_add(VectorType &dst, const VectorType &src) const override;

  //   void
  //   local_apply(const MatrixFree<dim, number>               &data,
  //               VectorType                                  &dst,
  //               const VectorType                            &src,
  //               const std::pair<unsigned int, unsigned int> &cell_range)
  //               const;

  //   void
  //   do_operation_on_cell(FECellIntegrator &phi) const;

  //   Table<2, VectorizedArray<number>>           coefficient;
  //   LinearAlgebra::CUDAWrappers::Vector<double> coef;
  //   double                                      delta_t = 0;
  // };



  // template <int dim, typename number>
  // LaplaceOperator<dim, number, MemorySpace::CUDA>::LaplaceOperator()
  //   : MatrixFreeOperators::Base<dim, VectorType>()
  // {}



  // template <int dim, typename number>
  // void
  // LaplaceOperator<dim, number, MemorySpace::CUDA>::clear()
  // {
  //   coefficient.reinit(0, 0);
  //   MatrixFreeOperators::Base<dim, VectorType>::clear();
  // }



  // template <int dim, typename number>
  // void
  // LaplaceOperator<dim, number, MemorySpace::CUDA>::evaluate_coefficient(
  //   const Coefficient<dim> &coefficient_function)
  // {
  //   const unsigned int n_cells = this->data->n_cell_batches();
  //   FECellIntegrator   phi(*this->data);

  //   coefficient.reinit(n_cells, phi.n_q_points);
  //   for (unsigned int cell = 0; cell < n_cells; ++cell)
  //     {
  //       phi.reinit(cell);
  //       for (unsigned int q = 0; q < phi.n_q_points; ++q)
  //         coefficient(cell, q) =
  //           coefficient_function.value(phi.quadrature_point(q), 0);
  //     }
  // }



  // template <int dim, typename number>
  // void
  // LaplaceOperator<dim, number, MemorySpace::CUDA>::local_apply(
  //   const MatrixFree<dim, number>               &data,
  //   VectorType                                  &dst,
  //   const VectorType                            &src,
  //   const std::pair<unsigned int, unsigned int> &cell_range) const
  // {
  //   FECellIntegrator phi(data);

  //   for (unsigned int cell = cell_range.first; cell < cell_range.second;
  //   ++cell)
  //     {
  //       AssertDimension(coefficient.size(0), data.n_cell_batches());
  //       AssertDimension(coefficient.size(1), phi.n_q_points);

  //       phi.reinit(cell);
  //       phi.read_dof_values(src);
  //       do_operation_on_cell(phi);
  //       phi.distribute_local_to_global(dst);
  //     }
  // }



  // template <int dim, typename number>
  // void
  // LaplaceOperator<dim, number, MemorySpace::CUDA>::apply_add(
  //   VectorType       &dst,
  //   const VectorType &src) const
  // {
  //   this->data->cell_loop(&LaplaceOperator::local_apply, this, dst, src);
  // }



  // template <int dim, typename number>
  // void
  // LaplaceOperator<dim, number, MemorySpace::CUDA>::compute_diagonal()
  // {
  //   this->inverse_diagonal_entries.reset(new DiagonalMatrix<VectorType>());
  //   VectorType &inverse_diagonal =
  //   this->inverse_diagonal_entries->get_vector();
  //   this->data->initialize_dof_vector(inverse_diagonal);

  //   MatrixFreeTools::compute_diagonal(*(this->data),
  //                                     inverse_diagonal,
  //                                     &LaplaceOperator::do_operation_on_cell,
  //                                     this);

  //   this->set_constrained_entries_to_one(inverse_diagonal);

  //   for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
  //     {
  //       Assert(inverse_diagonal.local_element(i) > 0.,
  //              ExcMessage("No diagonal entry in a positive definite operator
  //              "
  //                         "should be zero"));
  //       inverse_diagonal.local_element(i) =
  //         1. / inverse_diagonal.local_element(i);
  //     }
  // }



  // template <int dim, typename number>
  // void
  // LaplaceOperator<dim, number, MemorySpace::CUDA>::do_operation_on_cell(
  //   FECellIntegrator &phi) const
  // {
  //   Assert(delta_t > 0, ExcNotInitialized());
  //   const unsigned int cell = phi.get_current_cell_index();
  //   phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
  //   for (unsigned int q = 0; q < phi.n_q_points; ++q)
  //     {
  //       phi.submit_value(phi.get_value(q), q);
  //       phi.submit_gradient(coefficient(cell, q) * delta_t *
  //                             phi.get_gradient(q),
  //                           q);
  //     }
  //   phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  // }
} // namespace Heat_Transfer

#endif