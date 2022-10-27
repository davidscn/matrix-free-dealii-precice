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

  template <int dim, int fe_degree, typename number>
  class VaryingCoefficientFunctor
  {
  public:
    VaryingCoefficientFunctor(double *coefficient)
      : coef(coefficient)
    {}

    __device__ void
    operator()(
      const unsigned int                                          cell,
      const typename CUDAWrappers::MatrixFree<dim, number>::Data *gpu_data);


    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = ::Utilities::pow(n_dofs_1d, dim);
    static const unsigned int n_q_points   = ::Utilities::pow(n_dofs_1d, dim);

  private:
    double *coef;
  };


  template <int dim, int fe_degree, typename number>
  __device__ void
  VaryingCoefficientFunctor<dim, fe_degree, number>::operator()(
    const unsigned int                                          cell,
    const typename CUDAWrappers::MatrixFree<dim, number>::Data *gpu_data)
  {
    const unsigned int pos = CUDAWrappers::local_q_point_id<dim, number>(
      cell, gpu_data, n_dofs_1d, n_q_points);

    coef[pos] = 1.;
  }

  template <int dim, int fe_degree, typename number>
  class LaplaceOperatorQuad
  {
  public:
    using FECellIntegrators =
      CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number>;
    __device__
    LaplaceOperatorQuad(double coef, double delta_t)
      : coef(coef)
      , delta_t(delta_t)
    {}

    __device__ void
    operator()(FECellIntegrators *fe_eval) const;

  private:
    double coef;
    // TODO: Maybe remove from this class
    double delta_t;
  };



  template <int dim, int fe_degree, typename number>
  __device__ void
  LaplaceOperatorQuad<dim, fe_degree, number>::operator()(
    FECellIntegrators *fe_eval) const
  {
    fe_eval->submit_value(fe_eval->get_value());
    fe_eval->submit_gradient(coef * fe_eval->get_gradient() * delta_t);
  }

  template <int dim, int fe_degree, typename number>
  class LocalLaplaceOperator
  {
  public:
    using FECellIntegrators =
      CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number>;

    LocalLaplaceOperator(double *coefficient, double delta_t)
      : coef(coefficient)
      , delta_t(delta_t)
    {}

    __device__ void
    operator()(
      const unsigned int                                          cell,
      const typename CUDAWrappers::MatrixFree<dim, number>::Data *gpu_data,
      CUDAWrappers::SharedData<dim, number> *                     shared_data,
      const double *                                              src,
      double *                                                    dst) const;

    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
    static const unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

  private:
    double *coef;
    double  delta_t;
  };


  template <int dim, int fe_degree, typename number>
  __device__ void
  LocalLaplaceOperator<dim, fe_degree, number>::operator()(
    const unsigned int                                          cell,
    const typename CUDAWrappers::MatrixFree<dim, number>::Data *gpu_data,
    CUDAWrappers::SharedData<dim, number> *                     shared_data,
    const double *                                              src,
    double *                                                    dst) const
  {
    const unsigned int pos = CUDAWrappers::local_q_point_id<dim, number>(
      cell, gpu_data, n_dofs_1d, n_q_points);

    FECellIntegrators fe_eval(cell, gpu_data, shared_data);

    fe_eval.read_dof_values(src);
    fe_eval.evaluate(true, true);
    fe_eval.apply_for_each_quad_point(
      LaplaceOperatorQuad<dim, fe_degree, number>(coef[pos], delta_t));
    fe_eval.integrate(true, true);

    fe_eval.distribute_local_to_global(dst);
  }

  template <int dim, int fe_degree, typename number>
  class CUDALaplaceOperator
  {
  public:
    using VectorType =
      LinearAlgebra::distributed::Vector<number, MemorySpace::CUDA>;

    CUDALaplaceOperator() = default;

    // and initialize the coefficient
    void
    initialize(std::shared_ptr<CUDAWrappers::MatrixFree<dim, number>> data);

    void
    evaluate_coefficient();

    void
    set_delta_t(double dt);

    void
    vmult(VectorType &dst, const VectorType &src) const;

    void
    initialize_dof_vector(VectorType &vec) const;

    void
    clear();

  private:
    std::shared_ptr<CUDAWrappers::MatrixFree<dim, number>> mf_data;
    LinearAlgebra::CUDAWrappers::Vector<number>            coef;
    double                                                 delta_t;
  };



  template <int dim, int fe_degree, typename number>
  void
  CUDALaplaceOperator<dim, fe_degree, number>::initialize(
    std::shared_ptr<CUDAWrappers::MatrixFree<dim, number>> data)
  {
    mf_data = data;

    AssertThrow(
      fe_degree == mf_data->get_dof_handler().get_fe().degree,
      ExcMessage(
        "The configured degree was not compiled into the cuda laplace operator."));
  }



  template <int dim, int fe_degree, typename number>
  void
  CUDALaplaceOperator<dim, fe_degree, number>::evaluate_coefficient()
  {
    const unsigned int n_owned_cells =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(
        &(mf_data->get_dof_handler().get_triangulation()))
        ->n_locally_owned_active_cells();
    coef.reinit(Utilities::pow(fe_degree + 1, dim) * n_owned_cells);

    const VaryingCoefficientFunctor<dim, fe_degree, number> functor(
      coef.get_values());
    mf_data->evaluate_coefficients(functor);
  }

  template <int dim, int fe_degree, typename number>
  void
  CUDALaplaceOperator<dim, fe_degree, number>::set_delta_t(double dt)
  {
    delta_t = dt;
  }


  template <int dim, int fe_degree, typename number>
  void
  CUDALaplaceOperator<dim, fe_degree, number>::vmult(
    VectorType &      dst,
    const VectorType &src) const
  {
    dst = 0.;
    LocalLaplaceOperator<dim, fe_degree, number> local_operator(
      coef.get_values(), delta_t);
    mf_data->cell_loop(local_operator, src, dst);
    // We handle here only homogeneous constraints, so the copy here can
    // probably ne omitted
    mf_data->copy_constrained_values(src, dst);
  }


  template <int dim, int fe_degree, typename number>
  void
  CUDALaplaceOperator<dim, fe_degree, number>::initialize_dof_vector(
    VectorType &vec) const
  {
    mf_data->initialize_dof_vector(vec);
  }


  template <int dim, int fe_degree, typename number>
  void
  CUDALaplaceOperator<dim, fe_degree, number>::clear()
  {
    coef = 0;
    mf_data.reset();
  }
} // namespace Heat_Transfer

#endif