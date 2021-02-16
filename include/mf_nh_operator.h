#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>

#include <material.h>

// Define an operation that takes two tensors $ \mathbf{A} $ and
// $ \mathbf{B} $ such that their outer-product
// $ \mathbf{A} \bar{\otimes} \mathbf{B} \Rightarrow C_{ijkl} = A_{il} B_{jk}
// $
template <int dim, typename Number>
Tensor<4, dim, Number>
outer_product_iljk(const Tensor<2, dim, Number> &A,
                   const Tensor<2, dim, Number> &B)
{
  Tensor<4, dim, Number> A_il_B_jk{};

  for (unsigned int i = 0; i < dim; ++i)
    {
      for (unsigned int j = 0; j < dim; ++j)
        {
          for (unsigned int k = 0; k < dim; ++k)
            {
              for (unsigned int l = 0; l < dim; ++l)
                {
                  A_il_B_jk[i][j][k][l] += A[i][l] * B[j][k];
                }
            }
        }
    }

  return A_il_B_jk;
}

// Define an operation that takes two tensors $ \mathbf{A} $ and
// $ \mathbf{B} $ such that their outer-product
// $ \mathbf{A} \bar{\otimes} \mathbf{B} \Rightarrow C_{ijkl} = A_{ik} B_{jl}
// $
template <int dim, typename Number>
Tensor<4, dim, Number>
outer_product_ikjl(const Tensor<2, dim, Number> &A,
                   const Tensor<2, dim, Number> &B)
{
  Tensor<4, dim, Number> A_ik_B_jl{};
  for (unsigned int i = 0; i < dim; ++i)
    {
      for (unsigned int j = 0; j < dim; ++j)
        {
          for (unsigned int k = 0; k < dim; ++k)
            {
              for (unsigned int l = 0; l < dim; ++l)
                {
                  A_ik_B_jl[i][j][k][l] += A[i][k] * B[j][l];
                }
            }
        }
    }

  return A_ik_B_jl;
}

using namespace dealii;

template <typename Number>
void
adjust_ghost_range_if_necessary(
  const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner,
  const LinearAlgebra::distributed::Vector<Number> &        vec)
{
  if (vec.get_partitioner().get() != partitioner.get())
    {
      LinearAlgebra::distributed::Vector<Number> copy(vec);
      const_cast<LinearAlgebra::distributed::Vector<Number> &>(vec).reinit(
        partitioner);
      const_cast<LinearAlgebra::distributed::Vector<Number> &>(vec)
        .copy_locally_owned_data_from(copy);
    }
}


/**
 * Large strain Neo-Hook tangent operator.
 *
 * Follow
 * https://github.com/dealii/dealii/blob/master/tests/matrix_free/step-37.cc
 */
template <int dim, int fe_degree, int n_q_points_1d, typename Number>
class NeoHookOperator : public Subscriptor
{
public:
  NeoHookOperator();

  using size_type =
    typename LinearAlgebra::distributed::Vector<Number>::size_type;
  using VectorType          = LinearAlgebra::distributed::Vector<Number>;
  using VectorizedArrayType = VectorizedArray<Number>;

  void
  clear();

  void
  initialize(std::shared_ptr<const MatrixFree<dim, Number>> data_current,
             std::shared_ptr<const MatrixFree<dim, Number>> data_reference,
             const VectorType &                             displacement,
             const std::string &                            caching);

  void
  set_material(
    std::shared_ptr<
      Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArrayType>>
      material,
    std::shared_ptr<
      Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArrayType>>
      material_inclusion);

  void
  compute_diagonal();

  unsigned int
  m() const;
  unsigned int
  n() const;

  void
  vmult(VectorType &dst, const VectorType &src) const;

  void
  Tvmult(VectorType &dst, const VectorType &src) const;

  void
  vmult_add(VectorType &dst, const VectorType &src) const;
  void
  Tvmult_add(VectorType &dst, const VectorType &src) const;

  Number
  el(const unsigned int row, const unsigned int col) const;

  void
  precondition_Jacobi(VectorType &      dst,
                      const VectorType &src,
                      const Number      omega) const;

  const std::shared_ptr<DiagonalMatrix<VectorType>>
  get_matrix_diagonal_inverse() const
  {
    return inverse_diagonal_entries;
  }

  /**
   * Cache a few things for the current displacement.
   */
  void
  cache();

  /**
   * Cache memory consumption by this class.
   */
  std::size_t
  memory_consumption() const;

private:
  /**
   * Apply operator on a range of cells.
   */
  void
  local_apply_cell(
    const MatrixFree<dim, Number> &              data,
    VectorType &                                 dst,
    const VectorType &                           src,
    const std::pair<unsigned int, unsigned int> &cell_range) const;

  /**
   * Apply diagonal part of the operator on a cell range.
   */
  void
  local_diagonal_cell(
    const MatrixFree<dim, Number> &data,
    VectorType &                   dst,
    const unsigned int &,
    const std::pair<unsigned int, unsigned int> &cell_range) const;

  /**
   * Perform operation on a cell. @p phi_current corresponds to the deformed configuration
   * where @p phi_reference is for the current configuration.
   */
  void
  do_operation_on_cell(
    FEEvaluation<dim, fe_degree, n_q_points_1d, dim, Number> &phi_current,
    FEEvaluation<dim, fe_degree, n_q_points_1d, dim, Number> &phi_reference,
    const unsigned int                                        cell) const;

  std::shared_ptr<const MatrixFree<dim, Number>> data_current;
  std::shared_ptr<const MatrixFree<dim, Number>> data_reference;

  const VectorType *displacement;

  std::shared_ptr<
    Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArrayType>>
    material;

  std::shared_ptr<
    Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArrayType>>
    material_inclusion;

  std::shared_ptr<DiagonalMatrix<VectorType>> inverse_diagonal_entries;
  std::shared_ptr<DiagonalMatrix<VectorType>> diagonal_entries;

  Table<2, VectorizedArrayType>                          cached_scalar;
  Table<2, VectorizedArrayType>                          cached_second_scalar;
  Table<2, Tensor<2, dim, VectorizedArrayType>>          cached_tensor2;
  Table<2, SymmetricTensor<4, dim, VectorizedArrayType>> cached_tensor4;
  Table<2, Tensor<4, dim, VectorizedArrayType>>          cached_tensor4_ns;

  bool diagonal_is_available;

  enum class MFCaching
  {
    none,
    scalar_referential,
    scalar,
    tensor2,
    tensor4,
    tensor4_ns
  };
  MFCaching mf_caching;

  Tensor<4, dim, VectorizedArrayType> IxI_ikjl;
};



template <int dim, int fe_degree, int n_q_points_1d, typename Number>
std::size_t
NeoHookOperator<dim, fe_degree, n_q_points_1d, Number>::memory_consumption()
  const
{
  auto res = cached_scalar.memory_consumption() +
             cached_second_scalar.memory_consumption() +
             cached_tensor2.memory_consumption() +
             cached_tensor4.memory_consumption() +
             cached_tensor4_ns.memory_consumption();

  // matrix-free data:
  if (mf_caching == MFCaching::tensor4_ns ||
      mf_caching == MFCaching::scalar_referential)
    {
      res += data_reference->memory_consumption();
    }
  else
    {
      res += data_current->memory_consumption() +
             (mf_caching == MFCaching::scalar ?
                data_reference->memory_consumption() :
                0);
    }

  // note: do not include diagonals, we want to measure only memory needed for
  // vmult for performance analysis.
  return res;
}



template <int dim, int fe_degree, int n_q_points_1d, typename Number>
NeoHookOperator<dim, fe_degree, n_q_points_1d, Number>::NeoHookOperator()
  : Subscriptor()
  , diagonal_is_available(false)
  , mf_caching(MFCaching::none)
{
  Tensor<2, dim, VectorizedArrayType> I;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      I[i][j] = make_vectorized_array<Number>(i == j);

  IxI_ikjl = outer_product_ikjl(I, I);
}



template <int dim, int fe_degree, int n_q_points_1d, typename Number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, Number>::precondition_Jacobi(
  VectorType &      dst,
  const VectorType &src,
  const Number      omega) const
{
  Assert(inverse_diagonal_entries.get() && inverse_diagonal_entries->m() > 0,
         ExcNotInitialized());
  inverse_diagonal_entries->vmult(dst, src);
  dst *= omega;
}



template <int dim, int fe_degree, int n_q_points_1d, typename Number>
unsigned int
NeoHookOperator<dim, fe_degree, n_q_points_1d, Number>::m() const
{
  return data_current->get_vector_partitioner()->size();
}



template <int dim, int fe_degree, int n_q_points_1d, typename Number>
unsigned int
NeoHookOperator<dim, fe_degree, n_q_points_1d, Number>::n() const
{
  return data_current->get_vector_partitioner()->size();
}



template <int dim, int fe_degree, int n_q_points_1d, typename Number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, Number>::clear()
{
  data_current.reset();
  data_reference.reset();
  diagonal_is_available = false;
  diagonal_entries.reset();
  inverse_diagonal_entries.reset();
}



template <int dim, int fe_degree, int n_q_points_1d, typename Number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, Number>::initialize(
  std::shared_ptr<const MatrixFree<dim, Number>> data_current_,
  std::shared_ptr<const MatrixFree<dim, Number>> data_reference_,
  const VectorType &                             displacement_,
  const std::string &                            caching)
{
  data_current   = data_current_;
  data_reference = data_reference_;
  displacement   = &displacement_;

  const unsigned int n_cells = data_reference_->n_cell_batches();
  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, Number> phi(
    *data_reference_);
  if (caching == "scalar")
    {
      mf_caching = MFCaching::scalar;
      cached_scalar.reinit(n_cells, phi.n_q_points);
    }
  else if (caching == "scalar_referential")
    {
      mf_caching = MFCaching::scalar_referential;
      cached_scalar.reinit(n_cells, phi.n_q_points);
      cached_second_scalar.reinit(n_cells, phi.n_q_points * dim);
    }
  else if (caching == "tensor2")
    {
      mf_caching = MFCaching::tensor2;
      cached_scalar.reinit(n_cells, phi.n_q_points);
      cached_second_scalar.reinit(n_cells, phi.n_q_points);
      cached_tensor2.reinit(n_cells, phi.n_q_points);
    }
  else if (caching == "tensor4")
    {
      mf_caching = MFCaching::tensor4;
      cached_tensor2.reinit(n_cells, phi.n_q_points);
      cached_tensor4.reinit(n_cells, phi.n_q_points);
      cached_second_scalar.reinit(n_cells, phi.n_q_points);
    }
  else if (caching == "tensor4_ns")
    {
      mf_caching = MFCaching::tensor4_ns;
      cached_tensor4_ns.reinit(n_cells, phi.n_q_points);
    }
  else
    {
      mf_caching = MFCaching::none;
      AssertThrow(false, ExcMessage("Unknown caching"));
    }
}



template <int dim, int fe_degree, int n_q_points_1d, typename Number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, Number>::cache()
{
  const unsigned int n_cells = data_reference->n_cell_batches();

  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, Number> phi_reference(
    *data_reference);

  for (unsigned int cell = 0; cell < n_cells; ++cell)
    {
      const unsigned int material_id =
        data_current->get_cell_iterator(cell, 0)->material_id();
      const auto &cell_mat = (material_id == 0 ? material : material_inclusion);

      Assert(cell_mat->formulation <= 1,
             ExcMessage("Unknown material formulation"));

      phi_reference.reinit(cell);
      phi_reference.read_dof_values_plain(*displacement);
      phi_reference.evaluate(mf_caching == MFCaching::scalar_referential,
                             true,
                             false);

      if (cell_mat->formulation == 0)
        {
          Assert(mf_caching == MFCaching::scalar, ExcInternalError());
          for (unsigned int q = 0; q < phi_reference.n_q_points; ++q)
            {
              const Tensor<2, dim, VectorizedArrayType> &grad_u =
                phi_reference.get_gradient(q);
              const Tensor<2, dim, VectorizedArrayType> F =
                Physics::Elasticity::Kinematics::F(grad_u);
              const VectorizedArrayType det_F = determinant(F);

              Assert(*std::min_element(
                       det_F.begin(),
                       det_F.begin() +
                         data_current->n_active_entries_per_cell_batch(cell)) >
                       0,
                     ExcMessage("det_F is not positive. "));

              cached_scalar(cell, q) = std::pow(det_F, Number(-1.0 / dim));
            }
        }
      else
        for (unsigned int q = 0; q < phi_reference.n_q_points; ++q)
          {
            const Tensor<2, dim, VectorizedArrayType> &grad_u =
              phi_reference.get_gradient(q);
            const Tensor<2, dim, VectorizedArrayType> F =
              Physics::Elasticity::Kinematics::F(grad_u);
            const VectorizedArrayType det_F = determinant(F);

            Assert(*std::min_element(
                     det_F.begin(),
                     det_F.begin() +
                       data_current->n_active_entries_per_cell_batch(cell)) > 0,
                   ExcMessage("det_F is not positive. "));

            const VectorizedArrayType scalar =
              cell_mat->mu - 2.0 * cell_mat->lambda * std::log(det_F);
            switch (mf_caching)
              {
                  case MFCaching::scalar: {
                    cached_scalar(cell, q) = scalar;
                    break;
                  }
                  case MFCaching::scalar_referential: {
                    cached_scalar(cell, q) = scalar;
                    // MK:
                    // This is to avoid the phi_reference.read_dof_values()
                    // call and the full phi_reference.evaluate(false, true)
                    // calls. With this quadrature point information, I only
                    // need to call a cheap "collocation gradient" function,
                    // which is likely the best compromise in terms of caching
                    // some data versus computing: read_dof_values is
                    // expensive because it is not fully vectorized (indirect
                    // addressing gather access) and once you start to store
                    // things element-by-element you can eliminate the
                    // interpolation from nodes to quadrature points (that
                    // happens in the usual matrix-free interpolations as
                    // well). So my code makes use of some internal workings
                    // of the matrix-free framework that one would need to
                    // clean up in case one really wanted to make
                    // user-friendly programs
                    for (unsigned int d = 0; d < dim; ++d)
                      cached_second_scalar(cell,
                                           q + d * phi_reference.n_q_points) =
                        phi_reference
                          .begin_values()[q + d * phi_reference.n_q_points];
                    break;
                  }
                  case MFCaching::tensor2: {
                    cached_scalar(cell, q) = scalar * 2. / det_F;
                    cached_second_scalar(cell, q) =
                      make_vectorized_array<Number>(1.) / det_F;

                    SymmetricTensor<2, dim, VectorizedArrayType> tau;
                    {
                      tau =
                        cell_mat->mu * Physics::Elasticity::Kinematics::b(F);
                      for (unsigned int d = 0; d < dim; ++d)
                        tau[d][d] -= scalar;
                    }
                    cached_tensor2(cell, q) = tau / det_F;
                    break;
                  }
                  case MFCaching::tensor4: {
                    SymmetricTensor<2, dim, VectorizedArrayType> tau;
                    {
                      tau =
                        cell_mat->mu * Physics::Elasticity::Kinematics::b(F);
                      for (unsigned int d = 0; d < dim; ++d)
                        tau[d][d] -= scalar;
                    }
                    cached_second_scalar(cell, q) =
                      make_vectorized_array<Number>(1.) / det_F;
                    cached_tensor2(cell, q) = tau / det_F;
                    cached_tensor4(cell, q) =
                      (scalar * 2. / det_F) *
                        Physics::Elasticity::StandardTensors<dim>::S +
                      (cell_mat->lambda * 2. / det_F) *
                        Physics::Elasticity::StandardTensors<dim>::IxI;
                    break;
                  }
                  case MFCaching::tensor4_ns: {
                    const Tensor<2, dim, VectorizedArrayType> F_inv = invert(F);
                    const Tensor<2, dim, VectorizedArrayType> F_inv_t(
                      transpose(F_inv));
                    const VectorizedArrayType ln_J = std::log(det_F);

                    const Tensor<4, dim, VectorizedArrayType>
                      F_inv_t_otimes_F_inv_t = outer_product(F_inv_t, F_inv_t);

                    const Tensor<4, dim, VectorizedArrayType> F_inv_t_F_inv =
                      outer_product_iljk(F_inv_t, F_inv);

                    cached_tensor4_ns(cell, q) =
                      (2. * cell_mat->lambda) * F_inv_t_otimes_F_inv_t +
                      (cell_mat->mu - 2.0 * cell_mat->lambda * ln_J) *
                        F_inv_t_F_inv +
                      cell_mat->mu * IxI_ikjl;
                    break;
                  }
                  case MFCaching::none: {
                    AssertThrow(false, ExcMessage("Unknown caching"));
                    break;
                  }
              }
          }
    }
}



template <int dim, int fe_degree, int n_q_points_1d, typename Number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, Number>::set_material(
  std::shared_ptr<
    Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArrayType>>
    material_,
  std::shared_ptr<
    Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArrayType>>
    material_inclusion_)
{
  material           = material_;
  material_inclusion = material_inclusion_;
}



template <int dim, int fe_degree, int n_q_points_1d, typename Number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, Number>::vmult(
  VectorType &      dst,
  const VectorType &src) const
{
  dst = 0;
  vmult_add(dst, src);
}



template <int dim, int fe_degree, int n_q_points_1d, typename Number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, Number>::Tvmult(
  VectorType &      dst,
  const VectorType &src) const
{
  dst = 0;
  vmult_add(dst, src);
}



template <int dim, int fe_degree, int n_q_points_1d, typename Number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, Number>::Tvmult_add(
  VectorType &      dst,
  const VectorType &src) const
{
  vmult_add(dst, src);
}



template <int dim, int fe_degree, int n_q_points_1d, typename Number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, Number>::vmult_add(
  VectorType &      dst,
  const VectorType &src) const
{
  const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner =
    data_current->get_vector_partitioner();

  Assert(partitioner->is_globally_compatible(
           *data_reference->get_vector_partitioner().get()),
         ExcMessage("Current and reference partitioners are incompatible"));

  adjust_ghost_range_if_necessary(partitioner, dst);
  adjust_ghost_range_if_necessary(partitioner, src);

  // FIXME: use cell_loop, should work even though we need
  // both matrix-free data objects.
  Assert(data_current->n_cell_batches() == data_reference->n_cell_batches(),
         ExcInternalError());

  // MatrixFree::cell_loop() is more complicated than a simple
  // update_ghost_values() / compress(), it loops on different cells (inner
  // without ghosts and outer) in different order and do update_ghost_values()
  // and compress_start()/compress_finish() in between.
  // https://www.dealii.org/developer/doxygen/deal.II/matrix__free_8h_source.html#l00109

  // 1. make sure ghosts are updated
  src.update_ghost_values();

  // 2. loop over all locally owned cell blocks
  local_apply_cell(*data_current,
                   dst,
                   src,
                   std::make_pair<unsigned int, unsigned int>(
                     0, data_current->n_cell_batches()));

  // 3. communicate results with MPI
  dst.compress(VectorOperation::add);

  // 4. constraints
  for (const auto dof : data_current->get_constrained_dofs())
    dst.local_element(dof) += src.local_element(dof);
}



template <int dim, int fe_degree, int n_q_points_1d, typename Number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, Number>::local_apply_cell(
  const MatrixFree<dim, Number> & /*data*/,
  VectorType &                                 dst,
  const VectorType &                           src,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, Number> phi_current(
    *data_current);
  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, Number> phi_reference(
    *data_reference);

  Assert(phi_current.n_q_points == phi_reference.n_q_points,
         ExcInternalError());

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // initialize on this cell

      // VMult reint read and write
      if (mf_caching == MFCaching::tensor4_ns ||
          mf_caching == MFCaching::scalar_referential)
        // fully referential
        {
          phi_reference.reinit(cell);
          phi_reference.read_dof_values(src);
        }
      else
        // gradients in deformed configuration
        {
          phi_current.reinit(cell);

          // read-in total displacement and src vector and evaluate
          // gradients
          phi_current.read_dof_values(src);
        }

      if (mf_caching == MFCaching::scalar)
        {
          phi_reference.reinit(cell);
          phi_reference.read_dof_values_plain(*displacement);
        }


      do_operation_on_cell(phi_current, phi_reference, cell);

      if (mf_caching == MFCaching::tensor4_ns ||
          mf_caching == MFCaching::scalar_referential)
        {
          phi_reference.distribute_local_to_global(dst);
        }
      else
        {
          phi_current.distribute_local_to_global(dst);
        }
    }
}



template <int dim, int fe_degree, int n_q_points_1d, typename Number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, Number>::local_diagonal_cell(
  const MatrixFree<dim, Number> & /*data*/,
  VectorType &dst,
  const unsigned int &,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  const VectorizedArrayType one  = make_vectorized_array<Number>(1.);
  const VectorizedArrayType zero = make_vectorized_array<Number>(0.);

  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, Number> phi_current(
    *data_current);
  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, Number> phi_reference(
    *data_reference);

  // keep fully referntial here and bail out if needed
  if (mf_caching == MFCaching::tensor4_ns ||
      mf_caching == MFCaching::scalar_referential)
    {
      for (unsigned int cell = cell_range.first; cell < cell_range.second;
           ++cell)
        {
          phi_reference.reinit(cell);
          AlignedVector<VectorizedArrayType> local_diagonal_vector(
            phi_reference.dofs_per_component * phi_reference.n_components);

          // Loop over all DoFs and set dof values to zero everywhere but i-th
          // DoF. With this input (instead of read_dof_values()) we do the
          // action and store the result in a diagonal vector
          for (unsigned int i = 0; i < phi_reference.dofs_per_component; ++i)
            for (unsigned int ic = 0; ic < phi_reference.n_components; ++ic)
              {
                for (unsigned int j = 0; j < phi_reference.dofs_per_component;
                     ++j)
                  for (unsigned int jc = 0; jc < phi_reference.n_components;
                       ++jc)
                    {
                      const auto ind_j =
                        j + jc * phi_reference.dofs_per_component;
                      phi_reference.begin_dof_values()[ind_j] = zero;
                    }

                const auto ind_i = i + ic * phi_reference.dofs_per_component;

                phi_reference.begin_dof_values()[ind_i] = one;

                do_operation_on_cell(phi_current, phi_reference, cell);

                local_diagonal_vector[ind_i] =
                  phi_reference.begin_dof_values()[ind_i];
              }

          // Finally, in order to distribute diagonal, write it again into one
          // of FEEvaluations and do the standard distribute_local_to_global.
          // Note that here non-diagonal matrix elements are ignored and so the
          // result is not equivalent to matrix-based case when hanging nodes
          // are present. see Section 5.3 in Korman 2016, A time-space adaptive
          // method for the Schrodinger equation,
          // doi: 10.4208/cicp.101214.021015a for a discussion.
          for (unsigned int i = 0; i < phi_reference.dofs_per_component; ++i)
            for (unsigned int ic = 0; ic < phi_reference.n_components; ++ic)
              {
                const auto ind_i = i + ic * phi_reference.dofs_per_component;
                phi_reference.begin_dof_values()[ind_i] =
                  local_diagonal_vector[ind_i];
              }

          phi_reference.distribute_local_to_global(dst);
        } // end of cell loop
      return;
    }

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // initialize on this cell
      phi_current.reinit(cell);
      phi_reference.reinit(cell);

      // read-in total displacement.
      phi_reference.read_dof_values_plain(*displacement);

      // FIXME: although we override DoFs manually later, somehow
      // we still need to read some dummy here
      phi_current.read_dof_values(*displacement);

      AlignedVector<VectorizedArrayType> local_diagonal_vector(
        phi_current.dofs_per_component * phi_current.n_components);

      // Loop over all DoFs and set dof values to zero everywhere but i-th DoF.
      // With this input (instead of read_dof_values()) we do the action and
      // store the result in a diagonal vector
      for (unsigned int i = 0; i < phi_current.dofs_per_component; ++i)
        for (unsigned int ic = 0; ic < phi_current.n_components; ++ic)
          {
            for (unsigned int j = 0; j < phi_current.dofs_per_component; ++j)
              for (unsigned int jc = 0; jc < phi_current.n_components; ++jc)
                {
                  const auto ind_j = j + jc * phi_current.dofs_per_component;
                  phi_current.begin_dof_values()[ind_j] = zero;
                }

            const auto ind_i = i + ic * phi_current.dofs_per_component;

            phi_current.begin_dof_values()[ind_i] = one;

            do_operation_on_cell(phi_current, phi_reference, cell);

            local_diagonal_vector[ind_i] =
              phi_current.begin_dof_values()[ind_i];
          }

      // Finally, in order to distribute diagonal, write it again into one of
      // FEEvaluations and do the standard distribute_local_to_global.
      // Note that here non-diagonal matrix elements are ignored and so the
      // result is not equivalent to matrix-based case when hanging nodes are
      // present. see Section 5.3 in Korman 2016, A time-space adaptive method
      // for the Schrodinger equation, doi: 10.4208/cicp.101214.021015a for a
      // discussion.
      for (unsigned int i = 0; i < phi_current.dofs_per_component; ++i)
        for (unsigned int ic = 0; ic < phi_current.n_components; ++ic)
          {
            const auto ind_i = i + ic * phi_current.dofs_per_component;
            phi_current.begin_dof_values()[ind_i] =
              local_diagonal_vector[ind_i];
          }

      phi_current.distribute_local_to_global(dst);
    } // end of cell loop
}



template <int dim, int fe_degree, int n_q_points_1d, typename Number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, Number>::do_operation_on_cell(
  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, Number> &phi_current,
  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, Number> &phi_reference,
  const unsigned int                                        cell) const
{
  const unsigned int material_id =
    data_current->get_cell_iterator(cell, 0)->material_id();
  const auto &cell_mat = (material_id == 0 ? material : material_inclusion);

  Assert(cell_mat->formulation <= 1,
         ExcMessage("Unknown material formulation"));
  // make sure all filled cells have the same ID
#ifdef DEBUG
  for (unsigned int i = 1;
       i < data_current->n_active_entries_per_cell_batch(cell);
       ++i)
    {
      const unsigned int other_id =
        data_current->get_cell_iterator(cell, i)->material_id();
      Assert(other_id == material_id,
             ExcMessage("Cell block " + std::to_string(cell) + " element " +
                        std::to_string(i) + " has material ID " +
                        std::to_string(other_id) +
                        ", different form 0-th element " +
                        std::to_string(material_id)));
      // make sure both MatrixFree objects use the same cells
      Assert(data_current->get_cell_iterator(cell, i) ==
               data_reference->get_cell_iterator(cell, i),
             ExcMessage("Cell block " + std::to_string(cell) + " element " +
                        std::to_string(i) +
                        " does not match between two MatrixFree objects."));
    }
#endif

  // make sure both MatrixFree objects use the same cells
  AssertDimension(data_current->n_active_entries_per_cell_batch(cell),
                  data_reference->n_active_entries_per_cell_batch(cell));

  static constexpr Number inv_dim_f    = 1.0 / dim;
  static constexpr Number two_over_dim = 2.0 / dim;
  const Number            kappa        = cell_mat->kappa;
  const Number            c_1          = cell_mat->c_1;
  const Number            mu           = cell_mat->mu;
  const Number            lambda       = cell_mat->lambda;

  // VMult sum factorization

  if (mf_caching == MFCaching::tensor4_ns ||
      mf_caching == MFCaching::scalar_referential)
    {
      phi_reference.evaluate(true, true, false);
    }
  else
    {
      if (mf_caching == MFCaching::scalar)
        phi_reference.evaluate(false, true, false);

      phi_current.evaluate(true, true, false);
    }


  // VMult quadrature loop
  if (cell_mat->formulation == 0)
    // volumetric/deviatoric formulation (like step-44)
    {
      for (unsigned int q = 0; q < phi_current.n_q_points; ++q)
        {
          // reference configuration:
          const Tensor<2, dim, VectorizedArrayType> &grad_u =
            phi_reference.get_gradient(q);
          const Tensor<2, dim, VectorizedArrayType> F =
            Physics::Elasticity::Kinematics::F(grad_u);
          const VectorizedArrayType                 det_F = determinant(F);
          const Tensor<2, dim, VectorizedArrayType> F_bar =
            F * cached_scalar(cell, q);
          const SymmetricTensor<2, dim, VectorizedArrayType> b_bar =
            Physics::Elasticity::Kinematics::b(F_bar);

          Assert(cached_scalar(cell, q) == std::pow(det_F, Number(-1.0 / dim)),
                 ExcMessage("Cached scalar and det_F do not match"));

          Assert(*std::min_element(
                   det_F.begin(),
                   det_F.begin() +
                     data_current->n_active_entries_per_cell_batch(cell)) > 0,
                 ExcMessage("det_F is not positive."));

          // current configuration
          const Tensor<2, dim, VectorizedArrayType> grad_Nx_v =
            phi_current.get_gradient(q);
          const SymmetricTensor<2, dim, VectorizedArrayType> symm_grad_Nx_v =
            symmetrize(grad_Nx_v);

          // Next, determine the isochoric Kirchhoff stress
          // $\boldsymbol{\tau}_{\textrm{iso}} =
          // \mathcal{P}:\overline{\boldsymbol{\tau}}$
          const SymmetricTensor<2, dim, VectorizedArrayType> tau_bar =
            b_bar * (2.0 * c_1);

          // trace of fictitious Kirchhoff stress
          // $\overline{\boldsymbol{\tau}}$:
          // 2.0 * c_1 * b_bar
          const VectorizedArrayType tr_tau_bar = trace(tau_bar);

          const VectorizedArrayType tr_tau_bar_dim = tr_tau_bar * inv_dim_f;

          // Derivative of the volumetric free energy with respect to
          // $J$ return $\frac{\partial
          // \Psi_{\text{vol}}(J)}{\partial J}$
          const VectorizedArrayType dPsi_vol_dJ =
            (kappa / 2.0) * (det_F - 1.0 / det_F);

          const VectorizedArrayType dPsi_vol_dJ_J = dPsi_vol_dJ * det_F;

          const VectorizedArrayType d2Psi_vol_dJ2 =
            ((kappa / 2.0) * (1.0 + 1.0 / (det_F * det_F)));

          // Kirchoff stress:
          SymmetricTensor<2, dim, VectorizedArrayType> tau;
          {
            tau = VectorizedArrayType();
            // See Holzapfel p231 eq6.98 onwards

            // The following functions are used internally in determining
            // the result of some of the public functions above. The first
            // one determines the volumetric Kirchhoff stress
            // $\boldsymbol{\tau}_{\textrm{vol}}$. Note the difference in
            // its definition when compared to step-44.
            for (unsigned int d = 0; d < dim; ++d)
              tau[d][d] = dPsi_vol_dJ_J - tr_tau_bar_dim;

            tau += tau_bar;
          }

          // material part of the action of tangent:
          // The action of the fourth-order material elasticity tensor in
          // the spatial setting on symmetric tensor.
          // $\mathfrak{c}$ is calculated from the SEF $\Psi$ as $ J
          // \mathfrak{c}_{ijkl} = F_{iA} F_{jB} \mathfrak{C}_{ABCD} F_{kC}
          // F_{lD}$ where $ \mathfrak{C} = 4 \frac{\partial^2
          // \Psi(\mathbf{C})}{\partial \mathbf{C} \partial \mathbf{C}}$
          SymmetricTensor<2, dim, VectorizedArrayType> jc_part;
          {
            const VectorizedArrayType tr = trace(symm_grad_Nx_v);

            SymmetricTensor<2, dim, VectorizedArrayType> dev_src(
              symm_grad_Nx_v);
            for (unsigned int i = 0; i < dim; ++i)
              dev_src[i][i] -= tr * inv_dim_f;

            // 1) The volumetric part of the tangent $J
            // \mathfrak{c}_\textrm{vol}$. Again, note the difference in its
            // definition when compared to step-44. The extra terms result
            // from two quantities in $\boldsymbol{\tau}_{\textrm{vol}}$
            // being dependent on $\boldsymbol{F}$.
            // See Holzapfel p265

            // the term with the 4-th order symmetric tensor which gives
            // symmetric part of the tensor it acts on
            jc_part = symm_grad_Nx_v;
            jc_part *= -dPsi_vol_dJ_J * 2.0;

            // term with IxI results in trace of the tensor times I
            const VectorizedArrayType tmp =
              det_F * (dPsi_vol_dJ + det_F * d2Psi_vol_dJ2) * tr;
            for (unsigned int i = 0; i < dim; ++i)
              jc_part[i][i] += tmp;

            // 2) the isochoric part of the tangent $J
            // \mathfrak{c}_\textrm{iso}$:

            // The isochoric Kirchhoff stress
            // $\boldsymbol{\tau}_{\textrm{iso}} =
            // \mathcal{P}:\overline{\boldsymbol{\tau}}$:
            SymmetricTensor<2, dim, VectorizedArrayType> tau_iso(tau_bar);
            for (unsigned int i = 0; i < dim; ++i)
              tau_iso[i][i] -= tr_tau_bar_dim;

            // term with deviatoric part of the tensor
            jc_part += (two_over_dim * tr_tau_bar) * dev_src;

            // term with tau_iso_x_I + I_x_tau_iso
            jc_part -= (two_over_dim * tr) * tau_iso;
            const VectorizedArrayType tau_iso_src = tau_iso * symm_grad_Nx_v;
            for (unsigned int i = 0; i < dim; ++i)
              jc_part[i][i] -= two_over_dim * tau_iso_src;

            // c_bar==0 so we don't have a term with it.
          }

#ifdef DEBUG
          const VectorizedArrayType &JxW_current = phi_current.JxW(q);
          VectorizedArrayType        JxW_scale   = phi_reference.JxW(q);
          for (unsigned int i = 0;
               i < data_current->n_active_entries_per_cell_batch(cell);
               ++i)
            {
              Assert(std::abs(JxW_current[i]) > 0., ExcInternalError());
              JxW_scale[i] *= 1. / JxW_current[i];
              // indirect check of consistency between MappingQEulerian in
              // MatrixFree data and displacement vector stored in this
              // operator.
              Assert(std::abs(JxW_scale[i] * det_F[i] - 1.) <
                       1000. * std::numeric_limits<Number>::epsilon(),
                     ExcMessage(
                       std::to_string(i) + " out of " +
                       std::to_string(VectorizedArrayType::size()) +
                       ", filled " +
                       std::to_string(
                         data_current->n_active_entries_per_cell_batch(cell)) +
                       " : " + std::to_string(det_F[i]) +
                       "!=" + std::to_string(1. / JxW_scale[i]) + " " +
                       std::to_string(std::abs(JxW_scale[i] * det_F[i] - 1.))));
            }
#endif

          // jc_part is the $\mathsf{\mathbf{k}}_{\mathbf{u} \mathbf{u}}$
          // contribution. It comprises a material contribution, and a
          // geometrical stress contribution which is only added along
          // the local matrix diagonals:

          // geometrical stress contribution
          // In index notation this tensor is $ [j e^{geo}]_{ijkl} = j
          // \delta_{ik} \sigma^{tot}_{jl} = \delta_{ik} \tau^{tot}_{jl} $.
          // the product is actually  GradN * tau^T but due to symmetry of
          // tau we can do GradN * tau
          const VectorizedArrayType inv_det_F = Number(1.0) / det_F;
          const Tensor<2, dim, VectorizedArrayType> tau_ns(tau);
          const Tensor<2, dim, VectorizedArrayType> geo = grad_Nx_v * tau_ns;
          phi_current.submit_gradient(
            (jc_part + geo) * inv_det_F
            // Note: We need to integrate over the reference element,
            // thus we divide by det_F so that FEEvaluation with
            // mapping does the right thing.
            ,
            q);
          phi_current.submit_value(phi_current.get_value(q) *
                                     cell_mat->rho_alpha * inv_det_F,
                                   q);
        } // end of the loop over quadrature points
    }
  else
    switch (mf_caching)
      {
          // the least amount of cache and the most calculations
          case MFCaching::scalar: {
            for (unsigned int q = 0; q < phi_current.n_q_points; ++q)
              {
                // reference configuration:
                const Tensor<2, dim, VectorizedArrayType> grad_u =
                  phi_reference.get_gradient(q);
                const Tensor<2, dim, VectorizedArrayType> F =
                  Physics::Elasticity::Kinematics::F(grad_u);
                const SymmetricTensor<2, dim, VectorizedArrayType> b =
                  Physics::Elasticity::Kinematics::b(F);

                const Tensor<2, dim, VectorizedArrayType> grad_Nx_v =
                  phi_current.get_gradient(q);
                const SymmetricTensor<2, dim, VectorizedArrayType>
                  symm_grad_Nx_v = symmetrize(grad_Nx_v);

                SymmetricTensor<2, dim, VectorizedArrayType> tau;
                {
                  tau = mu * b;
                  for (unsigned int d = 0; d < dim; ++d)
                    tau[d][d] -= cached_scalar(cell, q);
                }

                SymmetricTensor<2, dim, VectorizedArrayType> jc_part =
                  (Number(2.0) * cached_scalar(cell, q)) * symm_grad_Nx_v;
                {
                  const VectorizedArrayType tmp =
                    Number(2.0) * lambda * trace(symm_grad_Nx_v);
                  for (unsigned int i = 0; i < dim; ++i)
                    jc_part[i][i] += tmp;
                }

                const VectorizedArrayType det_F = determinant(F);
                Assert(cached_scalar(cell, q) ==
                         (mu - Number(2.0) * lambda * std::log(det_F)),
                       ExcMessage("Cached scalar and det_F do not match"));

#ifdef DEBUG
                const VectorizedArrayType JxW_current = phi_current.JxW(q);
                VectorizedArrayType       JxW_scale   = phi_reference.JxW(q);
                for (unsigned int i = 0;
                     i < data_current->n_active_entries_per_cell_batch(cell);
                     ++i)
                  {
                    Assert(std::abs(JxW_current[i]) > 0., ExcInternalError());
                    JxW_scale[i] *= 1. / JxW_current[i];
                    // indirect check of consistency between
                    // MappingQEulerian in MatrixFree data and displacement
                    // vector stored in this operator.
                    Assert(std::abs(JxW_scale[i] * det_F[i] - 1.) <
                             100000. * std::numeric_limits<Number>::epsilon(),
                           ExcMessage(
                             std::to_string(i) + " out of " +
                             std::to_string(VectorizedArrayType::size()) +
                             ", filled " +
                             std::to_string(
                               data_current->n_active_entries_per_cell_batch(
                                 cell)) +
                             " : " + std::to_string(det_F[i]) +
                             "!=" + std::to_string(1. / JxW_scale[i]) + " " +
                             std::to_string(
                               std::abs(JxW_scale[i] * det_F[i] - 1.))));
                  }
#endif
                const Tensor<2, dim, VectorizedArrayType> tau_ns(tau);
                const Tensor<2, dim, VectorizedArrayType> geo =
                  grad_Nx_v * tau_ns;
                const VectorizedArrayType inv_det_F = Number(1.0) / det_F;
                phi_current.submit_gradient((jc_part + geo) * inv_det_F, q);

                phi_current.submit_value(phi_current.get_value(q) *
                                           cell_mat->rho_alpha * inv_det_F,
                                         q);
              }
            break;
          }
        // the least amount of cache and the most calculations
        case MFCaching::scalar_referential:
          // MK:
          // What I implemented here is essentially the same as the usual
          // scalar variant. The only thing I had to change was to replace
          // grad x (and submit_gradient() in the spatial frame) by Grad x
          // (get_gradient() in referential frame) and then multiplying by
          // F^{-T}. And this is the F that gets computed in the scalar case
          // as well, I just hardcoded it as this was the way I thought
          // about it in the implementation, but one could of course use the
          // phi_reference.get_gradient() function. And similarly by F^{-T}
          // for submit_gradient(). In other words, I simply pulled out the
          // Eulerian motion of the grid into F; nothing else changed.
          {
            const VectorizedArrayType  one = make_vectorized_array<Number>(1.);
            const VectorizedArrayType *cached_position =
              &cached_second_scalar(cell, 0);
            constexpr unsigned int n_q_points =
              Utilities::pow(n_q_points_1d, dim);
            VectorizedArrayType *ref_grads = phi_current.begin_gradients();
            VectorizedArrayType *x_grads   = phi_reference.begin_gradients();

#if DEAL_II_VERSION_GTE(9, 3, 0)
            dealii::internal::FEEvaluationImplCollocation<
              dim,
              n_q_points_1d - 1,
              VectorizedArrayType>::evaluate(dim,
                                             EvaluationFlags::gradients,
                                             data_reference->get_shape_info(),
                                             cached_position,
                                             nullptr,
                                             ref_grads,
                                             nullptr,
                                             nullptr);
#else
            dealii::internal::FEEvaluationImplCollocation<
              dim,
              n_q_points_1d - 1,
              dim,
              VectorizedArrayType>::evaluate(data_reference->get_shape_info(),
                                             cached_position,
                                             nullptr,
                                             ref_grads,
                                             nullptr,
                                             nullptr,
                                             false,
                                             true,
                                             false);
#endif

            for (unsigned int q = 0; q < phi_current.n_q_points; ++q)
              {
                // Jacobian of element in referential space
                const Tensor<2, dim, VectorizedArrayType> inv_jac =
                  phi_reference.inverse_jacobian(q);
                Tensor<2, dim, VectorizedArrayType> F;
                Tensor<2, dim, VectorizedArrayType> grad_Nx_v;
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    for (unsigned int e = 0; e < dim; ++e)
                      {
                        VectorizedArrayType sum =
                          inv_jac[e][0] *
                          ref_grads[(d * dim + 0) * n_q_points + q];
                        for (unsigned int f = 1; f < dim; ++f)
                          sum += inv_jac[e][f] *
                                 ref_grads[(d * dim + f) * n_q_points + q];
                        F[d][e] = sum;

                        // since we already have the inverse Jacobian,
                        // simply apply the inverse Jacobian here rather
                        // than call get_gradient (the operations are the
                        // same otherwise)
                        VectorizedArrayType sum2 =
                          inv_jac[e][0] *
                          x_grads[(d * dim + 0) * n_q_points + q];
                        for (unsigned int f = 1; f < dim; ++f)
                          sum2 += inv_jac[e][f] *
                                  x_grads[(d * dim + f) * n_q_points + q];
                        grad_Nx_v[d][e] = sum2;
                      }
                    F[d][d] += one;
                  }
                const SymmetricTensor<2, dim, VectorizedArrayType> b =
                  Physics::Elasticity::Kinematics::b(F);

                SymmetricTensor<2, dim, VectorizedArrayType> tau = mu * b;
                for (unsigned int d = 0; d < dim; ++d)
                  tau[d][d] -= cached_scalar(cell, q);

                const Tensor<2, dim, VectorizedArrayType> F_inv = invert(F);
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    VectorizedArrayType tmp[dim];
                    for (unsigned int e = 0; e < dim; ++e)
                      tmp[e] = grad_Nx_v[d][e];
                    for (unsigned int e = 0; e < dim; ++e)
                      {
                        VectorizedArrayType sum = F_inv[0][e] * tmp[0];
                        for (unsigned int f = 1; f < dim; ++f)
                          sum += F_inv[f][e] * tmp[f];
                        grad_Nx_v[d][e] = sum;
                      }
                  }

                SymmetricTensor<2, dim, VectorizedArrayType> jc_part =
                  (Number(2.0) * cached_scalar(cell, q)) *
                  symmetrize(grad_Nx_v);
                {
                  const VectorizedArrayType tmp =
                    Number(2.0) * lambda * trace(grad_Nx_v);
                  for (unsigned int i = 0; i < dim; ++i)
                    jc_part[i][i] += tmp;
                }

                Tensor<2, dim, VectorizedArrayType> queued =
                  jc_part +
                  (grad_Nx_v * Tensor<2, dim, VectorizedArrayType>(tau));
                phi_reference.submit_gradient(queued * transpose(F_inv), q);
                phi_reference.submit_value(phi_reference.get_value(q) *
                                             cell_mat->rho_alpha,
                                           q);
                // MK: The 60 lines above this are the interesting part: I
                // only need to work with phi_reference. What happens in
                // addition to the scalar caching variant is that I have to
                // multiply grad_Nx_v also by F^{-T} to transform it to the
                // spatial configuration. Note that I expanded some of the
                // contractions manually in terms of the loops to be fully
                // sure what happens but we could also express them via
                // operator* between Tensor<2,dim> objects.
              }
            break;
          }
          // moderate cache of two scalar + 2nd order tensor
          case MFCaching::tensor2: {
            for (unsigned int q = 0; q < phi_current.n_q_points; ++q)
              {
                const Tensor<2, dim, VectorizedArrayType> grad_Nx_v =
                  phi_current.get_gradient(q);
                const SymmetricTensor<2, dim, VectorizedArrayType>
                  symm_grad_Nx_v = symmetrize(grad_Nx_v);

                SymmetricTensor<2, dim, VectorizedArrayType> jc_part =
                  cached_scalar(cell, q) * symm_grad_Nx_v;
                {
                  const VectorizedArrayType tmp =
                    2 * cell_mat->lambda * cached_second_scalar(cell, q) *
                    trace(symm_grad_Nx_v);

                  for (unsigned int i = 0; i < dim; ++i)
                    jc_part[i][i] += tmp;
                }

                phi_current.submit_gradient(jc_part + grad_Nx_v *
                                                        cached_tensor2(cell, q),
                                            q);
                phi_current.submit_value(phi_current.get_value(q) *
                                           cell_mat->rho_alpha *
                                           cached_second_scalar(cell, q),
                                         q);
              }
            break;
          }
        case MFCaching::tensor4:
          // maximum cache (2nd order  4th order sym)
          {
            for (unsigned int q = 0; q < phi_current.n_q_points; ++q)
              {
                const Tensor<2, dim, VectorizedArrayType> &grad_Nx_v =
                  phi_current.get_gradient(q);
                const SymmetricTensor<2, dim, VectorizedArrayType>
                  &symm_grad_Nx_v = symmetrize(grad_Nx_v);

                phi_current.submit_gradient(grad_Nx_v *
                                                cached_tensor2(cell, q) +
                                              cached_tensor4(cell, q) *
                                                symm_grad_Nx_v,
                                            q);
                phi_current.submit_value(phi_current.get_value(q) *
                                           cell_mat->rho_alpha *
                                           cached_second_scalar(cell, q),
                                         q);
              }
            break;
          }
        case MFCaching::tensor4_ns:
          // dP/dF, fully referential
          {
            for (unsigned int q = 0; q < phi_current.n_q_points; ++q)
              {
                const Tensor<2, dim, VectorizedArrayType> &grad_Nx_v =
                  phi_reference.get_gradient(q);

                phi_reference.submit_gradient(
                  double_contract<2, 0, 3, 1>(cached_tensor4_ns(cell, q),
                                              grad_Nx_v),
                  q);
                phi_reference.submit_value(phi_reference.get_value(q) *
                                             cell_mat->rho_alpha,
                                           q);
              }
            break;
          }
          case MFCaching::none: {
            AssertThrow(false, ExcMessage("Unknown caching"));
            break;
          }
      } // quadrature loop compile-time enum

  // VMult sum factorization
  if (mf_caching == MFCaching::tensor4_ns ||
      mf_caching == MFCaching::scalar_referential)
    {
      phi_reference.integrate(true, true);
    }
  else
    {
      phi_current.integrate(true, true);
    }
}



template <int dim, int fe_degree, int n_q_points_1d, typename Number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, Number>::compute_diagonal()
{
  inverse_diagonal_entries.reset(new DiagonalMatrix<VectorType>());
  diagonal_entries.reset(new DiagonalMatrix<VectorType>());
  VectorType &inverse_diagonal_vector = inverse_diagonal_entries->get_vector();
  VectorType &diagonal_vector         = diagonal_entries->get_vector();

  data_current->initialize_dof_vector(inverse_diagonal_vector);
  data_current->initialize_dof_vector(diagonal_vector);

  unsigned int dummy = 0;
  diagonal_vector    = 0.;
  local_diagonal_cell(*data_current,
                      diagonal_vector,
                      dummy,
                      std::make_pair<unsigned int, unsigned int>(
                        0, data_current->n_cell_batches()));
  diagonal_vector.compress(VectorOperation::add);

  // data_current->cell_loop (&NeoHookOperator::local_diagonal_cell,
  //                          this, diagonal_vector, dummy);

  // set_constrained_entries_to_one
  Assert(data_current->get_constrained_dofs().size() ==
           data_reference->get_constrained_dofs().size(),
         ExcInternalError());
  for (const auto dof : data_current->get_constrained_dofs())
    diagonal_vector.local_element(dof) = 1.;

  // calculate inverse:
  inverse_diagonal_vector = diagonal_vector;

  for (unsigned int i = 0; i < inverse_diagonal_vector.local_size(); ++i)
    {
      Assert(inverse_diagonal_vector.local_element(i) > 0.,
             ExcMessage("No diagonal entry in a positive definite operator "
                        "should be zero or negative"));
      inverse_diagonal_vector.local_element(i) =
        1. / diagonal_vector.local_element(i);
    }

  inverse_diagonal_vector.update_ghost_values();
  diagonal_vector.update_ghost_values();

  diagonal_is_available = true;
}



template <int dim, int fe_degree, int n_q_points_1d, typename Number>
Number
NeoHookOperator<dim, fe_degree, n_q_points_1d, Number>::el(
  const unsigned int row,
  const unsigned int col) const
{
  Assert(row == col, ExcNotImplemented());
  (void)col;
  Assert(diagonal_is_available == true, ExcNotInitialized());
  return diagonal_entries->get_vector()(row);
}
