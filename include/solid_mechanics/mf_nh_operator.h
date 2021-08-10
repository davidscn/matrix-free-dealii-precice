#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>

#include <base/fe_integrator.h>
#include <solid_mechanics/material.h>

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
template <int dim, typename Number>
class NeoHookOperator : public Subscriptor
{
public:
  NeoHookOperator();

  using size_type =
    typename LinearAlgebra::distributed::Vector<Number>::size_type;
  using VectorType          = LinearAlgebra::distributed::Vector<Number>;
  using VectorizedArrayType = VectorizedArray<Number>;
  using FECellIntegrator =
    FECellIntegrators<dim, dim, Number, VectorizedArrayType>;

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

  template <typename Timer>
  void
  compute_diagonal(Timer &timer);

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
   * Perform operation on a cell. @p phi_current corresponds to the deformed configuration
   * where @p phi_reference is for the initial configuration.
   */
  void
  do_operation_on_cell(FECellIntegrator &phi) const;

  std::shared_ptr<const MatrixFree<dim, Number>> data_current;
  std::shared_ptr<const MatrixFree<dim, Number>> data_reference;
  std::shared_ptr<const MatrixFree<dim, Number>> data_in_use;

  const VectorType *displacement;

  std::shared_ptr<
    Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArrayType>>
    material;

  std::shared_ptr<
    Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArrayType>>
    material_inclusion;

  std::shared_ptr<DiagonalMatrix<VectorType>> inverse_diagonal_entries;

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
    tensor2,
    tensor4,
    tensor4_ns
  };
  MFCaching mf_caching;

  enum class MFFrame
  {
    none,
    referential,
    deformed
  };
  MFFrame mf_frame;

  Tensor<4, dim, VectorizedArrayType> IxI_ikjl;
};



template <int dim, typename Number>
std::size_t
NeoHookOperator<dim, Number>::memory_consumption() const
{
  auto res = cached_scalar.memory_consumption() +
             cached_second_scalar.memory_consumption() +
             cached_tensor2.memory_consumption() +
             cached_tensor4.memory_consumption() +
             cached_tensor4_ns.memory_consumption();

  // matrix-free data:
  if (mf_frame == MFFrame::referential)
    res += data_reference->memory_consumption();
  else
    res += data_current->memory_consumption();

  // note: do not include diagonals, we want to measure only memory needed for
  // vmult for performance analysis.
  return res;
}



template <int dim, typename Number>
NeoHookOperator<dim, Number>::NeoHookOperator()
  : Subscriptor()
  , diagonal_is_available(false)
  , mf_caching(MFCaching::none)
  , mf_frame(MFFrame::none)
{
  Tensor<2, dim, VectorizedArrayType> I;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      I[i][j] = make_vectorized_array<Number>(i == j);

  IxI_ikjl = outer_product_ikjl(I, I);
}



template <int dim, typename Number>
void
NeoHookOperator<dim, Number>::precondition_Jacobi(VectorType &      dst,
                                                  const VectorType &src,
                                                  const Number      omega) const
{
  Assert(inverse_diagonal_entries.get() && inverse_diagonal_entries->m() > 0,
         ExcNotInitialized());
  inverse_diagonal_entries->vmult(dst, src);
  dst *= omega;
}



template <int dim, typename Number>
unsigned int
NeoHookOperator<dim, Number>::m() const
{
  return data_current->get_vector_partitioner()->size();
}



template <int dim, typename Number>
unsigned int
NeoHookOperator<dim, Number>::n() const
{
  return data_current->get_vector_partitioner()->size();
}



template <int dim, typename Number>
void
NeoHookOperator<dim, Number>::clear()
{
  data_current.reset();
  data_reference.reset();
  data_in_use.reset();

  diagonal_is_available = false;
  inverse_diagonal_entries.reset();

  mf_caching = MFCaching::none;
  mf_frame   = MFFrame::none;
}



template <int dim, typename Number>
void
NeoHookOperator<dim, Number>::initialize(
  std::shared_ptr<const MatrixFree<dim, Number>> data_current_,
  std::shared_ptr<const MatrixFree<dim, Number>> data_reference_,
  const VectorType &                             displacement_,
  const std::string &                            caching)
{
  data_current   = data_current_;
  data_reference = data_reference_;
  displacement   = &displacement_;
  inverse_diagonal_entries.reset(new DiagonalMatrix<VectorType>());

  const unsigned int n_cells = data_reference_->n_cell_batches();
  FECellIntegrator   phi(*data_reference_);

  if (caching == "scalar_referential")
    {
      mf_caching  = MFCaching::scalar_referential;
      mf_frame    = MFFrame::referential;
      data_in_use = data_reference_;
      cached_scalar.reinit(n_cells, phi.n_q_points);
      // Store plain valus obtained from 'read_dof_values'
      cached_second_scalar.reinit(n_cells, phi.dofs_per_cell);
    }
  else if (caching == "tensor2")
    {
      // The second scalar is only used for formulation one and not for
      // formulation 0
      mf_caching  = MFCaching::tensor2;
      mf_frame    = MFFrame::deformed;
      data_in_use = data_current_;
      cached_scalar.reinit(n_cells, phi.n_q_points);
      cached_second_scalar.reinit(n_cells, phi.n_q_points);
      cached_tensor2.reinit(n_cells, phi.n_q_points);
    }
  else if (caching == "tensor4")
    {
      mf_caching  = MFCaching::tensor4;
      mf_frame    = MFFrame::deformed;
      data_in_use = data_current_;
      cached_tensor2.reinit(n_cells, phi.n_q_points);
      cached_tensor4.reinit(n_cells, phi.n_q_points);
      cached_second_scalar.reinit(n_cells, phi.n_q_points);
    }
  else if (caching == "tensor4_ns")
    {
      mf_caching  = MFCaching::tensor4_ns;
      mf_frame    = MFFrame::referential;
      data_in_use = data_reference_;
      cached_tensor4_ns.reinit(n_cells, phi.n_q_points);
    }
  else
    {
      mf_caching = MFCaching::none;
      mf_frame   = MFFrame::none;
      AssertThrow(false, ExcMessage("Unknown caching"));
    }
}



template <int dim, typename Number>
void
NeoHookOperator<dim, Number>::cache()
{
  const unsigned int n_cells = data_reference->n_cell_batches();

  FECellIntegrator phi_reference(*data_reference);

  for (unsigned int cell = 0; cell < n_cells; ++cell)
    {
      const unsigned int material_id =
        data_current->get_cell_iterator(cell, 0)->material_id();
      const auto &cell_mat = (material_id == 0 ? material : material_inclusion);

      Assert(cell_mat->formulation <= 1,
             ExcMessage("Unknown material formulation"));

      phi_reference.reinit(cell);
      phi_reference.read_dof_values_plain(*displacement);
      phi_reference.evaluate(EvaluationFlags::gradients);

      // In order to avoid the phi_reference.read_dof_values() using
      // indirect addressing
      if (mf_caching == MFCaching::scalar_referential)
        for (unsigned int i = 0; i < phi_reference.dofs_per_cell; ++i)
          cached_second_scalar(cell, i) = phi_reference.begin_dof_values()[i];

      if (cell_mat->formulation == 0)
        {
          Assert(mf_caching == MFCaching::tensor2, ExcNotImplemented());
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

              cached_scalar(cell, q)  = std::pow(det_F, Number(-1.0 / dim));
              cached_tensor2(cell, q) = F;
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
                  case MFCaching::scalar_referential: {
                    cached_scalar(cell, q) = scalar;
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
              } // End MFCaching
          }     // End n_q_points_loop
    }           // End cell loop
}



template <int dim, typename Number>
void
NeoHookOperator<dim, Number>::set_material(
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



template <int dim, typename Number>
void
NeoHookOperator<dim, Number>::vmult(VectorType &      dst,
                                    const VectorType &src) const
{
  dst = 0;
  vmult_add(dst, src);
}



template <int dim, typename Number>
void
NeoHookOperator<dim, Number>::Tvmult(VectorType &      dst,
                                     const VectorType &src) const
{
  dst = 0;
  vmult_add(dst, src);
}



template <int dim, typename Number>
void
NeoHookOperator<dim, Number>::Tvmult_add(VectorType &      dst,
                                         const VectorType &src) const
{
  vmult_add(dst, src);
}



template <int dim, typename Number>
void
NeoHookOperator<dim, Number>::vmult_add(VectorType &      dst,
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



template <int dim, typename Number>
void
NeoHookOperator<dim, Number>::local_apply_cell(
  const MatrixFree<dim, Number> & /*data*/,
  VectorType &                                 dst,
  const VectorType &                           src,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  Assert(data_in_use.get() != nullptr, ExcInternalError());
  FECellIntegrator phi(*data_in_use);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values(src);
      do_operation_on_cell(phi);
      phi.distribute_local_to_global(dst);
    }
}



template <int dim, typename Number>
void
NeoHookOperator<dim, Number>::do_operation_on_cell(FECellIntegrator &phi) const
{
  const unsigned int cell = phi.get_current_cell_index();
  const unsigned int material_id =
    data_in_use->get_cell_iterator(cell, 0)->material_id();
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
  phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

  // VMult quadrature loop
  // volumetric/deviatoric formulation (like step-44)
  if (cell_mat->formulation == 0)
    for (unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        // Assert the appropriate configuration
        Assert(mf_caching == MFCaching::tensor2, ExcNotImplemented());
        Assert(mf_frame == MFFrame::deformed, ExcNotImplemented());

        // reference configuration where F is precomputed:
        const Tensor<2, dim, VectorizedArrayType> &F = cached_tensor2(cell, q);
        const VectorizedArrayType                  det_F = determinant(F);
        const Tensor<2, dim, VectorizedArrayType>  F_bar =
          F * cached_scalar(cell, q);
        const SymmetricTensor<2, dim, VectorizedArrayType> b_bar =
          Physics::Elasticity::Kinematics::b(F_bar);

        Assert(cached_scalar(cell, q) == std::pow(det_F, Number(-1.0 / dim)),
               ExcMessage("Cached scalar and det_F do not match"));

        Assert(*std::min_element(det_F.begin(),
                                 det_F.begin() +
                                   data_in_use->n_active_entries_per_cell_batch(
                                     cell)) > 0,
               ExcMessage("det_F is not positive."));

        // current configuration
        const Tensor<2, dim, VectorizedArrayType> grad_Nx_v =
          phi.get_gradient(q);
        const SymmetricTensor<2, dim, VectorizedArrayType> symm_grad_Nx_v =
          symmetrize(grad_Nx_v);

        // Next, determine the isochoric Kirchhoff stress
        // $\boldsymbol{\tau}_{\textrm{iso}} =
        // \mathcal{P}:\overline{\boldsymbol{\tau}}$
        const SymmetricTensor<2, dim, VectorizedArrayType> tau_bar =
          b_bar * (2.0 * c_1);

        // trace of fictitious Kirchhoff stress
        // $\overline{\boldsymbol{\tau}}$: 2.0 * c_1 * b_bar
        const VectorizedArrayType tr_tau_bar     = trace(tau_bar);
        const VectorizedArrayType tr_tau_bar_dim = tr_tau_bar * inv_dim_f;

        // Derivative of the volumetric free energy with respect to
        // $J$ return $\frac{\partial \Psi_{\text{vol}}(J)}{\partial J}$
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
          // The following functions are used internally in determining the
          // result of some of the public functions above. The first one
          // determines the volumetric Kirchhoff stress
          // $\boldsymbol{\tau}_{\textrm{vol}}$. Note the difference in its
          // definition when compared to step-44.
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

          SymmetricTensor<2, dim, VectorizedArrayType> dev_src(symm_grad_Nx_v);
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

        // jc_part is the $\mathsf{\mathbf{k}}_{\mathbf{u} \mathbf{u}}$
        // contribution. It comprises a material contribution, and a
        // geometrical stress contribution which is only added along
        // the local matrix diagonals: geometrical stress contribution
        // In index notation this tensor is $ [j e^{geo}]_{ijkl} = j
        // \delta_{ik} \sigma^{tot}_{jl} = \delta_{ik} \tau^{tot}_{jl} $.
        // the product is actually  GradN * tau^T but due to symmetry of
        // tau we can do GradN * tau
        const VectorizedArrayType inv_det_F = Number(1.0) / det_F;
        const Tensor<2, dim, VectorizedArrayType> tau_ns(tau);
        const Tensor<2, dim, VectorizedArrayType> geo = grad_Nx_v * tau_ns;
        phi.submit_gradient((jc_part + geo) * inv_det_F
                            // Note: We need to integrate over the reference
                            // element, thus we divide by det_F so that
                            // FEEvaluation with mapping does the right thing.
                            ,
                            q);
        phi.submit_value(phi.get_value(q) * cell_mat->rho_alpha * inv_det_F, q);
      } // end of the loop over quadrature points
  else
    switch (mf_caching)
      {
          // the least amount of cache and the most calculations Instead of
          // using the current dof handler grad x (and submit_gradient()), Grad
          // x (get_gradient() in referential frame) is used and then multiplied
          // by F^{-T}
          case MFCaching::scalar_referential: {
            // Assert frame
            Assert(mf_frame == MFFrame::referential, ExcNotImplemented());
            const VectorizedArrayType  one = make_vectorized_array<Number>(1.);
            const VectorizedArrayType *cached_position =
              &cached_second_scalar(cell, 0);
            const unsigned int n_q_points = phi.n_q_points;

            VectorizedArrayType *x_grads = phi.begin_gradients();

            // FIXME: The implementation here relies on the cached dof values.
            // However, it would be faster to cache the data on quadrature
            // points and call here a cheaper Collocation gradient, since the
            // evaluation to quadrature point is bypassed. For compatibility
            // reasons with the new FECellIntegrator<-1,0> this is not possible,
            // because the the collocation function requires explicit
            // instantiations with degree and n_quadrature_points
            FECellIntegrator phi_grad(phi);
            phi_grad.evaluate(cached_position, EvaluationFlags::gradients);
            VectorizedArrayType *ref_grads = phi_grad.begin_gradients();

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                // Jacobian of element in referential space
                const Tensor<2, dim, VectorizedArrayType> inv_jac =
                  phi.inverse_jacobian(q);
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
                phi.submit_gradient(queued * transpose(F_inv), q);
                phi.submit_value(phi.get_value(q) * cell_mat->rho_alpha, q);
              }
            break;
          }
          // moderate cache of two scalar + 2nd order tensor
          case MFCaching::tensor2: {
            // Assert frame
            Assert(mf_frame == MFFrame::deformed, ExcNotImplemented());
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                const Tensor<2, dim, VectorizedArrayType> grad_Nx_v =
                  phi.get_gradient(q);
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

                phi.submit_gradient(jc_part +
                                      grad_Nx_v * cached_tensor2(cell, q),
                                    q);
                phi.submit_value(phi.get_value(q) * cell_mat->rho_alpha *
                                   cached_second_scalar(cell, q),
                                 q);
              }
            break;
          }
          // maximum cache (2nd order  4th order sym)
          case MFCaching::tensor4: {
            // Assert frame
            Assert(mf_frame == MFFrame::deformed, ExcNotImplemented());
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                const Tensor<2, dim, VectorizedArrayType> &grad_Nx_v =
                  phi.get_gradient(q);
                const SymmetricTensor<2, dim, VectorizedArrayType>
                  &symm_grad_Nx_v = symmetrize(grad_Nx_v);

                phi.submit_gradient(grad_Nx_v * cached_tensor2(cell, q) +
                                      cached_tensor4(cell, q) * symm_grad_Nx_v,
                                    q);
                phi.submit_value(phi.get_value(q) * cell_mat->rho_alpha *
                                   cached_second_scalar(cell, q),
                                 q);
              }
            break;
          }
          // dP/dF, fully referential
          case MFCaching::tensor4_ns: {
            // Assert frame
            Assert(mf_frame == MFFrame::referential, ExcNotImplemented());
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                const Tensor<2, dim, VectorizedArrayType> &grad_Nx_v =
                  phi.get_gradient(q);

                phi.submit_gradient(double_contract<2, 0, 3, 1>(
                                      cached_tensor4_ns(cell, q), grad_Nx_v),
                                    q);
                phi.submit_value(phi.get_value(q) * cell_mat->rho_alpha, q);
              }
            break;
          }
          case MFCaching::none: {
            AssertThrow(false, ExcMessage("Unknown caching"));
            break;
          }
      }

  // VMult sum factorization
  phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
}



template <int dim, typename Number>
template <typename Timer>
void
NeoHookOperator<dim, Number>::compute_diagonal(Timer &timer)
{
  VectorType &inverse_diagonal_vector = inverse_diagonal_entries->get_vector();
  data_in_use->initialize_dof_vector(inverse_diagonal_vector);
  timer.enter_subsection("Tools: diagonal");

  int dummy = 0;
  data_in_use->template cell_loop<VectorType, int>(
    [this](const auto &data, auto &dst, const auto &, const auto &cell_range) {
      // Initialize data structures
      const VectorizedArrayType one  = make_vectorized_array<Number>(1.);
      const VectorizedArrayType zero = make_vectorized_array<Number>(0.);
      FECellIntegrator          phi(data);
      for (unsigned int cell = cell_range.first; cell < cell_range.second;
           ++cell)
        {
          phi.reinit(cell);
          AlignedVector<VectorizedArrayType> local_diagonal_vector(
            phi.dofs_per_cell);
          // Loop over all DoFs and set dof values to zero everywhere but i-th
          // DoF. With this input (instead of read_dof_values()) we do the
          // action and store the result in a diagonal vector
          for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
                phi.begin_dof_values()[j] = zero;

              phi.begin_dof_values()[i] = one;
              do_operation_on_cell(phi);
              local_diagonal_vector[i] = phi.begin_dof_values()[i];
            }

          // Cannot handle hanging nodes
          for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
            phi.begin_dof_values()[i] = local_diagonal_vector[i];
          phi.distribute_local_to_global(dst);
        }
    },
    inverse_diagonal_vector,
    dummy);

  timer.leave_subsection();

  // set_constrained_entries_to_one
  Assert(data_current->get_constrained_dofs().size() ==
           data_reference->get_constrained_dofs().size(),
         ExcInternalError());
  for (const auto dof : data_in_use->get_constrained_dofs())
    inverse_diagonal_vector.local_element(dof) = 1.;

  for (unsigned int i = 0; i < inverse_diagonal_vector.locally_owned_size();
       ++i)
    {
      Assert(inverse_diagonal_vector.local_element(i) > 0.,
             ExcMessage("No diagonal entry in a positive definite operator "
                        "should be zero or negative"));
      inverse_diagonal_vector.local_element(i) =
        1. / inverse_diagonal_vector.local_element(i);
    }

  inverse_diagonal_vector.update_ghost_values();
  diagonal_is_available = true;
}



template <int dim, typename Number>
Number
NeoHookOperator<dim, Number>::el(const unsigned int row,
                                 const unsigned int col) const
{
  Assert(row == col, ExcNotImplemented());
  (void)col;
  Assert(diagonal_is_available == true, ExcNotInitialized());
  return 1. / inverse_diagonal_entries->get_vector()(row);
}
