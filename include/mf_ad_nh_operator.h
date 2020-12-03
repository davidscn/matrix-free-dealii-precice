#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/differentiation/ad.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.templates.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>

#include <deal.II/physics/notation.h>

#include <material.h>

using namespace dealii;

/**
 * Large strain Neo-Hook tangent operator.
 *
 * Follow
 * https://github.com/dealii/dealii/blob/master/tests/matrix_free/step-37.cc
 */
template <int dim, int fe_degree, int n_q_points_1d, typename number>
class NeoHookOperatorAD : public Subscriptor
{
public:
  NeoHookOperatorAD();

  typedef
    typename LinearAlgebra::distributed::Vector<number>::size_type size_type;

  void
  clear();

  void
  initialize(std::shared_ptr<const MatrixFree<dim, number>> data_current,
             std::shared_ptr<const MatrixFree<dim, number>> data_reference,
             LinearAlgebra::distributed::Vector<number> &   displacement);

  void
  set_material(
    std::shared_ptr<
      Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArray<number>>>
      material,
    std::shared_ptr<
      Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArray<number>>>
      material_inclusion);

  void
  cache()
  {}

  void
  compute_diagonal();

  unsigned int
  m() const;
  unsigned int
  n() const;

  void
  vmult(LinearAlgebra::distributed::Vector<double> &      dst,
        const LinearAlgebra::distributed::Vector<double> &src) const;

  void
  Tvmult(LinearAlgebra::distributed::Vector<double> &      dst,
         const LinearAlgebra::distributed::Vector<double> &src) const;
  void
  vmult_add(LinearAlgebra::distributed::Vector<double> &      dst,
            const LinearAlgebra::distributed::Vector<double> &src) const;
  void
  Tvmult_add(LinearAlgebra::distributed::Vector<double> &      dst,
             const LinearAlgebra::distributed::Vector<double> &src) const;

  number
  el(const unsigned int row, const unsigned int col) const;

  void
  precondition_Jacobi(LinearAlgebra::distributed::Vector<number> &      dst,
                      const LinearAlgebra::distributed::Vector<number> &src,
                      const number omega) const;

private:
  /**
   * Apply operator on a range of cells.
   */
  void
  local_apply_cell(
    const MatrixFree<dim, number> &                   data,
    LinearAlgebra::distributed::Vector<double> &      dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const;

  /**
   * Apply diagonal part of the operator on a cell range.
   */
  void
  local_diagonal_cell(
    const MatrixFree<dim, number> &             data,
    LinearAlgebra::distributed::Vector<double> &dst,
    const unsigned int &,
    const std::pair<unsigned int, unsigned int> &cell_range) const;


  /**
   * Perform operation on a cell. @p phi_current and @phi_current_s correspond to the deformed configuration
   * where @p phi_reference is for the current configuration.
   */
  void
  do_operation_on_cell(
    FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number> &phi_reference,
    FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number> &phi_solution,
    const unsigned int                                        cell) const;

  std::shared_ptr<const MatrixFree<dim, number>> data_reference;

  LinearAlgebra::distributed::Vector<number> *displacement;

  std::shared_ptr<
    Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArray<number>>>
    material;

  std::shared_ptr<
    Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArray<number>>>
    material_inclusion;

  std::shared_ptr<DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>>
    inverse_diagonal_entries;
  std::shared_ptr<DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>>
    diagonal_entries;

  bool diagonal_is_available;
};



template <int dim, int fe_degree, int n_q_points_1d, typename number>
NeoHookOperatorAD<dim, fe_degree, n_q_points_1d, number>::NeoHookOperatorAD()
  : Subscriptor()
  , diagonal_is_available(false)
{}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperatorAD<dim, fe_degree, n_q_points_1d, number>::precondition_Jacobi(
  LinearAlgebra::distributed::Vector<number> &      dst,
  const LinearAlgebra::distributed::Vector<number> &src,
  const number                                      omega) const
{
  Assert(inverse_diagonal_entries.get() && inverse_diagonal_entries->m() > 0,
         ExcNotInitialized());
  inverse_diagonal_entries->vmult(dst, src);
  dst *= omega;
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
unsigned int
NeoHookOperatorAD<dim, fe_degree, n_q_points_1d, number>::m() const
{
  return data_reference.get_vector_partitioner()->size();
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
unsigned int
NeoHookOperatorAD<dim, fe_degree, n_q_points_1d, number>::n() const
{
  return data_reference.get_vector_partitioner()->size();
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperatorAD<dim, fe_degree, n_q_points_1d, number>::clear()
{
  data_reference.reset();
  diagonal_is_available = false;
  diagonal_entries.reset();
  inverse_diagonal_entries.reset();
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperatorAD<dim, fe_degree, n_q_points_1d, number>::initialize(
  std::shared_ptr<const MatrixFree<dim, number>> /*data_current_*/,
  std::shared_ptr<const MatrixFree<dim, number>> data_reference_,
  LinearAlgebra::distributed::Vector<number> &   displacement_)
{
  data_reference = data_reference_;
  displacement   = &displacement_;
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperatorAD<dim, fe_degree, n_q_points_1d, number>::set_material(
  std::shared_ptr<
    Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArray<number>>>
    material_,
  std::shared_ptr<
    Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArray<number>>>
    material_inclusion_)
{
  material           = material_;
  material_inclusion = material_inclusion_;
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperatorAD<dim, fe_degree, n_q_points_1d, number>::vmult(
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src) const
{
  dst = 0;
  vmult_add(dst, src);
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperatorAD<dim, fe_degree, n_q_points_1d, number>::Tvmult(
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src) const
{
  dst = 0;
  vmult_add(dst, src);
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperatorAD<dim, fe_degree, n_q_points_1d, number>::Tvmult_add(
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src) const
{
  vmult_add(dst, src);
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperatorAD<dim, fe_degree, n_q_points_1d, number>::vmult_add(
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src) const
{
  // FIXME: can't use cell_loop as we need both matrix-free data objects.
  // for now do it by hand.
  // BUT I might try cell_loop(), and simply use another MF object inside...

  // MatrixFree::cell_loop() is more complicated than a simple
  // update_ghost_values() / compress(), it loops on different cells (inner
  // without ghosts and outer) in different order and do update_ghost_values()
  // and compress_start()/compress_finish() in between.
  // https://www.dealii.org/developer/doxygen/deal.II/matrix__free_8h_source.html#l00109

  // 1. make sure ghosts are updated
  src.update_ghost_values();

  // 2. loop over all locally owned cell blocks
  local_apply_cell(*data_reference,
                   dst,
                   src,
                   std::make_pair<unsigned int, unsigned int>(
                     0, data_reference->n_macro_cells()));

  // 3. communicate results with MPI
  dst.compress(VectorOperation::add);

  // 4. constraints
  const std::vector<unsigned int> &constrained_dofs =
    data_reference
      ->get_constrained_dofs(); // FIXME: is it current or reference?
  for (unsigned int i = 0; i < constrained_dofs.size(); ++i)
    dst.local_element(constrained_dofs[i]) +=
      src.local_element(constrained_dofs[i]);
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperatorAD<dim, fe_degree, n_q_points_1d, number>::local_apply_cell(
  const MatrixFree<dim, number> & /*data*/,
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src,
  const std::pair<unsigned int, unsigned int> &     cell_range) const
{
  // FIXME: I don't use data input, can this be bad?

  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number> phi_solution(
    *data_reference);
  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number> phi_reference(
    *data_reference);

  Assert(phi_solution.n_q_points == phi_reference.n_q_points,
         ExcInternalError());

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // initialize on this cell
      phi_solution.reinit(cell);
      phi_reference.reinit(cell);

      // read-in total displacement and src vector and evaluate gradients
      phi_solution.read_dof_values_plain(*displacement);
      phi_reference.read_dof_values(src);

      do_operation_on_cell(phi_reference, phi_solution, cell);

      phi_reference.distribute_local_to_global(dst);
    }
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperatorAD<dim, fe_degree, n_q_points_1d, number>::local_diagonal_cell(
  const MatrixFree<dim, number> & /*data*/,
  LinearAlgebra::distributed::Vector<double> &dst,
  const unsigned int &,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  // FIXME: I don't use data input, can this be bad?

  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number> phi_solution(
    *data_reference);
  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number> phi_reference(
    *data_reference);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // initialize on this cell
      phi_reference.reinit(cell);
      phi_solution.reinit(cell);

      // read-in total displacement.
      phi_solution.read_dof_values_plain(*displacement);

      // FIXME: although we override DoFs manually later, somehow
      // we still need to read some dummy here
      phi_reference.read_dof_values(*displacement);

      AlignedVector<VectorizedArray<number>> local_diagonal_vector(
        phi_reference.dofs_per_component * phi_reference.n_components);

      // Loop over all DoFs and set dof values to zero everywhere but i-th DoF.
      // With this input (instead of read_dof_values()) we do the action and
      // store the result in a diagonal vector
      for (unsigned int i = 0; i < phi_reference.dofs_per_component; ++i)
        for (unsigned int ic = 0; ic < phi_reference.n_components; ++ic)
          {
            for (unsigned int j = 0; j < phi_reference.dofs_per_component; ++j)
              for (unsigned int jc = 0; jc < phi_reference.n_components; ++jc)
                {
                  const auto ind_j = j + jc * phi_reference.dofs_per_component;
                  phi_reference.begin_dof_values()[ind_j] =
                    VectorizedArray<number>();
                }

            const auto ind_i = i + ic * phi_reference.dofs_per_component;

            phi_reference.begin_dof_values()[ind_i] = 1.;

            do_operation_on_cell(phi_reference, phi_solution, cell);

            local_diagonal_vector[ind_i] =
              phi_reference.begin_dof_values()[ind_i];
          }

      // Finally, in order to distribute diagonal, write it again into one of
      // FEEvaluations and do the standard distribute_local_to_global.
      // Note that here non-diagonal matrix elements are ignored and so the
      // result is not equivalent to matrix-based case when hanging nodes are
      // present. see Section 5.3 in Korman 2016, A time-space adaptive method
      // for the Schrodinger equation, doi: 10.4208/cicp.101214.021015a for a
      // discussion.
      for (unsigned int i = 0; i < phi_reference.dofs_per_component; ++i)
        for (unsigned int ic = 0; ic < phi_reference.n_components; ++ic)
          {
            const auto ind_i = i + ic * phi_reference.dofs_per_component;
            phi_reference.begin_dof_values()[ind_i] =
              local_diagonal_vector[ind_i];
          }

      phi_reference.distribute_local_to_global(dst);
    } // end of cell loop
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperatorAD<dim, fe_degree, n_q_points_1d, number>::do_operation_on_cell(
  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number> &phi_reference,
  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number> &phi_solution,
  const unsigned int /*cell*/) const
{
  phi_reference.evaluate(false, true, false);
  phi_solution.evaluate(false, true, false);

  using ADType = Sacado::Fad::DFad<double>;

  for (unsigned int q = 0; q < phi_reference.n_q_points; ++q)
    {
      Tensor<4, dim, VectorizedArray<double>> H;
      for (unsigned int vec = 0; vec < VectorizedArray<number>::size(); ++vec)
        {
          Tensor<2, dim, ADType> grad_u;
          Tensor<2, dim>         tmp_grad;
          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              tmp_grad[i][j] = phi_solution.get_gradient(q)[i][j][vec];

          for (unsigned int el = 0;
               el < Tensor<2, dim, ADType>::n_independent_components;
               ++el)
            grad_u[grad_u.unrolled_to_component_indices(el)] =
              ADType(Tensor<2, dim, ADType>::n_independent_components,
                     el,
                     tmp_grad[grad_u.unrolled_to_component_indices(el)]);

          // reference configuration:
          const Tensor<2, dim, ADType> F =
            Physics::Elasticity::Kinematics::F(grad_u);
          const SymmetricTensor<2, dim, ADType> b =
            Physics::Elasticity::Kinematics::b(F);
          const ADType                 det_F = determinant(F);
          const Tensor<2, dim, ADType> F_bar =
            Physics::Elasticity::Kinematics::F_iso(F);
          const SymmetricTensor<2, dim, ADType> b_bar =
            Physics::Elasticity::Kinematics::b(F_bar);

          SymmetricTensor<2, dim, ADType> tau;
          material->get_tau(tau, det_F, b_bar, b);

          Tensor<2, dim, ADType> P = tau * transpose(invert(F));
          for (unsigned int idx_i = 0; idx_i < P.n_independent_components;
               ++idx_i)
            {
              const TableIndices<2> tbl_idx_i =
                P.unrolled_to_component_indices(idx_i);
              for (unsigned int idx_j = 0; idx_j < P.n_independent_components;
                   ++idx_j)
                {
                  const TableIndices<2> tbl_idx_j =
                    P.unrolled_to_component_indices(idx_j);
                  TableIndices<4> tbl_idx_4;
                  tbl_idx_4[0]      = tbl_idx_i[0];
                  tbl_idx_4[1]      = tbl_idx_i[1];
                  tbl_idx_4[2]      = tbl_idx_j[0];
                  tbl_idx_4[3]      = tbl_idx_j[1];
                  H[tbl_idx_4][vec] = P[tbl_idx_i].dx(idx_j);
                }
            }
        }

      phi_reference.submit_gradient(
        double_contract<2, 0, 3, 1>(H, phi_reference.get_gradient(q)), q);
    } // end of the loop over quadrature points

  // actually do the contraction
  phi_reference.integrate(false, true);
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperatorAD<dim, fe_degree, n_q_points_1d, number>::compute_diagonal()
{
  typedef LinearAlgebra::distributed::Vector<number> VectorType;

  inverse_diagonal_entries.reset(new DiagonalMatrix<VectorType>());
  diagonal_entries.reset(new DiagonalMatrix<VectorType>());
  VectorType &inverse_diagonal_vector = inverse_diagonal_entries->get_vector();
  VectorType &diagonal_vector         = diagonal_entries->get_vector();

  data_reference->initialize_dof_vector(inverse_diagonal_vector);
  data_reference->initialize_dof_vector(diagonal_vector);

  unsigned int dummy = 0;
  local_diagonal_cell(*data_reference,
                      diagonal_vector,
                      dummy,
                      std::make_pair<unsigned int, unsigned int>(
                        0, data_reference->n_macro_cells()));
  diagonal_vector.compress(VectorOperation::add);

  // data_current->cell_loop (&NeoHookOperatorAD::local_diagonal_cell,
  //                          this, diagonal_vector, dummy);

  // set_constrained_entries_to_one
  {
    const std::vector<unsigned int> &constrained_dofs =
      data_reference->get_constrained_dofs();
    for (unsigned int i = 0; i < constrained_dofs.size(); ++i)
      diagonal_vector.local_element(constrained_dofs[i]) = 1.;
  }

  // calculate inverse:
  inverse_diagonal_vector = diagonal_vector;

  for (unsigned int i = 0; i < inverse_diagonal_vector.local_size(); ++i)
    if (std::abs(inverse_diagonal_vector.local_element(i)) >
        std::sqrt(std::numeric_limits<number>::epsilon()))
      inverse_diagonal_vector.local_element(i) =
        1. / inverse_diagonal_vector.local_element(i);
    else
      inverse_diagonal_vector.local_element(i) = 1.;

  inverse_diagonal_vector.update_ghost_values();
  diagonal_vector.update_ghost_values();

  diagonal_is_available = true;
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
number
NeoHookOperatorAD<dim, fe_degree, n_q_points_1d, number>::el(
  const unsigned int row,
  const unsigned int col) const
{
  Assert(row == col, ExcNotImplemented());
  (void)col;
  Assert(diagonal_is_available == true, ExcNotInitialized());
  return diagonal_entries->get_vector()(row);
}
