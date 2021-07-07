#pragma once

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <base/fe_integrator.h>

DEAL_II_NAMESPACE_OPEN

namespace MatrixFreeOperators
{
  /**
   * This class implements the operation of the action of a mass matrix on a
   * specified boundary ID. It is assumed that no constraints are relevant at
   * this particular boundary.
   *
   * Note that this class only supports the non-blocked vector variant of the
   * Base operator because only a single FEEvaluation object is used in the
   * apply function.
   */
  template <int dim,
            int n_components    = 1,
            typename VectorType = LinearAlgebra::distributed::Vector<double>,
            typename VectorizedArrayType =
              VectorizedArray<typename VectorType::value_type>>
  class BoundaryMassOperator : public Base<dim, VectorType, VectorizedArrayType>
  {
  public:
    /**
     * Number alias.
     */
    using value_type =
      typename Base<dim, VectorType, VectorizedArrayType>::value_type;

    /**
     * size_type needed for preconditioner classes.
     */
    using size_type =
      typename Base<dim, VectorType, VectorizedArrayType>::size_type;

    /**
     * FEFaceIntegrator definition used from the header file
     */
    using FEFaceIntegrator =
      FEFaceIntegrators<dim, n_components, value_type, VectorizedArrayType>;

    /**
     * Constructor.
     */
    BoundaryMassOperator(const types::boundary_id interface_id);

    /**
     * For preconditioning, we store a lumped mass matrix at the diagonal
     * entries.
     */
    virtual void
    compute_diagonal() override;

  private:
    /**
     * Applies the mass matrix operation at the defined boundary on an input
     * vector. It is assumed that the passed input and output vector are
     * correctly initialized using initialize_dof_vector().
     */
    virtual void
    apply_add(VectorType &dst, const VectorType &src) const override;

    virtual void
    Tapply_add(VectorType & /*dst*/, const VectorType & /*src*/) const override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    const types::boundary_id interface_id;
  };



  template <int dim,
            int n_components,
            typename VectorType,
            typename VectorizedArrayType>
  BoundaryMassOperator<dim, n_components, VectorType, VectorizedArrayType>::
    BoundaryMassOperator(const types::boundary_id interface_id)
    : Base<dim, VectorType, VectorizedArrayType>()
    , interface_id(interface_id)
  {}



  template <int dim,
            int n_components,
            typename VectorType,
            typename VectorizedArrayType>
  void
  BoundaryMassOperator<dim, n_components, VectorType, VectorizedArrayType>::
    compute_diagonal()
  {
    using Number = value_type;
    Assert((Base<dim, VectorType, VectorizedArrayType>::data.get() != nullptr),
           ExcNotInitialized());

    this->inverse_diagonal_entries =
      std::make_shared<DiagonalMatrix<VectorType>>();
    this->diagonal_entries = std::make_shared<DiagonalMatrix<VectorType>>();
    VectorType &inverse_diagonal_vector =
      this->inverse_diagonal_entries->get_vector();
    VectorType &diagonal_vector = this->diagonal_entries->get_vector();
    this->initialize_dof_vector(inverse_diagonal_vector);
    this->initialize_dof_vector(diagonal_vector);
    inverse_diagonal_vector = Number(1.);
    apply_add(diagonal_vector, inverse_diagonal_vector);

    this->set_constrained_entries_to_one(diagonal_vector);
    inverse_diagonal_vector = diagonal_vector;

    const unsigned int locally_owned_size =
      inverse_diagonal_vector.locally_owned_size();
    for (unsigned int i = 0; i < locally_owned_size; ++i)
      {
        if (inverse_diagonal_vector.local_element(i) > 0)
          inverse_diagonal_vector.local_element(i) =
            Number(1.) / inverse_diagonal_vector.local_element(i);
      }

    inverse_diagonal_vector.update_ghost_values();
    diagonal_vector.update_ghost_values();
  }



  template <int dim,
            int n_components,
            typename VectorType,
            typename VectorizedArrayType>
  void
  BoundaryMassOperator<dim, n_components, VectorType, VectorizedArrayType>::
    apply_add(VectorType &dst, const VectorType &src) const
  {
    const auto       data = Base<dim, VectorType, VectorizedArrayType>::data;
    FEFaceIntegrator phi(*data, true, this->selected_rows[0]);
    src.update_ghost_values();

    for (unsigned int face = data->n_inner_face_batches();
         face < data->n_boundary_face_batches() + data->n_inner_face_batches();
         ++face)
      {
        const auto boundary_id = data->get_boundary_id(face);
        // Only interfaces
        if (boundary_id != interface_id)
          continue;

        phi.reinit(face);
        phi.gather_evaluate(src, EvaluationFlags::values);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_value(phi.get_value(q), q);
        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    dst.compress(VectorOperation::add);
  }
} // namespace MatrixFreeOperators

DEAL_II_NAMESPACE_CLOSE
