#pragma once

#include <deal.II/matrix_free/fe_evaluation.h>

#include <adapter/coupling_interface.h>

namespace Adapter
{
  using namespace dealii;

  /**
   * Derived class of the CouplingInterface: the classical coupling approach,
   * where each participant defines an interface based on the locally owned
   * triangulation. Here, quadrature points are used for reading and writing.
   * data_dim is equivalent to n_components, indicating the type of your data in
   * the preCICE sense (Vector vs Scalar)
   */
  template <int dim, int data_dim, typename VectorizedArrayType>
  class OverlappingInterface
    : public CouplingInterface<dim, data_dim, VectorizedArrayType>
  {
  public:
    OverlappingInterface(
      std::shared_ptr<const MatrixFree<dim, double, VectorizedArrayType>> data,
      std::shared_ptr<precice::SolverInterface> precice,
      std::string                               mesh_name,
      types::boundary_id                        interface_id,
      int                                       mf_dof_index,
      int                                       mf_quad_index)
      : CouplingInterface<dim, data_dim, VectorizedArrayType>(data,
                                                              precice,
                                                              mesh_name,
                                                              interface_id)
      , mf_dof_index(mf_dof_index)
      , mf_quad_index(mf_quad_index)
    {}

    /// Alias as defined in the base class
    using FEFaceIntegrator =
      typename CouplingInterface<dim, data_dim, VectorizedArrayType>::
        FEFaceIntegrator;
    using FECellIntegrator =
      typename CouplingInterface<dim, data_dim, VectorizedArrayType>::
        FECellIntegrator;
    using value_type =
      typename CouplingInterface<dim, data_dim, VectorizedArrayType>::
        value_type;
    /**
     * @brief define_mesh_vertices Define a vertex coupling mesh for preCICE
     *        coupling the classical preCICE way
     */
    virtual void
    define_coupling_mesh() override;

    /**
     * @brief write_data Evaluates the given @param data at the
     *        quadrature_points of the defined mesh and passes
     *        them to preCICE
     *
     * @param[in] data_vector The data to be passed to preCICE (absolute
     *            displacement for FSI). Note that the data_vector needs to
     *            contain valid ghost values for parallel runs, i.e.
     *            update_ghost_values must be calles before
     */
    virtual void
    write_data(
      const LinearAlgebra::distributed::Vector<double> &data_vector) override;

    /**
     * @brief read_on_quadrature_point Returns the data_dim dimensional read data
     *        given the ID of the interface node we want to access.
     *
     * @param[in]  id_number Number of the quadrature point (counting from zero
     *             to the total number of quadrature point this process works
     *             on) we want to access. Here, we explicitly rely on the order
     *             we used when the coupling mesh was defined (see
     *             @ref define_coupling_mesh()). Note that the MatrixFree::loop
     *             functions may process cells in a different order than the
     *             manual loop during the mesh definition. Using an incremental
     *             iterator for the id_number would consequently fail.
     * @param[in]  active_faces Number of active faces the matrix-free object
     *             works on
     *
     * @return dim dimensional data associated to the interface node
     */
    virtual value_type
    read_on_quadrature_point(const unsigned int id_number,
                             const unsigned int active_faces) const override;

  private:
    /**
     * @brief write_data_factory Factory function in order to write different
     *        data (gradients, values..) to preCICE
     *
     * @param[in] data_vector The data to be passed to preCICE (absolute
     *            displacement for FSI)
     * @param[in] flags
     * @param[in] get_write_value
     */
    void
    write_data_factory(
      const LinearAlgebra::distributed::Vector<double> &data_vector,
      const EvaluationFlags::EvaluationFlags            flags,
      const std::function<value_type(FECellIntegrator &, unsigned int)>
        &get_write_value);

    /// The preCICE IDs
    std::vector<std::array<int, VectorizedArrayType::size()>>
      interface_nodes_ids;

    bool interface_is_defined = false;
    /// Indices related to the FEEvaluation (have a look at the initialization
    /// of the MatrixFree)
    const int mf_dof_index;
    const int mf_quad_index;

    // percentage of quads we want to send
    const double x_threshold = 1;
    virtual std::string
    get_interface_type() const override;
  };



  template <int dim, int data_dim, typename VectorizedArrayType>
  void
  OverlappingInterface<dim, data_dim, VectorizedArrayType>::
    define_coupling_mesh()
  {
    Assert(this->mesh_id != -1, ExcNotInitialized());

    // In order to avoid that we define the interface multiple times when reader
    // and writer refer to the same object
    if (interface_is_defined)
      return;

    // Initial guess: half of the boundary is part of the coupling interface
    interface_nodes_ids.reserve(this->mf_data->n_boundary_face_batches() * 0.5);

    // Set up data structures
    FECellIntegrator phi(*this->mf_data, mf_dof_index, mf_quad_index);
    std::array<double, dim * VectorizedArrayType::size()> unrolled_vertices;
    std::array<int, VectorizedArrayType::size()>          node_ids;
    unsigned int                                          size = 0;

    // Loop over all boundary faces
    for (unsigned int cell = 0; cell < this->mf_data->n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        const int active_cells =
          this->mf_data->n_active_entries_per_cell_batch(cell);

        // Loop over all quadrature points and pass the vertices to preCICE
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            // Only for interface nodes
            const auto local_vertex = phi.quadrature_point(q);

            bool q_is_relevant = false;

            for (int i = 0; i < active_cells; ++i)
              {
                double val = local_vertex[0][i];
                if ((1 - x_threshold) < val && val < (1 + x_threshold))
                  {
                    q_is_relevant = true;
                  }
              }

            if (!q_is_relevant)
              continue;

            // Transform Point<Vectorized> into preCICE conform format
            // We store here also the potential 'dummy'/empty lanes (not only
            // active_faces), but it allows us to use a static loop as well as a
            // static array for the indices
            for (int d = 0; d < dim; ++d)
              for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
                unrolled_vertices[d + dim * v] = local_vertex[d][v];

            this->precice->setMeshVertices(this->mesh_id,
                                           active_cells,
                                           unrolled_vertices.data(),
                                           node_ids.data());
            interface_nodes_ids.emplace_back(node_ids);
            ++size;
          }
      }
    // resize the IDs in case the initial guess was too large
    interface_nodes_ids.resize(size);
    interface_is_defined = true;
    // Consistency check: the number of IDs we stored is equal or greater than
    // the IDs preCICE knows
    Assert(size * VectorizedArrayType::size() >=
             static_cast<unsigned int>(
               this->precice->getMeshVertexSize(this->mesh_id)),
           ExcInternalError());

    if (this->read_data_id != -1)
      this->print_info(true, this->precice->getMeshVertexSize(this->mesh_id));
    if (this->write_data_id != -1)
      this->print_info(false, this->precice->getMeshVertexSize(this->mesh_id));
  }


  template <int dim, int data_dim, typename VectorizedArrayType>
  void
  OverlappingInterface<dim, data_dim, VectorizedArrayType>::write_data(
    const LinearAlgebra::distributed::Vector<double> &data_vector)
  {
    switch (this->write_data_type)
      {
        case WriteDataType::values_on_quads:
          write_data_factory(data_vector,
                             EvaluationFlags::values,
                             [](auto &phi, auto q_point) {
                               return phi.get_value(q_point);
                             });
          break;
        case WriteDataType::normal_gradients_on_quads:
          write_data_factory(data_vector,
                             EvaluationFlags::gradients,
                             [](auto &phi, auto q_point) {
                               Tensor<1, dim> x_normal;
                               x_normal[0] = 1;
                               return phi.get_gradient(q_point) * x_normal;
                             });
          break;
        default:
          AssertThrow(false, ExcNotImplemented());
      }
  }



  template <int dim, int data_dim, typename VectorizedArrayType>
  void
  OverlappingInterface<dim, data_dim, VectorizedArrayType>::write_data_factory(
    const LinearAlgebra::distributed::Vector<double> &data_vector,
    const EvaluationFlags::EvaluationFlags            flags,
    const std::function<value_type(FECellIntegrator &, unsigned int)>
      &get_write_value)
  {
    Assert(this->write_data_id != -1, ExcNotInitialized());
    Assert(interface_is_defined, ExcNotInitialized());
    // Similar as in define_coupling_mesh
    FECellIntegrator phi(*this->mf_data, mf_dof_index, mf_quad_index);

    // In order to unroll the vectorization
    std::array<double, data_dim * VectorizedArrayType::size()>
      unrolled_local_data;
    (void)unrolled_local_data;

    auto index = interface_nodes_ids.begin();

    // Loop over all faces
    for (unsigned int cell = 0; cell < this->mf_data->n_cell_batches(); ++cell)
      {
        // Read and interpolate
        phi.reinit(cell);
        phi.read_dof_values_plain(data_vector);
        phi.evaluate(flags);
        const int active_cells =
          this->mf_data->n_active_entries_per_cell_batch(cell);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto local_vertex = phi.quadrature_point(q);

            bool q_is_relevant = false;

            for (int i = 0; i < active_cells; ++i)
              {
                double val = local_vertex[0][i];
                if ((1 - x_threshold) < val && val < (1 + x_threshold))
                  {
                    q_is_relevant = true;
                  }
              }

            if (!q_is_relevant)
              continue;

            const auto local_data = get_write_value(phi, q);
            Assert(index != interface_nodes_ids.end(), ExcInternalError());

            // Constexpr evaluation required in order to comply with the
            // compiler here
            if constexpr (data_dim > 1)
              {
                // Transform Tensor<1,dim,VectorizedArrayType> into preCICE
                // conform format
                for (int d = 0; d < data_dim; ++d)
                  for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
                    unrolled_local_data[d + data_dim * v] = local_data[d][v];

                this->precice->writeBlockVectorData(this->write_data_id,
                                                    active_cells,
                                                    index->data(),
                                                    unrolled_local_data.data());
              }
            else
              {
                this->precice->writeBlockScalarData(this->write_data_id,
                                                    active_cells,
                                                    index->data(),
                                                    &local_data[0]);
              }
            ++index;
          }
      }
  }



  template <int dim, int data_dim, typename VectorizedArrayType>
  inline typename OverlappingInterface<dim, data_dim, VectorizedArrayType>::
    value_type
    OverlappingInterface<dim, data_dim, VectorizedArrayType>::
      read_on_quadrature_point(const unsigned int id_number,
                               const unsigned int active_faces) const
  {
    // Assert input
    Assert(active_faces <= VectorizedArrayType::size(), ExcInternalError());
    AssertIndexRange(id_number, interface_nodes_ids.size());
    Assert(this->read_data_id != -1, ExcNotInitialized());

    value_type dealii_data;
    const auto vertex_ids = &interface_nodes_ids[id_number];
    // Vector valued case
    if constexpr (data_dim > 1)
      {
        std::array<double, data_dim * VectorizedArrayType::size()> precice_data;
        this->precice->readBlockVectorData(this->read_data_id,
                                           active_faces,
                                           vertex_ids->data(),
                                           precice_data.data());
        // Transform back to Tensor format
        for (int d = 0; d < data_dim; ++d)
          for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
            dealii_data[d][v] = precice_data[d + data_dim * v];
      }
    else
      {
        // Scalar case
        this->precice->readBlockScalarData(this->read_data_id,
                                           active_faces,
                                           vertex_ids->data(),
                                           &dealii_data[0]);
      }
    return dealii_data;
  }


  template <int dim, int data_dim, typename VectorizedArrayType>
  std::string
  OverlappingInterface<dim, data_dim, VectorizedArrayType>::get_interface_type()
    const
  {
    return "overlapping (" + Utilities::to_string(x_threshold) +
           ") quadrature points using matrix-free quad index " +
           Utilities::to_string(mf_quad_index);
  }

  // TODO
  //  get_mesh_stats()
} // namespace Adapter
