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
   */
  template <int dim,
            int fe_degree,
            int n_qpoints_1d,
            typename VectorizedArrayType>
  class dealiiInterface : public CouplingInterface<dim, VectorizedArrayType>
  {
  public:
    dealiiInterface(
      std::shared_ptr<MatrixFree<dim, double, VectorizedArrayType>> data,
      std::shared_ptr<precice::SolverInterface>                     precice,
      const std::string                                             mesh_name,
      const types::boundary_id interface_id,
      const int                mf_dof_index,
      const int                mf_quad_index)
      : CouplingInterface<dim, VectorizedArrayType>(data,
                                                    precice,
                                                    mesh_name,
                                                    interface_id)
      , mf_dof_index(mf_dof_index)
      , mf_quad_index(mf_quad_index)
    {}

    /// Alias for the face integrator
    using FEFaceIntegrator = FEFaceEvaluation<dim,
                                              fe_degree,
                                              n_qpoints_1d,
                                              dim,
                                              double,
                                              VectorizedArrayType>;
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
     *            displacement for FSI)
     */
    virtual void
    write_data(
      const LinearAlgebra::distributed::Vector<double> &data_vector) override;


    /**
     * @brief read_on_quadrature_point Returns the dim dimensional read data
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
    virtual Tensor<1, dim, VectorizedArrayType>
    read_on_quadrature_point(const unsigned int id_number,
                             const unsigned int active_faces) const override;

  private:
    /// The preCICE IDs
    std::vector<std::array<int, VectorizedArrayType::size()>>
      interface_nodes_ids;

    bool interface_is_defined = false;
    /// Indices related to the FEEvaluation (have a look at the initialization
    /// of the MatrixFree)
    const int mf_dof_index;
    const int mf_quad_index;

    virtual std::string
    get_interface_type() const override;
  };



  template <int dim,
            int fe_degree,
            int n_qpoints_1d,
            typename VectorizedArrayType>
  void
  dealiiInterface<dim, fe_degree, n_qpoints_1d, VectorizedArrayType>::
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
    FEFaceIntegrator phi(*this->mf_data, true, mf_dof_index, mf_quad_index);
    Assert(phi.fast_evaluation_supported(fe_degree, n_qpoints_1d),
           ExcMessage("Fast evaluation is not supported."));

    std::array<double, dim * VectorizedArrayType::size()> unrolled_vertices;
    std::array<int, VectorizedArrayType::size()>          node_ids;
    unsigned int                                          size = 0;
    // Loop over all boundary faces
    for (unsigned int face = this->mf_data->n_inner_face_batches();
         face < this->mf_data->n_boundary_face_batches() +
                  this->mf_data->n_inner_face_batches();
         ++face)
      {
        const auto boundary_id = this->mf_data->get_boundary_id(face);

        // Only for interface nodes
        if (boundary_id != this->dealii_boundary_interface_id)
          continue;

        phi.reinit(face);
        const int active_faces =
          this->mf_data->n_active_entries_per_face_batch(face);

        // Loop over all quadrature points and pass the vertices to preCICE
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto local_vertex = phi.quadrature_point(q);

            // Transform Point<Vectorized> into preCICE conform format
            // We store here also the potential 'dummy'/empty lanes (not only
            // active_faces), but it allows us to use a static loop as well as a
            // static array for the indices
            for (int d = 0; d < dim; ++d)
              for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
                unrolled_vertices[d + dim * v] = local_vertex[d][v];

            this->precice->setMeshVertices(this->mesh_id,
                                           active_faces,
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
  }



  template <int dim,
            int fe_degree,
            int n_qpoints_1d,
            typename VectorizedArrayType>
  void
  dealiiInterface<dim, fe_degree, n_qpoints_1d, VectorizedArrayType>::
    write_data(const LinearAlgebra::distributed::Vector<double> &data_vector)
  {
    Assert(this->write_data_id != -1, ExcNotInitialized());
    Assert(interface_is_defined, ExcNotInitialized());
    // Similar as in define_coupling_mesh
    FEFaceIntegrator phi(*this->mf_data, true, mf_dof_index, mf_quad_index);

    // In order to unroll the vectorization
    std::array<double, dim * VectorizedArrayType::size()> unrolled_local_data;
    auto index = interface_nodes_ids.begin();

    // Loop over all faces
    for (unsigned int face = this->mf_data->n_inner_face_batches();
         face < this->mf_data->n_boundary_face_batches() +
                  this->mf_data->n_inner_face_batches();
         ++face)
      {
        const auto boundary_id = this->mf_data->get_boundary_id(face);

        // Only for interface nodes
        if (boundary_id != this->dealii_boundary_interface_id)
          continue;

        // Read and interpolate
        phi.reinit(face);
        phi.gather_evaluate(data_vector, EvaluationFlags::values);
        const int active_faces =
          this->mf_data->n_active_entries_per_face_batch(face);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto local_data = phi.get_value(q);
            Assert(index != interface_nodes_ids.end(), ExcInternalError());

            // Transform Tensor<1,dim,VectorizedArrayType> into preCICE conform
            // format
            for (int d = 0; d < dim; ++d)
              for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
                unrolled_local_data[d + dim * v] = local_data[d][v];

            this->precice->writeBlockVectorData(this->write_data_id,
                                                active_faces,
                                                index->data(),
                                                unrolled_local_data.data());
            ++index;
          }
      }
  }



  template <int dim,
            int fe_degree,
            int n_qpoints_1d,
            typename VectorizedArrayType>
  inline Tensor<1, dim, VectorizedArrayType>
  dealiiInterface<dim, fe_degree, n_qpoints_1d, VectorizedArrayType>::
    read_on_quadrature_point(const unsigned int id_number,
                             const unsigned int active_faces) const
  {
    // Assert input
    Assert(active_faces <= VectorizedArrayType::size(), ExcInternalError());
    AssertIndexRange(id_number, interface_nodes_ids.size());
    Assert(this->read_data_id != -1, ExcNotInitialized());

    Tensor<1, dim, VectorizedArrayType>                   dealii_data;
    std::array<double, dim * VectorizedArrayType::size()> precice_data;
    const auto vertex_ids = &interface_nodes_ids[id_number];
    this->precice->readBlockVectorData(this->read_data_id,
                                       active_faces,
                                       vertex_ids->data(),
                                       precice_data.data());
    // Transform back to Tensor format
    for (int d = 0; d < dim; ++d)
      for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
        dealii_data[d][v] = precice_data[d + dim * v];

    return dealii_data;
  }


  template <int dim,
            int fe_degree,
            int n_qpoints_1d,
            typename VectorizedArrayType>
  std::string
  dealiiInterface<dim, fe_degree, n_qpoints_1d, VectorizedArrayType>::
    get_interface_type() const
  {
    return "quadrature points using matrix-free index " +
           Utilities::to_string(mf_quad_index);
  }

  // TODO
  //  get_mesh_stats()
} // namespace Adapter
