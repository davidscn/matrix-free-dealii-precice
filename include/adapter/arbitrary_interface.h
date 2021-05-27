#pragma once

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <adapter/coupling_interface.h>

namespace Adapter
{
  using namespace dealii;
  /**
   * Derived class of the CouplingInterface: where data is written on an
   * arbitrary
   */
  template <int dim, typename VectorizedArrayType>
  class ArbitraryInterface : public CouplingInterface<dim, VectorizedArrayType>
  {
  public:
    ArbitraryInterface(
      std::shared_ptr<MatrixFree<dim, double, VectorizedArrayType>> data,
      std::shared_ptr<precice::SolverInterface>                     precice,
      const std::string                                             mesh_name,
      const types::boundary_id interface_id)
      : CouplingInterface<dim, VectorizedArrayType>(data,
                                                    precice,
                                                    mesh_name,
                                                    interface_id)
    {}

    /**
     * @brief write_mapped_data Evaluates the given @param data at arbitrary
     *        data points (defined by coupling participants) and passes them
     *        to preCICE
     *
     * @param[in] data The global (distributed) data vector to be passed to
     *            preCICE (absolute displacement for FSI)
     */
    virtual void
    define_coupling_mesh() override;

    /**
     * @brief process_coupling_mesh Compute the local partitioning
     */
    virtual void
    process_coupling_mesh() override;

    /**
     * @brief write_mapped_data Evaluates the given @param data at arbitrary
     *        data points (defined by coupling participants) and passes them
     *        to preCICE
     *
     * @param[in] data The global (distributed) data vector to be passed to
     *            preCICE (absolute displacement for FSI)
     */
    virtual void
    write_data(
      const LinearAlgebra::distributed::Vector<double> &data_vector) override;

    /**
     * @brief read_on_quadrature_point Not implemented in this derived class
     */
    virtual Tensor<1, dim, VectorizedArrayType>
    read_on_quadrature_point(const unsigned int,
                             const unsigned int) const override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    /**
     * @brief finish_initialization Handles received vertices
     */
    void
    finish_initialization();

  private:
    virtual std::string
    get_interface_type() const override;

    /**
     * @brief filter_vertices_to_local_partition Given some arbitrary
     *        (coarse pre-filtered) vertices this function filters the given
     *        @param points_in according to the locally owned partition of the
     *        triangulation. In case of parallel distributed computations, a
     *        consensus algorithm is applied in order to ensure that only a
     *        unique process works on a specific node (globally). The consensus
     *        algorithm assigns always the lowest rank to vertices which might
     *        be owned by more than one process
     *
     * @param[in] mapping The underlying mapping.
     * @param[in] tria The underlying triangulation.
     * @param[in] points_in The cloud of arbitrary points obtained by preCICE.
     * @param[in] tolerance  Tolerance in terms of unit cell coordinates.
     *            Depending on the problem, it might be necessary to adjust the
     *            tolerance in order to be able to identify a cell. Have a look
     *            at 'find_active_cell_around_point'
     *
     * @return The vector of pairs containing the relevent cell iterator and the point
     *         location in the reference frame
     */
    std::vector<
      std::pair<typename Triangulation<dim>::active_cell_iterator, Point<dim>>>
    filter_vertices_to_local_partition(const Mapping<dim> &          mapping,
                                       const Triangulation<dim> &    tria,
                                       const std::vector<Point<dim>> points_in,
                                       double tolerance = 1e-10);

    std::vector<int> interface_nodes_ids;
    std::vector<
      std::pair<typename Triangulation<dim>::active_cell_iterator, Point<dim>>>
      locally_relevant_points;
  };



  template <int dim, typename VectorizedArrayType>
  void
  ArbitraryInterface<dim, VectorizedArrayType>::define_coupling_mesh()
  {
    Assert(this->mesh_id != -1, ExcNotInitialized());
    const auto &triangulation =
      this->mf_data->get_dof_handler().get_triangulation();

    const auto bounding_box = GridTools::compute_mesh_predicate_bounding_box(
      triangulation,
      IteratorFilters::LocallyOwnedCell(),
      /*refinement-level*/ 1,
      /*merge*/ true,
      /*max-boxes*/ 1);

    // min and max per dim
    std::vector<double> precice_bounding_box;
    for (uint d = 0; d < dim; ++d)
      {
        precice_bounding_box.emplace_back(bounding_box[0].lower_bound(d));
        precice_bounding_box.emplace_back(bounding_box[0].upper_bound(d));
      }
    Assert(precice_bounding_box.size() == (2 * dim), ExcInternalError());
    this->precice->setBoundingBoxes(this->mesh_id,
                                    precice_bounding_box.data(),
                                    1);
  }



  template <int dim, typename VectorizedArrayType>
  void
  ArbitraryInterface<dim, VectorizedArrayType>::write_data(
    const LinearAlgebra::distributed::Vector<double> &data_vector)
  {
    Assert(this->write_data_id != -1, ExcNotInitialized());

    FEPointEvaluation<dim, dim> fe_evaluator(
      *(this->mf_data->get_mapping_info().mapping),
      this->mf_data->get_dof_handler().get_fe(),
      UpdateFlags::update_values);

    Vector<double> local_values(
      this->mf_data->get_dof_handler().get_fe().n_dofs_per_cell());

    // TODO: We should combine multiple points belonging to the same cell here
    for (size_t i = 0; i < interface_nodes_ids.size(); ++i)
      {
        AssertIndexRange(i, locally_relevant_points.size());
        const auto              point = locally_relevant_points[i];
        std::vector<Point<dim>> point_vec{point.second};
        fe_evaluator.reinit(point.first, point_vec);

        // Convert TriaIterator to DoFCellAccessor
        TriaIterator<DoFCellAccessor<dim, dim, false>> dof_cell(
          &(this->mf_data->get_dof_handler().get_triangulation()),
          point.first->level(),
          point.first->index(),
          &(this->mf_data->get_dof_handler()));

        dof_cell->get_dof_values(data_vector, local_values);
        fe_evaluator.evaluate(make_array_view(local_values),
                              EvaluationFlags::values);

        const auto val = fe_evaluator.get_value(0);
        this->precice->writeVectorData(this->write_data_id,
                                       interface_nodes_ids[i],
                                       val.begin_raw());
      }
  }



  template <int dim, typename VectorizedArrayType>
  void
  ArbitraryInterface<dim, VectorizedArrayType>::process_coupling_mesh()
  {
    Assert(this->mesh_id != -1, ExcNotInitialized());

    // For solver mapping
    const int received_mesh_size =
      this->precice->getMeshVertexSize(this->mesh_id);

    // Allocate a vector containing the vertices
    std::vector<double> received_coordinates(received_mesh_size * dim);
    interface_nodes_ids.resize(received_mesh_size);

    this->precice->getMeshVerticesWithIDs(this->mesh_id,
                                          received_mesh_size,
                                          interface_nodes_ids.data(),
                                          received_coordinates.data());

    // Transform the received points into a vector of points
    std::vector<Point<dim>> received_points(received_mesh_size);
    for (int i = 0; i < received_mesh_size; ++i)
      for (int d = 0; d < dim; ++d)
        {
          AssertIndexRange(i * dim + d, received_coordinates.size());
          received_points[i][d] = received_coordinates[i * dim + d];
        }

    // TODO: Maybe perform some coarse pre-filtering here
    locally_relevant_points = filter_vertices_to_local_partition(
      *(this->mf_data->get_mapping_info().mapping),
      this->mf_data->get_dof_handler(0).get_triangulation(),
      received_points);

    Assert(this->read_data_id == -1, ExcInternalError());
    Assert(this->write_data_id != -1, ExcInternalError());

    this->print_info(false, interface_nodes_ids.size());
  }


  template <int dim, typename VectorizedArrayType>
  std::string
  ArbitraryInterface<dim, VectorizedArrayType>::get_interface_type() const
  {
    return "arbitrary nodes defined by the coupling partner ";
  }



  template <int dim, typename VectorizedArrayType>
  std::vector<
    std::pair<typename Triangulation<dim>::active_cell_iterator, Point<dim>>>
  ArbitraryInterface<dim, VectorizedArrayType>::
    filter_vertices_to_local_partition(const Mapping<dim> &          mapping,
                                       const Triangulation<dim> &    tria,
                                       const std::vector<Point<dim>> points_in,
                                       double                        tolerance)
  {
    std::vector<
      std::pair<typename Triangulation<dim>::active_cell_iterator, Point<dim>>>
      unique_points;

    GridTools::Cache<dim>                             cache(tria, mapping);
    typename Triangulation<dim>::active_cell_iterator cell_hint;
    const std::vector<bool>                           marked_vertices;

    const unsigned int my_rank =
      Utilities::MPI::this_mpi_process(tria.get_communicator());
    std::vector<int> relevant_interface_ids;

    for (size_t i = 0; i < points_in.size(); ++i)
      {
        const auto &point      = points_in[i];
        const auto  first_cell = GridTools::find_active_cell_around_point(
          cache, point, cell_hint, marked_vertices, tolerance);

        cell_hint = first_cell.first;

        if (cell_hint.state() != IteratorState::valid)
          continue;

        const auto active_cells_around_point =
          GridTools::find_all_active_cells_around_point(
            cache.get_mapping(),
            cache.get_triangulation(),
            point,
            tolerance,
            first_cell);

        unsigned int lowest_rank = numbers::invalid_unsigned_int;

        for (const auto &cell : active_cells_around_point)
          lowest_rank = std::min(lowest_rank, cell.first->subdomain_id());

        if (lowest_rank != my_rank)
          continue;

        for (const auto &cell : active_cells_around_point)
          if (cell.first->is_locally_owned())
            {
              unique_points.emplace_back(cell);
              relevant_interface_ids.emplace_back(interface_nodes_ids[i]);
              break;
            }
      }
    interface_nodes_ids = std::move(relevant_interface_ids);
    Assert(interface_nodes_ids.size() == unique_points.size(),
           ExcInternalError());
    return unique_points;
  }

} // namespace Adapter
