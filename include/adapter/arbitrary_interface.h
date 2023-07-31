#pragma once

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <adapter/coupling_interface.h>

namespace Adapter
{
  using namespace dealii;
  /**
   * Derived class of the CouplingInterface, where data is written on an
   * arbitrary interface defined by other participants. Using this interface
   * requires a direct access to received meshes.
   */
  template <int dim, int data_dim, typename VectorizedArrayType>
  class ArbitraryInterface
    : public CouplingInterface<dim, data_dim, VectorizedArrayType>
  {
  public:
    ArbitraryInterface(
      std::shared_ptr<const MatrixFree<dim, double, VectorizedArrayType>> data,
      std::shared_ptr<precice::Participant> precice,
      std::string                           mesh_name,
      types::boundary_id                    interface_id)
      : CouplingInterface<dim, data_dim, VectorizedArrayType>(data,
                                                              precice,
                                                              mesh_name,
                                                              interface_id)
    {}

    /**
     * @brief define_mesh_vertices As opposed to the usual preCICE mappings
     *        and meshes, this function defines just a region of interest
     *        this process wants to access directly.
     */
    virtual void
    define_coupling_mesh() override;

    /**
     * @brief process_coupling_mesh This function is called after precice.initialize(),
     *        i.e., after the communication was established and the meshes were
     *        exchanged. Here, the relevant data of the received mesh is queried
     *        so that we receive the data location and the corresponding data
     *        IDs, which are required to write data to the received mesh. After
     *        receiving the data, it is filtered to the according to the local
     *        partition (see @ref filter_vertices_to_local_partition()) below.
     */
    virtual void
    process_coupling_mesh() override;

    /**
     * @brief write_data Evaluates the given @param data_vector at arbitrary
     *        data points (defined by coupling participants) and passes them
     *        to preCICE
     *
     * @param[in] data_vector The global (distributed) data vector containing
     *            the relevant coupling data (absolute displacement for FSI)
     */
    virtual void
    write_data(
      const LinearAlgebra::distributed::Vector<double> &data_vector) override;

  private:
    /**
     * @brief write_data_factory Factory function in order to write different
     *        data (gradients, values..) to preCICE
     *
     * @param[in] data_vector The data to be passed to preCICE (absolute
     *            displacement for FSI)
     * @param[in] flags UpdateFlags passed to the FEPointEvaluation object
     * @param[in] write_value A function which finally calculates the desired
     *            data and writes them to preCICE. Have a look at the examples
     *            given below. local index when iterating over all values
     */
    void
    write_data_factory(
      const LinearAlgebra::distributed::Vector<double> &data_vector,
      const UpdateFlags                                 flags,
      const std::function<void(FEPointEvaluation<data_dim, dim> &,
                               const Vector<double> &,
                               const size_t)>          &write_value) const;

    /**
     * @brief get_interface_type A function that returns a description of the
     *        used interface, which is printed on the console
     */
    virtual std::string
    get_interface_type() const override;

    /**
     * @brief filter_vertices_to_local_partition Given some arbitrary
     *        (coarse pre-filtered) vertices this function filters the given
     *        @param points_in according to the locally owned partition of the
     *        triangulation. For parallel computations, we need to make sure
     *        that only a single process (globally) writes data to a specific
     *        data point. However, preCICE might distribute points to multiple
     *        processes depending on the defined bounding-box and the
     *        safety-factor defined in the configuration file (overlapping
     *        partitions are also required for some mappings, which can be
     *        defined in addition to the direct access here). Therefore, this
     *        function filteres the received vertices using a consensus
     *        algorithm in order to ensure that only a unique process works on a
     *        specific node (globally). The consensus algorithm assigns always
     *        the lowest rank to vertices which might be owned by more than one
     *        process.
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
    filter_vertices_to_local_partition(const Mapping<dim>           &mapping,
                                       const Triangulation<dim>     &tria,
                                       const std::vector<Point<dim>> points_in,
                                       double tolerance = 1e-10);

    std::vector<int> interface_nodes_ids;
    std::vector<
      std::pair<typename Triangulation<dim>::active_cell_iterator, Point<dim>>>
      locally_relevant_points;
  };



  template <int dim, int data_dim, typename VectorizedArrayType>
  void
  ArbitraryInterface<dim, data_dim, VectorizedArrayType>::define_coupling_mesh()
  {
    Assert(!this->mesh_name.empty(), ExcNotInitialized());
    const auto &triangulation =
      this->mf_data->get_dof_handler().get_triangulation();

    // Get a bounding box which filter according to locally owned interface
    // cells
    const auto bounding_box_pair =
      GridTools::compute_bounding_box(triangulation, [this](const auto &cell) {
        bool cell_at_interface = false;
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary() &&
              face->boundary_id() == this->dealii_boundary_interface_id)
            cell_at_interface = true;

        return cell->is_locally_owned() && cell_at_interface;
      });

    // In case a bounding box collection supposed to be used in the future
    std::vector<BoundingBox<dim>> bounding_box{
      BoundingBox<dim>{bounding_box_pair}};
    // Currently only one bounding box supported
    Assert(bounding_box.size() <= 1, ExcInternalError());
    // Get the min and max per dim
    std::vector<double> precice_bounding_box;
    for (const auto &box : bounding_box)
      for (uint d = 0; d < dim; ++d)
        {
          // Emplace direction-wise
          precice_bounding_box.emplace_back(box.lower_bound(d));
          precice_bounding_box.emplace_back(box.upper_bound(d));
        }

    // Finally pass the bounding box to preCICE
    this->precice->setMeshAccessRegion(this->mesh_name, precice_bounding_box);
  }



  template <int dim, int data_dim, typename VectorizedArrayType>
  void
  ArbitraryInterface<dim, data_dim, VectorizedArrayType>::write_data(
    const LinearAlgebra::distributed::Vector<double> &data_vector)
  {
    // Write different data depending on the write_data_type
    switch (this->write_data_type)
      {
        case WriteDataType::values_on_other_mesh:
          write_data_factory(
            data_vector,
            UpdateFlags::update_values,
            [this](auto &fe_evaluator, auto &local_values, auto i) {
              fe_evaluator.evaluate(make_array_view(local_values),
                                    EvaluationFlags::values);
              const auto val = fe_evaluator.get_value(0);
              if constexpr (data_dim > 1)
                this->precice->writeData(this->mesh_name,
                                         this->write_data_name,
                                         {&interface_nodes_ids[i], 1},
                                         {val.begin_raw(),
                                          static_cast<std::size_t>(data_dim)});
              else
                this->precice->writeData(this->mesh_name,
                                         this->write_data_name,
                                         {&interface_nodes_ids[i], 1},
                                         {&val, 1});
            });
          break;
        case WriteDataType::gradients_on_other_mesh:
          write_data_factory(
            data_vector,
            UpdateFlags::update_gradients,
            [this](auto &fe_evaluator, auto &local_values, auto i) {
              Assert(data_dim == 1, ExcNotImplemented());
              fe_evaluator.evaluate(make_array_view(local_values),
                                    EvaluationFlags::gradients);
              const auto val = fe_evaluator.get_gradient(0);
              this->precice->writeData(this->mesh_name,
                                       this->write_data_name,
                                       {&interface_nodes_ids[i], 1},
                                       {val.begin_raw(),
                                        static_cast<std::size_t>(data_dim)});
            });
          break;
        default:
          AssertThrow(false, ExcNotImplemented());
      }
  }



  template <int dim, int data_dim, typename VectorizedArrayType>
  void
  ArbitraryInterface<dim, data_dim, VectorizedArrayType>::write_data_factory(
    const LinearAlgebra::distributed::Vector<double> &data_vector,
    const UpdateFlags                                 flags,
    const std::function<void(FEPointEvaluation<data_dim, dim> &,
                             const Vector<double> &,
                             const size_t)>          &write_value) const
  {
    Assert(!this->write_data_name.empty(), ExcNotInitialized());

    // This class allows to evaluate data at arbitrary points
    FEPointEvaluation<data_dim, dim> fe_evaluator(
      *(this->mf_data->get_mapping_info().mapping),
      this->mf_data->get_dof_handler().get_fe(),
      flags);

    Vector<double> local_values(
      this->mf_data->get_dof_handler().get_fe().n_dofs_per_cell());

    // TODO: We should combine multiple points belonging to the same cell here
    for (size_t i = 0; i < interface_nodes_ids.size(); ++i)
      {
        // locally_relevant_points contains our filtered vertices
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

        // Get the relevant DoF values and write them to preCICE
        dof_cell->get_dof_values(data_vector, local_values);
        write_value(fe_evaluator, local_values, i);
      }
  }



  template <int dim, int data_dim, typename VectorizedArrayType>
  void
  ArbitraryInterface<dim, data_dim, VectorizedArrayType>::
    process_coupling_mesh()
  {
    Assert(!this->mesh_name.empty(), ExcNotInitialized());

    // Ask preCICE for the relevant mesh size we work (preliminary) on
    const int received_mesh_size =
      this->precice->getMeshVertexSize(this->mesh_name);

    // Allocate a vector for the vertices and the corresponding IDs
    std::vector<double> received_coordinates(received_mesh_size * dim);
    interface_nodes_ids.resize(received_mesh_size);

    // ... and let preCICE fill the data containers
    this->precice->getMeshVertexIDsAndCoordinates(this->mesh_name,
                                                  interface_nodes_ids,
                                                  received_coordinates);

    // Transform the received points into a more deal.II like format, which is
    // vector of points
    std::vector<Point<dim>> received_points(received_mesh_size);
    for (int i = 0; i < received_mesh_size; ++i)
      for (int d = 0; d < dim; ++d)
        {
          AssertIndexRange(i * dim + d, received_coordinates.size());
          received_points[i][d] = received_coordinates[i * dim + d];
        }

    // TODO: Maybe perform some coarse pre-filtering here
    // Now filter the received points using the consensus algorithm (have a look
    // at the description of the function)
    locally_relevant_points = filter_vertices_to_local_partition(
      *(this->mf_data->get_mapping_info().mapping),
      this->mf_data->get_dof_handler(0).get_triangulation(),
      received_points);

    // Some consistency checks: we can only write data using this interface,
    // reading doesn't make sense
    Assert(this->read_data_name.empty(), ExcInternalError());
    Assert(!this->write_data_name.empty(), ExcInternalError());

    this->print_info(false, interface_nodes_ids.size());
  }


  template <int dim, int data_dim, typename VectorizedArrayType>
  std::string
  ArbitraryInterface<dim, data_dim, VectorizedArrayType>::get_interface_type()
    const
  {
    return "arbitrary nodes defined by the coupling partner ";
  }



  template <int dim, int data_dim, typename VectorizedArrayType>
  std::vector<
    std::pair<typename Triangulation<dim>::active_cell_iterator, Point<dim>>>
  ArbitraryInterface<dim, data_dim, VectorizedArrayType>::
    filter_vertices_to_local_partition(const Mapping<dim>           &mapping,
                                       const Triangulation<dim>     &tria,
                                       const std::vector<Point<dim>> points_in,
                                       double                        tolerance)
  {
    std::vector<
      std::pair<typename Triangulation<dim>::active_cell_iterator, Point<dim>>>
      unique_points;

    // Set up the relevant data objects
    GridTools::Cache<dim>                             cache(tria, mapping);
    typename Triangulation<dim>::active_cell_iterator cell_hint;
    const std::vector<bool>                           marked_vertices;

    const unsigned int my_rank =
      Utilities::MPI::this_mpi_process(tria.get_communicator());
    std::vector<int> relevant_interface_ids;

    // Loop over all received points
    for (size_t i = 0; i < points_in.size(); ++i)
      {
        // First look for all cells around the given point
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

        // Afterwards, compare the globally unique subdomain ID of all cells
        // around this points in order to find the cell with the lowest
        // subdomain_id, i.e. the lowest rank owning this cell
        for (const auto &cell : active_cells_around_point)
          lowest_rank = std::min(lowest_rank, cell.first->subdomain_id());

        // If the current rank is the lowest rank, it works on this point (the
        // point is added to the set), otherwise the point is skipped and
        // filtered out
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
    // Finally, assign the relevant IDs to the class member and return the
    // unique data set
    interface_nodes_ids = std::move(relevant_interface_ids);
    Assert(interface_nodes_ids.size() == unique_points.size(),
           ExcInternalError());
    return unique_points;
  }

} // namespace Adapter
