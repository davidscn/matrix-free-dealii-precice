#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <adapter/dealii_interface.h>
#include <precice/SolverInterface.hpp>
#include <q_equidistant.h>

#include <ostream>

namespace Adapter
{
  using namespace dealii;

  /**
   * The Adapter class keeps together with the CouplingInterfaes all
   * functionalities to couple deal.II to other solvers with preCICE i.e. data
   * structures are set up, necessary information is passed to preCICE etc.
   */
  template <int dim,
            int fe_degree,
            typename VectorType,
            typename VectorizedArrayType = VectorizedArray<double>>
  class Adapter
  {
  public:
    /**
     * @brief      Constructor, which sets up the precice Solverinterface
     *
     * @param[in]  parameters Parameter class, which hold the data specified
     *             in the parameters.prm file
     * @param[in]  dealii_boundary_interface_id Boundary ID of the
     *             triangulation, which is associated with the coupling
     *             interface.
     * @param[in]  data The applied matrix-free object
     * @param[in]  dof_index Index of the relevant dof_handler in the
     *             corresponding MatrixFree object
     * @param[in]  read_quad_index Index of the quadrature formula in the
     *             corresponding MatrixFree object which should be used for data
     *             reading
     * @param[in]  write_quad_index Index of the quadrature formula in the
     *             corresponding MatrixFree object which should be used for data
     *             writing
     */
    template <typename ParameterClass>
    Adapter(const ParameterClass &parameters,
            const unsigned int    dealii_boundary_interface_id,
            std::shared_ptr<MatrixFree<dim, double, VectorizedArrayType>> data,
            const unsigned int dof_index        = 0,
            const unsigned int read_quad_index  = 0,
            const unsigned int write_quad_index = 1);

    /**
     * @brief      Initializes preCICE and passes all relevant data to preCICE
     *
     * @param[in]  dealii_to_precice Data, which should be given to preCICE and
     *             exchanged with other participants. Wether this data is
     *             required already in the beginning depends on your
     *             individual configuration and preCICE determines it
     *             automatically. In many cases, this data will just represent
     *             your initial condition.
     */
    void
    initialize(const VectorType &dealii_to_precice);

    /**
     * @brief      Advances preCICE after every timestep, converts data formats
     *             between preCICE and dealii
     *
     * @param[in]  dealii_to_precice Same data as in @p initialize_precice() i.e.
     *             data, which should be given to preCICE after each time step
     *             and exchanged with other participants.
     * @param[in]  computed_timestep_length Length of the timestep used by
     *             the solver.
     */
    void
    advance(const VectorType &dealii_to_precice,
            const double      computed_timestep_length);

    /**
     * @brief      Saves current state of time dependent variables in case of an
     *             implicit coupling
     *
     * @param[in]  state_variables Vector containing all variables to store as
     *             reference
     *
     * @note       This function only makes sense, if it is used with
     *             @p reload_old_state_if_required. Therefore, the order, in which the
     *             variables are passed into the vector must be the same for
     *             both functions.
     */
    void
    save_current_state_if_required(const std::function<void()> &save_state);

    /**
     * @brief      Reloads the previously stored variables in case of an implicit
     *             coupling. The current implementation supports subcycling,
     *             i.e. previously refers o the last time
     *             @p save_current_state_if_required() has been called.
     *
     * @param[out] state_variables Vector containing all variables to reload
     *             as reference
     *
     * @note       This function only makes sense, if the state variables have been
     *             stored by calling @p save_current_state_if_required. Therefore,
     *             the order, in which the variables are passed into the
     *             vector must be the same for both functions.
     */
    void
    reload_old_state_if_required(const std::function<void()> &reload_old_state);

    /**
     * @brief Public API adapter method, which calls the respective implementation
     *        in derived classes of the CouplingInterface. Have a look at the
     *        documentation there.
     */
    Tensor<1, dim, VectorizedArrayType>
    read_on_quadrature_point(const unsigned int id_number,
                             const unsigned int active_faces) const;

    /**
     * @brief is_coupling_ongoing Calls the preCICE API function isCouplingOnGoing
     *
     * @return returns true if the coupling has not yet been finished
     */
    bool
    is_coupling_ongoing() const;

    /**
     * @brief is_time_window_complete Calls the preCICE API function isTimeWindowComplete
     *
     * @return returns true if the coupling time window has been completed in the current
     *         iteration
     */
    bool
    is_time_window_complete() const;


    /// Boundary ID of the deal.II mesh, associated with the coupling
    /// interface. The variable is public and should be used during grid
    /// generation, but is also involved during system assembly. The only thing,
    /// one needs to make sure is, that this ID is not given to another part of
    /// the boundary e.g. clamped one.
    const unsigned int dealii_boundary_interface_id;


  private:
    // public precice solverinterface, needed in order to steer the time loop
    // inside the solver.
    std::shared_ptr<precice::SolverInterface> precice;
    /// The objects handling reading and writing data
    std::shared_ptr<CouplingInterface<dim, VectorizedArrayType>> writer;
    std::shared_ptr<CouplingInterface<dim, VectorizedArrayType>> reader;

    // Container to store time dependent data in case of an implicit coupling
    std::vector<VectorType> old_state_data;
    double                  old_time_value = 0;
  };



  template <int dim,
            int fe_degree,
            typename VectorType,
            typename VectorizedArrayType>
  template <typename ParameterClass>
  Adapter<dim, fe_degree, VectorType, VectorizedArrayType>::Adapter(
    const ParameterClass &parameters,
    const unsigned int    dealii_boundary_interface_id,
    const bool            shared_memory_parallel,
    std::shared_ptr<MatrixFree<dim, double, VectorizedArrayType>> data)
    : dealii_boundary_interface_id(dealii_boundary_interface_id)
    , read_mesh_name(parameters.read_mesh_name)
    , write_mesh_name(parameters.write_mesh_name)
    , read_data_name(parameters.read_data_name)
    , write_data_name(parameters.write_data_name)
    , read_write_on_same(read_mesh_name == write_mesh_name)
    , write_sampling(parameters.write_sampling)
    , shared_memory_parallel(shared_memory_parallel)
    , mf_data_reference(data)
  {
    precice = std::make_shared<precice::SolverInterface>(
      parameters.participant_name,
      parameters.config_file,
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),
      Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));

    AssertThrow(
      dim == precice->getDimensions(),
      ExcMessage("The dimension of your solver needs to be consistent with the "
                 "dimension specified in your precice-config file. In case you "
                 "run one of the tutorials, the dimension can be specified via "
                 "cmake -D DIM=dim ."));

    AssertThrow(dim > 1, ExcNotImplemented());
  }



  template <int dim,
            int fe_degree,
            typename VectorType,
            typename VectorizedArrayType>
  void
  Adapter<dim, fe_degree, VectorType, VectorizedArrayType>::initialize(
    const VectorType &dealii_to_precice,
    const int         dof_index,
    const int         read_quad_index_,
    const int         write_quad_index_)
  {
    // Check if the value has been set or if we choose a default one
    //    const int selected_sampling =
    //      write_sampling == std::numeric_limits<int>::max() ? fe_degree + 1 :
    //    write_sampling;
    read_quad_index  = read_quad_index_;
    write_quad_index = read_write_on_same ?
                         read_quad_index_ :
                         write_quad_index_ /*TODO: Implement sampling variant*/;

    // get precice specific IDs from precice and store them in
    // the class they are later needed for data transfer
    read_mesh_id  = precice->getMeshID(read_mesh_name);
    read_data_id  = precice->getDataID(read_data_name, read_mesh_id);
    write_mesh_id = precice->getMeshID(write_mesh_name);
    write_data_id = precice->getDataID(write_data_name, write_mesh_id);

    set_mesh_vertices(true, dof_index, read_quad_index);
    if (!read_write_on_same)
      set_mesh_vertices(false, dof_index, write_quad_index);
    else // TODO: Replace copy by some smart pointer
      write_nodes_ids = read_nodes_ids;
    print_info();

    // Initialize preCICE internally
    precice->initialize();

    // write initial writeData to preCICE if required
    if (precice->isActionRequired(precice::constants::actionWriteInitialData()))
      {
        write_all_quadrature_nodes(dealii_to_precice);

        precice->markActionFulfilled(
          precice::constants::actionWriteInitialData());
      }
    precice->initializeData();

    // Maybe, read block-wise and work with an AlignedVector since the read data
    // (forces) is multiple times required during the Newton iteration
    //    if (shared_memory_parallel && precice->isReadDataAvailable())
    //      precice->readBlockVectorData(read_data_id,
    //                                   read_nodes_ids.size(),
    //                                   read_nodes_ids.data(),
    //                                   read_data.data());
  }



  template <int dim,
            int fe_degree,
            typename VectorType,
            typename VectorizedArrayType>
  void
  Adapter<dim, fe_degree, VectorType, VectorizedArrayType>::advance(
    const VectorType &dealii_to_precice,
    const double      computed_timestep_length)
  {
    if (precice->isWriteDataRequired(computed_timestep_length))
      write_all_quadrature_nodes(dealii_to_precice);

    // Here, we need to specify the computed time step length and pass it to
    // preCICE
    precice->advance(computed_timestep_length);

    //    if (shared_memory_parallel && precice->isReadDataAvailable())
    //      precice->readBlockVectorData(read_data_id,
    //                                   read_nodes_ids.size(),
    //                                   read_nodes_ids.data(),
    //                                   read_data.data());
  }



  template <int dim,
            int fe_degree,
            typename VectorType,
            typename VectorizedArrayType>
  inline void
  Adapter<dim, fe_degree, VectorType, VectorizedArrayType>::
    save_current_state_if_required(const std::function<void()> &save_state)
  {
    // First, we let preCICE check, whether we need to store the variables.
    // Then, the data is stored in the class
    if (precice->isActionRequired(
          precice::constants::actionWriteIterationCheckpoint()))
      {
        save_state();
        precice->markActionFulfilled(
          precice::constants::actionWriteIterationCheckpoint());
      }
  }



  template <int dim,
            int fe_degree,
            typename VectorType,
            typename VectorizedArrayType>
  inline void
  Adapter<dim, fe_degree, VectorType, VectorizedArrayType>::
    reload_old_state_if_required(const std::function<void()> &reload_old_state)
  {
    // In case we need to reload a state, we just take the internally stored
    // data vectors and write then in to the input data
    if (precice->isActionRequired(
          precice::constants::actionReadIterationCheckpoint()))
      {
        reload_old_state();
        precice->markActionFulfilled(
          precice::constants::actionReadIterationCheckpoint());
      }
  }



  template <int dim,
            int fe_degree,
            typename VectorType,
            typename VectorizedArrayType>
  void
  Adapter<dim, fe_degree, VectorType, VectorizedArrayType>::
    write_all_quadrature_nodes(const VectorType &data)
  {
    // TODO: n_qpoints_1D is hard coded
    FEFaceEvaluation<dim,
                     fe_degree,
                     fe_degree + 1 /*n_qpoints_1D*/,
                     dim,
                     double,
                     VectorizedArrayType>
      // TODO: Parametrize dof_index
      phi(*mf_data_reference, true, 0, write_quad_index);

    // In order to unroll the vectorization
    std::array<double, dim * VectorizedArrayType::size()> unrolled_local_data;
    auto index = write_nodes_ids.begin();

    for (unsigned int face = mf_data_reference->n_inner_face_batches();
         face < mf_data_reference->n_boundary_face_batches() +
                  mf_data_reference->n_inner_face_batches();
         ++face)
      {
        const auto boundary_id = mf_data_reference->get_boundary_id(face);

        // Only for interface nodes
        if (boundary_id != dealii_boundary_interface_id)
          continue;

        // Read and interpolate
        phi.reinit(face);
        phi.gather_evaluate(data, true, false);
        const int active_faces =
          mf_data_reference->n_active_entries_per_face_batch(face);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto local_data = phi.get_value(q);
            Assert(index != write_nodes_ids.end(), ExcInternalError());

            // Transform Tensor<1,dim,VectorizedArrayType> into preCICE conform
            // format
            // Alternatively: Loop directly over active_faces instead of size()
            // and use writeVectorData
            for (int d = 0; d < dim; ++d)
              for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
                unrolled_local_data[d + dim * v] = local_data[d][v];

            precice->writeBlockVectorData(write_data_id,
                                          active_faces,
                                          index->data(),
                                          unrolled_local_data.data());
            ++index;
          }
      }
  }



  template <int dim,
            int fe_degree,
            typename VectorType,
            typename VectorizedArrayType>
  inline Tensor<1, dim, VectorizedArrayType>
  Adapter<dim, fe_degree, VectorType, VectorizedArrayType>::
    read_on_quadrature_point(
      const std::array<int, VectorizedArrayType::size()> &vertex_ids,
      const unsigned int                                  active_faces) const
  {
    Tensor<1, dim, VectorizedArrayType>                   dealii_data;
    std::array<double, dim * VectorizedArrayType::size()> precice_data;
    // Assertion is exclusive at the boundaries
    AssertIndexRange(active_faces, VectorizedArrayType::size() + 1);
    // TODO: Check if the if statement still makes sense
    //      if (precice->isReadDataAvailable())
    precice->readBlockVectorData(read_data_id,
                                 active_faces,
                                 vertex_ids.begin(),
                                 precice_data.data());
    // Transform back to Tensor format
    for (int d = 0; d < dim; ++d)
      for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
        dealii_data[d][v] = precice_data[d + dim * v];

    return dealii_data;
  }



  template <int dim,
            int fe_degree,
            typename VectorType,
            typename VectorizedArrayType>
  void
  Adapter<dim, fe_degree, VectorType, VectorizedArrayType>::set_mesh_vertices(
    const bool         is_read_mesh,
    const unsigned int dof_index,
    const unsigned int quad_index)
  {
    const unsigned int mesh_id = is_read_mesh ? read_mesh_id : write_mesh_id;
    auto &interface_nodes_ids = is_read_mesh ? read_nodes_ids : write_nodes_ids;

    // Initial guess: half of the boundary is part of the coupling interface
    interface_nodes_ids.reserve(mf_data_reference->n_boundary_face_batches() *
                                0.5);
    // TODO: n_qpoints_1D is hard coded
    static constexpr unsigned int n_qpoints_1d = fe_degree + 1;
    FEFaceEvaluation<dim,
                     fe_degree,
                     n_qpoints_1d,
                     dim,
                     double,
                     VectorizedArrayType>
      phi(*mf_data_reference, true, dof_index, quad_index);

    std::array<double, dim * VectorizedArrayType::size()> unrolled_vertices;
    std::array<int, VectorizedArrayType::size()>          node_ids;
    unsigned int                                          size              = 0;
    unsigned int                                          active_faces_size = 0;

    for (unsigned int face = mf_data_reference->n_inner_face_batches();
         face < mf_data_reference->n_boundary_face_batches() +
                  mf_data_reference->n_inner_face_batches();
         ++face)
      {
        const auto boundary_id = mf_data_reference->get_boundary_id(face);

        // Only for interface nodes
        if (boundary_id != dealii_boundary_interface_id)
          continue;

        phi.reinit(face);
        const int active_faces =
          mf_data_reference->n_active_entries_per_face_batch(face);

        // Only relevant for write meshes
        if (!is_read_mesh)
          for (int i = 0; i < active_faces; ++i)
            {
              const auto &f_pair =
                mf_data_reference->get_face_iterator(face, i);
              write_mesh_stats +=
                f_pair.first->face(f_pair.second)->diameter() / n_qpoints_1d;
              active_faces_size += active_faces;
            }


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

            precice->setMeshVertices(mesh_id,
                                     active_faces,
                                     unrolled_vertices.data(),
                                     node_ids.data());
            interface_nodes_ids.emplace_back(node_ids);
            ++size;
          }
      }
    // resize the IDs in case the initial guess was too large
    interface_nodes_ids.resize(size);
    if (active_faces_size != 0)
      write_mesh_stats /= active_faces_size;
  }



  template <int dim,
            int fe_degree,
            typename VectorType,
            typename VectorizedArrayType>
  inline auto
  Adapter<dim, fe_degree, VectorType, VectorizedArrayType>::
    begin_interface_IDs() const
  {
    return read_nodes_ids.begin();
  }



  template <int dim,
            int fe_degree,
            typename VectorType,
            typename VectorizedArrayType>
  inline bool
  Adapter<dim, fe_degree, VectorType, VectorizedArrayType>::
    is_coupling_ongoing() const
  {
    return precice->isCouplingOngoing();
  }



  template <int dim,
            int fe_degree,
            typename VectorType,
            typename VectorizedArrayType>
  inline bool
  Adapter<dim, fe_degree, VectorType, VectorizedArrayType>::
    is_time_window_complete() const
  {
    return precice->isTimeWindowComplete();
  }



  template <int dim,
            int fe_degree,
            typename VectorType,
            typename VectorizedArrayType>
  void
  Adapter<dim, fe_degree, VectorType, VectorizedArrayType>::print_info() const
  {
    const unsigned int r_size =
      Utilities::MPI::sum(static_cast<unsigned int>(read_nodes_ids.size()),
                          MPI_COMM_WORLD);

    const bool warn_unused_write_option =
      (write_sampling != std::numeric_limits<int>::max());
    const std::string write_message =
      ("--     . (a write sampling different from default is not yet supported)");
    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);
    pcout << "\n--     . Read and write on same location: "
          << (read_write_on_same ? "true" : "false") << "\n"
          << (warn_unused_write_option ?
                "--     . Ignoring specified write sampling." :
                write_message)
          << std::endl;
    pcout
      << "--     . Number of interface nodes (upper bound due to potential empty "
      << "lanes):\n--     . " << std::setw(5)
      << r_size * VectorizedArrayType::size()
      << " ( = " << (r_size / Utilities::pow(fe_degree + 1, dim - 1))
      << " [face-batches] x " << Utilities::pow(fe_degree + 1, dim - 1)
      << " [nodes/face] x " << VectorizedArrayType::size()
      << " [faces/face-batch]) \n"
      << "--     . Read node location: Gauss-Legendre\n"
      << "--     . Write node location:"
      << (read_write_on_same ? "Gauss-Legendre" : "equidistant") << "\n"
      << "--     . Average radial write node distance: " << write_mesh_stats
      << "\n"
      << std::endl;
  }
} // namespace Adapter
