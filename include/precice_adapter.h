#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <precice/SolverInterface.hpp>
#include <q_equidistant.h>

#include <ostream>

namespace Adapter
{
  using namespace dealii;

  /**
   * The Adapter class keeps all functionalities to couple deal.II to other
   * solvers with preCICE i.e. data structures are set up, necessary information
   * is passed to preCICE etc.
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
     * @param[in]  deal_boundary_interface_id Boundary ID of the triangulation,
     *             which is associated with the coupling interface.
     * @param[in]
     */
    template <typename ParameterClass>
    Adapter(const ParameterClass &parameters,
            const unsigned int    dealii_boundary_interface_id,
            const bool            shared_memory_parallel,
            std::shared_ptr<MatrixFree<dim, double, VectorizedArrayType>> data);

    /**
     * @brief      Initializes preCICE and passes all relevant data to preCICE
     *
     * @param[in]  dof_handler Initialized dof_handler
     * @param[in]  deal_to_precice Data, which should be given to preCICE and
     *             exchanged with other participants. Wether this data is
     *             required already in the beginning depends on your
     *             individual configuration and preCICE determines it
     *             automatically. In many cases, this data will just represent
     *             your initial condition.
     * TODO
     */
    void
    initialize(const VectorType &dealii_to_precice,
               const int         dof_index        = 0,
               const int         read_quad_index_ = 0);

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
    save_current_state_if_required(
      const std::vector<VectorType *> &state_variables);

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
    reload_old_state_if_required(std::vector<VectorType *> &state_variables);

    /**
     * @brief read_on_quadrature_point Returns the dim dimensional read data
     *        given the ID of the interface node we want to access. The ID needs
     *        to be associated to the 'read' mesh. The function is in practice
     *        used together with @p begin_interface_IDs() (see below), so that
     *        we can iterate over all IDs consecutively. The function is not
     *        thread safe since the used preCICE function is not thread safe.
     *        Also, the functionality assumes that we iterate during the
     *        initialization in the same way over the interface than during the
     *        assembly step.
     *
     * @param[out] data dim dimensional data associated to the interface node
     * @param[in]  vertex_id preCICE related index of the read_data vertex
     */
    Tensor<1, dim, VectorizedArrayType>
    read_on_quadrature_point(
      const std::array<int, VectorizedArrayType::size()> &vertex_ids,
      const unsigned int                                  active_faces) const;

    /**
     * @brief begin_interface_IDs Returns an iterator of the interface node IDs
     *        of the 'read' mesh´. Allows to iterate over the read node IDs.
     */
    auto
    begin_interface_IDs() const;

    bool
    is_coupling_ongoing() const;

    bool
    is_time_window_complete() const;


    // public precice solverinterface, needed in order to steer the time loop
    // inside the solver.
    std::shared_ptr<precice::SolverInterface> precice;

    // Boundary ID of the deal.II mesh, associated with the coupling
    // interface. The variable is public and should be used during grid
    // generation, but is also involved during system assembly. The only thing,
    // one needs to make sure is, that this ID is not given to another part of
    // the boundary e.g. clamped one.
    const unsigned int dealii_boundary_interface_id;


  private:
    // preCICE related initializations
    // These variables are specified and read from the parameter file
    const std::string read_mesh_name;
    const std::string write_mesh_name;
    const std::string read_data_name;
    const std::string write_data_name;
    const bool        read_write_on_same;
    const int         write_sampling;


    // These IDs are given by preCICE during initialization
    int read_mesh_id;
    int read_data_id;
    int write_mesh_id;
    int write_data_id;

    const bool shared_memory_parallel;
    // Data containers which are passed to preCICE in an appropriate preCICE
    // specific format
    // The format cannot be used with vectorization, but this is not required.
    // The IDs are only required for preCICE
    std::vector<std::array<int, VectorizedArrayType::size()>> read_nodes_ids;
    std::vector<std::array<int, VectorizedArrayType::size()>> write_nodes_ids;
    // Only required for shared parallelism
    std::map<unsigned int, unsigned int> read_id_map;
    std::vector<double>                  read_data;

    // Container to store time dependent data in case of an implicit coupling
    std::vector<VectorType> old_state_data;
    double                  old_time_value;

    int write_quad_index;
    int read_quad_index;

    // preCICE can only handle double precision
    std::shared_ptr<MatrixFree<dim, double, VectorizedArrayType>>
      mf_data_reference;

    /**
     * @brief set_mesh_vertices Define a vertex coupling mesh for preCICE coupling
     *
     * @param[in] is_read_mesh Defines whether the mesh is associated to a aread
     *            or a write mesh
     * @param[in] dof_index index of the dof_handler in MatrixFree
     * @param[in] quad_index index of the dof_handler in MatrixFree
     */
    void
    set_mesh_vertices(const bool         is_read_mesh,
                      const unsigned int dof_index  = 0,
                      const unsigned int quad_index = 0);

    /**
     * @brief write_all_quadrature_nodes Evaluates the given @param data at the
     *        quadrature_points of the given @param write_quadrature formula and
     *        passes it to preCICE
     *
     * @param[in] data The data to be passed to preCICE (absolute displacement
     *            for FSI)
     */
    void
    write_all_quadrature_nodes(const VectorType &data);

    void
    print_info() const;
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
  }



  template <int dim,
            int fe_degree,
            typename VectorType,
            typename VectorizedArrayType>
  void
  Adapter<dim, fe_degree, VectorType, VectorizedArrayType>::initialize(
    const VectorType &dealii_to_precice,
    const int         dof_index,
    const int         read_quad_index_)
  {
    AssertThrow(
      dim == precice->getDimensions(),
      ExcMessage("The dimension of your solver needs to be consistent with the "
                 "dimension specified in your precice-config file. In case you "
                 "run one of the tutorials, the dimension can be specified via "
                 "cmake -D DIM=dim ."));

    AssertThrow(dim > 1, ExcNotImplemented());

    // Check if the value has been set or if we choose a default one
    //    const int selected_sampling =
    //      write_sampling == std::numeric_limits<int>::max() ? fe_degree + 1 :
    //    write_sampling;

    read_quad_index  = read_quad_index_;
    write_quad_index = read_write_on_same ?
                         read_quad_index :
                         read_quad_index /*TODO: Implement sampling variant*/;

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

    //    if (shared_memory_parallel)
    //      read_data.resize(read_nodes_ids.size() * dim);

    // Initialize preCICE internally
    precice->initialize();

    // write initial writeData to preCICE if required
    if (precice->isActionRequired(precice::constants::actionWriteInitialData()))
      {
        write_all_quadrature_nodes(dealii_to_precice);

        precice->markActionFulfilled(
          precice::constants::actionWriteInitialData());

        precice->initializeData();
      }

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

    // Alternative, if you don't want to store the indices
    //    const int rsize = (1 / dim) * read_data.size();
    //    if (precice->isReadDataAvailable())
    //      for (const auto i : std_cxx20::ranges::iota_view<int, int>{0,
    //      rsize})
    //        precice->readVectorData(read_data_id, i, &read_data[i * dim]);
  }



  template <int dim,
            int fe_degree,
            typename VectorType,
            typename VectorizedArrayType>
  void
  Adapter<dim, fe_degree, VectorType, VectorizedArrayType>::
    save_current_state_if_required(
      const std::vector<VectorType *> &state_variables)
  {
    // First, we let preCICE check, whether we need to store the variables.
    // Then, the data is stored in the class
    if (precice->isActionRequired(
          precice::constants::actionWriteIterationCheckpoint()))
      {
        old_state_data.resize(state_variables.size());

        for (uint i = 0; i < state_variables.size(); ++i)
          old_state_data[i] = *(state_variables[i]);

        precice->markActionFulfilled(
          precice::constants::actionWriteIterationCheckpoint());
      }
  }



  template <int dim,
            int fe_degree,
            typename VectorType,
            typename VectorizedArrayType>
  void
  Adapter<dim, fe_degree, VectorType, VectorizedArrayType>::
    reload_old_state_if_required(std::vector<VectorType *> &state_variables)
  {
    // In case we need to reload a state, we just take the internally stored
    // data vectors and write then in to the input data
    if (precice->isActionRequired(
          precice::constants::actionReadIterationCheckpoint()))
      {
        Assert(state_variables.size() == old_state_data.size(),
               ExcMessage(
                 "state_variables are not the same as previously saved."));
        // TODO:: Copy locally owned data from
        for (uint i = 0; i < state_variables.size(); ++i)
          *(state_variables[i]) = old_state_data[i];

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
            // Alternatively: Loop directly iver active_faces instead of size()
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
    Tensor<1, dim, VectorizedArrayType>             dealii_data;
    std::array<double, VectorizedArrayType::size()> precice_data;
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

    // TODO: Find a suitable guess for the number of interface points (optional)
    interface_nodes_ids.reserve(20);
    // TODO: n_qpoints_1D is hard coded
    FEFaceEvaluation<dim,
                     fe_degree,
                     fe_degree + 1 /*n_qpoints_1D*/,
                     dim,
                     double,
                     VectorizedArrayType>
      phi(*mf_data_reference, true, dof_index, quad_index);

    std::array<double, dim * VectorizedArrayType::size()> unrolled_vertices;
    std::array<int, VectorizedArrayType::size()>          node_ids;

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

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto local_vertex = phi.quadrature_point(q);

            // Transform Point<Vectorized> into preCICE conform format
            for (int d = 0; d < dim; ++d)
              for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
                unrolled_vertices[d + dim * v] = local_vertex[d][v];

            precice->setMeshVertices(mesh_id,
                                     active_faces,
                                     unrolled_vertices.data(),
                                     node_ids.data());
            interface_nodes_ids.emplace_back(node_ids);
          }
      }
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
    const unsigned int r_size = read_nodes_ids.size();
    const bool         warn_unused_write_option =
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
      << " ( = " << (r_size / (fe_degree + 1)) << " [face-batches] x "
      << fe_degree + 1 << " [nodes/face] x " << VectorizedArrayType::size()
      << " [faces/face-batch]) \n"
      << "--     . Node location: Gauss-Legendre\n"
      << std::endl;
  }
} // namespace Adapter
