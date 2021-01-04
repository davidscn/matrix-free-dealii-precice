#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_generic.h>

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
  template <int dim, typename VectorType, typename ParameterClass>
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
    Adapter(const ParameterClass &parameters,
            const unsigned int    dealii_boundary_interface_id,
            const bool            shared_memory_parallel);

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
    initialize(const DoFHandler<dim> &                    dof_handler,
               std::shared_ptr<const Mapping<dim>>        mapping,
               std::shared_ptr<const Quadrature<dim - 1>> read_quadrature,
               const VectorType &                         dealii_to_precice);

    /**
     * @brief      Advances preCICE after every timestep, converts data formats
     *             between preCICE and dealii
     *
     * @param[in]  deal_to_precice Same data as in @p initialize_precice() i.e.
     *             data, which should be given to preCICE after each time step
     *             and exchanged with other participants.
     * @param[out] precice_to_deal Same data as in @p initialize_precice() i.e.
     *             data, which is received from preCICE/other participants
     *             after each time step and exchanged with other participants.
     * @param[in]  computed_timestep_length Length of the timestep used by
     *             the solver.
     * TODO
     */
    void
    advance(const VectorType &     dealii_to_precice,
            const DoFHandler<dim> &dof_handler,
            const double           computed_timestep_length);

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
    void
    read_on_quadrature_point(std::array<double, dim> &data,
                             const unsigned int       vertex_id) const;

    /**
     * @brief begin_interface_IDs Returns an iterator of the interface node IDs
     *        of the 'read' mesh´. Allows to iterate over the read node IDs.
     */
    auto
    begin_interface_IDs() const;

    /**
     * @brief read_on_quadrature_point_fron_block_data Reads data from a block
     *        data set and returns values associated to a quadrature point.
     *
     *        As opposed to @p read_on_quadrature_point(), this function reads
     *        data from a class-own vector, i.e., it is assumed that the read
     *        data has been copied from preCICE using `readBlockVectorData()`.
     *        As a consequence, no preCICE functionality is used throughout this
     *        function and it is thread safe. If a thread safe implementation is
     *        not required, please use @p read_on_quadrature_point(). In order
     *        to obtain the @param block_data_id, the function @p get_block_data_id
     *        (see below) can be used.
     *
     * @param[out] data coupling data to be filled by the function
     * @param[in]  block_data_id index of the data point in the block data set
     */
    void
    read_on_quadrature_point_from_block_data(
      Tensor<1, dim> &   data,
      const unsigned int block_data_id) const;

    /**
     * @brief get_block_data_id Returns the data id of the first data point in
     *        the class vector as described in
     *        @p read_on_quadrature_point_from_block_data given the global face
     *        index. Hence, the function is supposed to be only called once per
     *        interface. All consequent read operations for the remaining data
     *        of the other quadrature points can be performed by simply adding
     *        the local index of the quadrature point because the data is
     *        ordered contiguous for each element. The whole lookup might look
     *       the following way
     * @code
     *  // Loop over all cells
     *  for (const auto &face : cell->face_iterators())
     *  // Check if we are at a coupling interface
     *  if (face->boundary_id() == interface_id)
     *    {
     *      // Initialize the FE on this cell
     *      fe_face_values_ref.reinit(cell, face);
     *      // Get the data first data ID for this cell
     *      const unsigned int precice_id = adapter.get_block_data_id(
     *        cell->face_index(cell->face_iterator_to_index(face)));
     *
     *      // Initialize vector for values at each quad point
     *      Tensor<1, dim> precice_data;
     *
     *      for (unsigned int f_q_point = 0; f_q_point < n_q_points_f;
     *           ++f_q_point)
     *        {
     *          // Get data with the ID
     *          adapter.read_on_quadrature_point_from_block_data(precice_data,
     *                                                           precice_id +
     *                                                             f_q_point);
     *          ... do stuff with precice_data
     *        }
     *     }
     * @endcode
     *
     *        The reason for calling the function only once per face is that it
     *        looks for the data ID in a map and we want to reduce the map look
     *        ups as much as possible.
     *
     * @param[in] face_id Global face index as obtained by cell->face_index
     *
     * @return The first id of the data in the class-own data vector, which holds
     *         a copy of the preCICE read data.
     */
    unsigned int
    get_block_data_id(const unsigned int face_id) const;


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
    std::vector<int> read_nodes_ids;
    std::vector<int> write_nodes_ids;
    // Only required for shared parallelism
    std::map<unsigned int, unsigned int> read_id_map;
    std::vector<double>                  read_data;

    // Container to store time dependent data in case of an implicit coupling
    std::vector<VectorType> old_state_data;
    double                  old_time_value;

    std::shared_ptr<const Quadrature<dim - 1>> write_quadrature;
    std::shared_ptr<const Quadrature<dim - 1>> read_quadrature;
    std::shared_ptr<const Mapping<dim>>        mapping;

    /**
     * @brief set_mesh_vertices Define a vertex coupling mesh for preCICE coupling
     *
     * @param[in] dof_handler DofHandler to be used
     * @param[in] is_read_mesh Defines whether the mesh is associated to a aread
     *            or a write mesh
     */
    void
    set_mesh_vertices(const DoFHandler<dim> &dof_handler,
                      const bool             is_read_mesh);

    /**
     * @brief write_all_quadrature_nodes Evaluates the given @param data at the
     *        quadrature_points of the given @param write_quadrature formula and
     *        passes it to preCICE
     *
     * @param[in] data The data to be passed to preCICE (absolute displacement
     *            for FSI)
     * @param[in] dof_handler DofHandler to be used
     */
    void
    write_all_quadrature_nodes(const VectorType &     data,
                               const DoFHandler<dim> &dof_handler);

    void
    print_info() const;
  };



  template <int dim, typename VectorType, typename ParameterClass>
  Adapter<dim, VectorType, ParameterClass>::Adapter(
    const ParameterClass &parameters,
    const unsigned int    dealii_boundary_interface_id,
    const bool            shared_memory_parallel)
    : dealii_boundary_interface_id(dealii_boundary_interface_id)
    , read_mesh_name(parameters.read_mesh_name)
    , write_mesh_name(parameters.write_mesh_name)
    , read_data_name(parameters.read_data_name)
    , write_data_name(parameters.write_data_name)
    , read_write_on_same(read_mesh_name == write_mesh_name)
    , write_sampling(parameters.write_sampling)
    , shared_memory_parallel(shared_memory_parallel)
  {
    // TODO: Replace by MF communicator
    precice = std::make_shared<precice::SolverInterface>(
      parameters.participant_name,
      parameters.config_file,
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),
      Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::initialize(
    const DoFHandler<dim> &                    dof_handler,
    std::shared_ptr<const Mapping<dim>>        mapping_,
    std::shared_ptr<const Quadrature<dim - 1>> read_quadrature_,
    const VectorType &                         dealii_to_precice)
  {
    AssertThrow(
      dim == precice->getDimensions(),
      ExcMessage("The dimension of your solver needs to be consistent with the "
                 "dimension specified in your precice-config file. In case you "
                 "run one of the tutorials, the dimension can be specified via "
                 "cmake -D DIM=dim ."));

    AssertThrow(dim > 1, ExcNotImplemented());

    // Check if the value has been set or if we choose a default one
    const int selected_sampling =
      write_sampling == std::numeric_limits<int>::max() ?
        read_quadrature_->size() :
        write_sampling;
    write_quadrature =
      read_write_on_same ?
        read_quadrature_ :
        std::make_shared<const QEquidistant<dim - 1>>(selected_sampling);
    read_quadrature = read_quadrature_;
    mapping         = mapping_;
    // get precice specific IDs from precice and store them in
    // the class they are later needed for data transfer
    read_mesh_id  = precice->getMeshID(read_mesh_name);
    read_data_id  = precice->getDataID(read_data_name, read_mesh_id);
    write_mesh_id = precice->getMeshID(write_mesh_name);
    write_data_id = precice->getDataID(write_data_name, write_mesh_id);

    set_mesh_vertices(dof_handler, true);
    if (!read_write_on_same)
      set_mesh_vertices(dof_handler, false);
    else // TODO: Replace copy by some smart pointer
      write_nodes_ids = read_nodes_ids;
    print_info();

    if (shared_memory_parallel)
      read_data.resize(read_nodes_ids.size() * dim);

    // Initialize preCICE internally
    precice->initialize();

    // write initial writeData to preCICE if required
    if (precice->isActionRequired(precice::constants::actionWriteInitialData()))
      {
        write_all_quadrature_nodes(dealii_to_precice, dof_handler);

        precice->markActionFulfilled(
          precice::constants::actionWriteInitialData());

        precice->initializeData();
      }

    if (shared_memory_parallel && precice->isReadDataAvailable())
      precice->readBlockVectorData(read_data_id,
                                   read_nodes_ids.size(),
                                   read_nodes_ids.data(),
                                   read_data.data());
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::advance(
    const VectorType &     dealii_to_precice,
    const DoFHandler<dim> &dof_handler,
    const double           computed_timestep_length)
  {
    if (precice->isWriteDataRequired(computed_timestep_length))
      write_all_quadrature_nodes(dealii_to_precice, dof_handler);

    // Here, we need to specify the computed time step length and pass it to
    // preCICE
    precice->advance(computed_timestep_length);

    if (shared_memory_parallel && precice->isReadDataAvailable())
      precice->readBlockVectorData(read_data_id,
                                   read_nodes_ids.size(),
                                   read_nodes_ids.data(),
                                   read_data.data());

    // Alternative, if you don't want to store the indices
    //    const int rsize = (1 / dim) * read_data.size();
    //    if (precice->isReadDataAvailable())
    //      for (const auto i : std_cxx20::ranges::iota_view<int, int>{0,
    //      rsize})
    //        precice->readVectorData(read_data_id, i, &read_data[i * dim]);
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::save_current_state_if_required(
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



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::reload_old_state_if_required(
    std::vector<VectorType *> &state_variables)
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



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::write_all_quadrature_nodes(
    const VectorType &     data,
    const DoFHandler<dim> &dof_handler)
  {
    FEFaceValues<dim>           fe_face_values(*mapping,
                                     dof_handler.get_fe(),
                                     *write_quadrature,
                                     update_values);
    std::vector<Vector<double>> quad_values(write_quadrature->size(),
                                            Vector<double>(dim));
    std::array<double, dim>     local_data;
    auto                        index = write_nodes_ids.begin();

    for (const auto &cell : dof_handler.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() == true &&
            face->boundary_id() == dealii_boundary_interface_id)
          {
            fe_face_values.reinit(cell, face);
            fe_face_values.get_function_values(data, quad_values);

            // Alternative: write data of a cell as a whole block using
            // writeBlockVectorData
            for (const auto f_q_point :
                 fe_face_values.quadrature_point_indices())
              {
                Assert(index != write_nodes_ids.end(), ExcInternalError());
                // TODO: Check if the additional array is necessary. Maybe we
                // can directly use quad_values[f_q_point].data() for preCICE
                for (uint d = 0; d < dim; ++d)
                  local_data[d] = quad_values[f_q_point][d];

                precice->writeVectorData(write_data_id,
                                         *index,
                                         local_data.data());

                ++index;
              }
          }
  }



  template <int dim, typename VectorType, typename ParameterClass>
  inline void
  Adapter<dim, VectorType, ParameterClass>::read_on_quadrature_point(
    std::array<double, dim> &data,
    const unsigned int       vertex_id) const
  {
    // TODO: Check if the if statement still makes sense
    //      if (precice->isReadDataAvailable())
    precice->readVectorData(read_data_id, vertex_id, data.data());
  }



  template <int dim, typename VectorType, typename ParameterClass>
  inline void
  Adapter<dim, VectorType, ParameterClass>::
    read_on_quadrature_point_from_block_data(
      Tensor<1, dim> &   data,
      const unsigned int block_data_id) const
  {
    // Assert the accessed index
    AssertIndexRange(block_data_id * dim + (dim - 1), read_data.size());
    for (uint d = 0; d < dim; ++d)
      data[d] = read_data[block_data_id * dim + d];
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::set_mesh_vertices(
    const DoFHandler<dim> &dof_handler,
    const bool             is_read_mesh)
  {
    const unsigned int mesh_id = is_read_mesh ? read_mesh_id : write_mesh_id;
    auto &interface_nodes_ids = is_read_mesh ? read_nodes_ids : write_nodes_ids;
    const auto quadrature = is_read_mesh ? read_quadrature : write_quadrature;

    // TODO: Find a suitable guess for the number of interface points (optional)
    interface_nodes_ids.reserve(20);
    std::array<double, dim> vertex;
    FEFaceValues<dim>       fe_face_values(*mapping,
                                     dof_handler.get_fe(),
                                     *quadrature,
                                     update_quadrature_points);

    // Loop over all elements and evaluate data at quadrature points
    for (const auto &cell : dof_handler.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() == true &&
            face->boundary_id() == dealii_boundary_interface_id)
          {
            fe_face_values.reinit(cell, face);

            // Create a map for shared parallelism
            if (shared_memory_parallel && is_read_mesh)
              read_id_map[cell->face_index(cell->face_iterator_to_index(
                face))] = interface_nodes_ids.size();

            for (const auto f_q_point :
                 fe_face_values.quadrature_point_indices())
              {
                const auto &q_point =
                  fe_face_values.quadrature_point(f_q_point);
                for (uint d = 0; d < dim; ++d)
                  vertex[d] = q_point[d];

                interface_nodes_ids.emplace_back(
                  precice->setMeshVertex(mesh_id, vertex.data()));
              }
          }
  }



  template <int dim, typename VectorType, typename ParameterClass>
  auto
  Adapter<dim, VectorType, ParameterClass>::begin_interface_IDs() const
  {
    return read_nodes_ids.begin();
  }



  template <int dim, typename VectorType, typename ParameterClass>
  unsigned int
  Adapter<dim, VectorType, ParameterClass>::get_block_data_id(
    const unsigned int face_id) const
  {
    return read_id_map.at(face_id);
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::print_info() const
  {
    const unsigned int r_size = read_nodes_ids.size();
    const unsigned int w_size = write_nodes_ids.size();
    const bool         warn_unused_write_option =
      read_write_on_same && (write_sampling != std::numeric_limits<int>::max());
    const std::string write_message =
      ("Write sampling: " + std::to_string(write_quadrature->size()) +
       (write_quadrature->size() == read_quadrature->size() ? " ( = Default )" :
                                                              ""));
    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
    pcout << "\t Read and write on same location: "
          << (read_write_on_same ? "true" : "false") << "\n"
          << (warn_unused_write_option ?
                "\t Ignoring specified write sampling." :
                "\t " + write_message)
          << std::endl;
    pcout << "\t Number of read nodes: " << std::setw(5) << r_size
          << " ( = " << r_size / read_quadrature->size() << " [faces] x "
          << read_quadrature->size() << " [nodes/face] ) \n"
          << "\t Read node location: Gauss-Legendre" << std::endl;
    pcout << "\t Number of write nodes:" << std::setw(5) << w_size
          << " ( = " << w_size / write_quadrature->size() << " [faces] x "
          << write_quadrature->size() << " [nodes/face] ) \n"
          << "\t Write node location: "
          << (read_write_on_same ? "Gauss-Legendre" : "equidistant") << "\n"
          << std::endl;
  }
} // namespace Adapter
