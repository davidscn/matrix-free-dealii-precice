#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <base/fe_integrator.h>
#include <precice/SolverInterface.hpp>

namespace Adapter
{
  using namespace dealii;
  /**
   * An abstract base class, which defines the interface for the functions used
   * in the main Adapter class. Each instance of all derived classes are always
   * dedicated to a specific coupling mesh and may provide functions on how to
   * read and write data on this mesh and how to define the mesh.
   */
  template <int dim, int data_dim, typename VectorizedArrayType>
  class CouplingInterface
  {
  public:
    CouplingInterface(
      std::shared_ptr<const MatrixFree<dim, double, VectorizedArrayType>> data,
      const std::shared_ptr<precice::SolverInterface> &precice,
      const std::string &                              mesh_name,
      const types::boundary_id                         interface_id);

    virtual ~CouplingInterface() = default;

    using value_type =
      typename FEFaceIntegrators<dim, data_dim, double, VectorizedArrayType>::
        value_type;
    /**
     * @brief define_coupling_mesh Define the coupling mesh associated to the
     *        data points
     *
     * @param dealii_boundary_interface_id boundary ID of the deal.II mesh
     */
    virtual void
    define_coupling_mesh() = 0;

    /**
     * @brief write_data Write the data associated to the defined vertice
     *        to preCICE
     *
     * @param data_vector Vector holding the global solution to be passed to
     *        preCICE.
     */
    virtual void
    write_data(
      const LinearAlgebra::distributed::Vector<double> &data_vector) = 0;

    /**
     * @brief read_on_quadrature_point Returns the relevant read data on
     *        quadrature point. Currently only implemented in the
     *        dealiiInterface
     *
     * @param[in]  id_number Number of the quadrature point with respect to
     *             the total number of interface quadrature points this rank
     *             works on.
     * @param[in]  active_faces Number of active faces the matrix-free object
     *             works on
     *
     * @return dim dimensional data associated to the interface node
     */
    virtual value_type
    read_on_quadrature_point(const unsigned int id_number,
                             const unsigned int active_faces) const = 0;
    /**
     * @brief add_read_data
     * @param read_data_name
     */
    void
    add_read_data(const std::string &read_data_name);

    /**
     * @brief add_read_data
     * @param read_data_name
     */
    void
    add_write_data(const std::string &write_data_name);

    /**
     * @brief print_info
     * @param stream
     * @param reader Boolean in order to decide if we want read or write
     *        data information
     */
    void
    print_info(const bool reader) const;

  protected:
    /// The MatrixFree object (preCICE can only handle double precision)
    std::shared_ptr<const MatrixFree<dim, double, VectorizedArrayType>> mf_data;

    /// public precice solverinterface
    std::shared_ptr<precice::SolverInterface> precice;

    /// Configuration parameters
    const std::string mesh_name;
    std::string       read_data_name  = "";
    std::string       write_data_name = "";
    int               mesh_id         = -1;
    int               read_data_id    = -1;
    int               write_data_id   = -1;

    const types::boundary_id dealii_boundary_interface_id;

    virtual std::string
    get_interface_type() const = 0;
  };



  template <int dim, int data_dim, typename VectorizedArrayType>
  CouplingInterface<dim, data_dim, VectorizedArrayType>::CouplingInterface(
    std::shared_ptr<const MatrixFree<dim, double, VectorizedArrayType>> data,
    const std::shared_ptr<precice::SolverInterface> &                   precice,
    const std::string &      mesh_name,
    const types::boundary_id interface_id)
    : mf_data(data)
    , precice(precice)
    , mesh_name(mesh_name)
    , dealii_boundary_interface_id(interface_id)
  {
    Assert(data.get() != nullptr, ExcNotInitialized());
    Assert(precice.get() != nullptr, ExcNotInitialized());

    // Ask preCICE already in the constructor for the IDs
    mesh_id = precice->getMeshID(mesh_name);
  }


  template <int dim, int data_dim, typename VectorizedArrayType>
  void
  CouplingInterface<dim, data_dim, VectorizedArrayType>::add_read_data(
    const std::string &read_data_name_)
  {
    Assert(mesh_id != -1, ExcNotInitialized());
    read_data_name = read_data_name_;
    read_data_id   = precice->getDataID(read_data_name, mesh_id);
  }



  template <int dim, int data_dim, typename VectorizedArrayType>
  void
  CouplingInterface<dim, data_dim, VectorizedArrayType>::add_write_data(
    const std::string &write_data_name_)
  {
    Assert(mesh_id != -1, ExcNotInitialized());
    write_data_name = write_data_name_;
    write_data_id   = precice->getDataID(write_data_name, mesh_id);
  }



  template <int dim, int data_dim, typename VectorizedArrayType>
  void
  CouplingInterface<dim, data_dim, VectorizedArrayType>::print_info(
    const bool reader) const
  {
    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);

    pcout << "--     Data " << (reader ? "reading" : "writing") << ":\n"
          << "--     . data name: "
          << (reader ? read_data_name : write_data_name) << "\n"
          << "--     . associated mesh: " << mesh_name << "\n"
          << "--     . Number of interface nodes: "
          << Utilities::MPI::sum(precice->getMeshVertexSize(mesh_id),
                                 MPI_COMM_WORLD)
          << "\n"
          << "--     . Node location: " << get_interface_type() << "\n"
          << std::endl;
  }
} // namespace Adapter
