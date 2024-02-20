#pragma once

#include <deal.II/base/parameter_handler.h>

using namespace dealii;

/**
 * This class declares all preCICE parameters, which can be specified in the
 * parameter file. The subsection abut preCICE configurations is directly
 * interlinked to the Adapter class.
 */
namespace Parameters
{ /**
   * @brief PreciceAdapterConfiguration: Specifies preCICE related information.
   *        A lot of these information need to be consistent with the
   *        precice-config.xml file.
   */
  struct PreciceAdapterConfiguration
  {
    std::string config_file       = "precice config-file";
    std::string participant_name  = "dealiisolver";
    std::string mesh_name1        = "default1";
    std::string read_mesh_name1   = "default1";
    std::string write_mesh_name1  = "default1";
    std::string read_data_name1   = "received-data1";
    std::string write_data_name1d = "calculated-data1d";
    std::string write_data_name1v = "calculated-data1v";
    std::string mesh_name2        = "default2";
    std::string read_mesh_name2   = "default2";
    std::string write_mesh_name2  = "default2";
    std::string read_data_name2   = "received-data2";
    std::string write_data_name2d = "calculated-data2d";
    std::string write_data_name2v = "calculated-data2v";

    std::string write_data_specification = "values_on_quads";
    int         write_quad_index         = 0;
    void
    add_parameters(ParameterHandler &prm);
  };


  void
  PreciceAdapterConfiguration::add_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("precice configuration");
    {
      prm.add_parameter("precice config-file",
                        config_file,
                        "Name of the precice configuration file",
                        Patterns::Anything());
      prm.add_parameter(
        "Participant name",
        participant_name,
        "Name of the participant in the precice-config.xml file",
        Patterns::Anything());
      prm.add_parameter(
        "Mesh name1",
        mesh_name1,
        "Name of the coupling mesh in the precice-config.xml file",
        Patterns::Anything());
      prm.add_parameter(
        "Read mesh name1",
        read_mesh_name1,
        "Name of the read coupling mesh in the precice-config.xml file",
        Patterns::Anything());
      prm.add_parameter(
        "Write mesh name1",
        write_mesh_name1,
        "Name of the write coupling mesh in the precice-config.xml file",
        Patterns::Anything());
      prm.add_parameter("Read data name1",
                        read_data_name1,
                        "Name of the read data in the precice-config.xml file",
                        Patterns::Anything());
      prm.add_parameter("Write data name displacement1",
                        write_data_name1d,
                        "Name of the write data in the precice-config.xml file",
                        Patterns::Anything());
      prm.add_parameter("Write data name velocity1",
                        write_data_name1v,
                        "Name of the write data in the precice-config.xml file",
                        Patterns::Anything());


      prm.add_parameter(
        "Mesh name2",
        mesh_name2,
        "Name of the coupling mesh in the precice-config.xml file",
        Patterns::Anything());
      prm.add_parameter(
        "Read mesh name2",
        read_mesh_name2,
        "Name of the read coupling mesh in the precice-config.xml file",
        Patterns::Anything());
      prm.add_parameter(
        "Write mesh name2",
        write_mesh_name2,
        "Name of the write coupling mesh in the precice-config.xml file",
        Patterns::Anything());
      prm.add_parameter("Read data name2",
                        read_data_name2,
                        "Name of the read data in the precice-config.xml file",
                        Patterns::Anything());
      prm.add_parameter("Write data name displacement2",
                        write_data_name2d,
                        "Name of the write data in the precice-config.xml file",
                        Patterns::Anything());
      prm.add_parameter("Write data name velocity2",
                        write_data_name2v,
                        "Name of the write data in the precice-config.xml file",
                        Patterns::Anything());

      prm.add_parameter(
        "Write quadrature index",
        write_quad_index,
        "Index of the quadrature formula in MatrixFree used for initialization",
        Patterns::Integer(0));
      prm.add_parameter(
        "Write data specification",
        write_data_specification,
        "Specification of the write data location and the data type"
        "Available options are: values_on_dofs, values_on_quads, normal_gradients_on_quads",
        Patterns::Selection(
          "values_on_dofs|values_on_quads|normal_gradients_on_quads|"
          "values_on_other_mesh|gradients_on_other_mesh"));
    }
    prm.leave_subsection();
  }
} // namespace Parameters
