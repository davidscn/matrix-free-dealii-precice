#pragma once

#include <deal.II/base/parameter_handler.h>

using namespace dealii;

namespace FSI
{
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
      std::string scenario         = "FSI3";
      std::string config_file      = "precice config-file";
      std::string participant_name = "dealiisolver";
      std::string mesh_name        = "default";
      std::string read_mesh_name   = "default";
      std::string write_mesh_name  = "default";
      int         write_sampling   = std::numeric_limits<int>::max();
      std::string read_data_name   = "received-data";
      std::string write_data_name  = "calculated-data";

      void
      add_parameters(ParameterHandler &prm);
    };


    void
    PreciceAdapterConfiguration::add_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("precice configuration");
      {
        prm.add_parameter("Scenario",
                          scenario,
                          "Cases: FSI3 or PF for perpendicular flap",
                          Patterns::Selection("FSI3|PF"));
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
          "Mesh name",
          mesh_name,
          "Name of the coupling mesh in the precice-config.xml file",
          Patterns::Anything());
        prm.add_parameter(
          "Read mesh name",
          read_mesh_name,
          "Name of the read coupling mesh in the precice-config.xml file",
          Patterns::Anything());
        prm.add_parameter(
          "Write mesh name",
          write_mesh_name,
          "Name of the write coupling mesh in the precice-config.xml file",
          Patterns::Anything());
        prm.add_parameter("Write sampling",
                          write_sampling,
                          "Nodes per coupling face of the write mesh",
                          Patterns::Integer(0));
        prm.add_parameter(
          "Read data name",
          read_data_name,
          "Name of the read data in the precice-config.xml file",
          Patterns::Anything());
        prm.add_parameter(
          "Write data name",
          write_data_name,
          "Name of the write data in the precice-config.xml file",
          Patterns::Anything());
      }

      const std::string error_message(
        "Either specify a 'Mesh name', which will be applied to the read and write mesh (data location)"
        " or a separate 'Read Mesh name' and a 'Write mesh name' in order to enable more mapping frindly "
        "specialized data locations at the interface. Specifying both or none of these "
        "options is invalid. Make sure you adjust your configuration file '" +
        config_file + "' according to your settings.");

      if (read_mesh_name == write_mesh_name)
        {
          AssertThrow((mesh_name != read_mesh_name), ExcMessage(error_message));

          read_mesh_name  = mesh_name;
          write_mesh_name = mesh_name;
        }
      else
        {
          AssertThrow(("default" != read_mesh_name) &&
                        ("default" != write_mesh_name),
                      ExcMessage(error_message));

          AssertThrow("default" == mesh_name, ExcMessage(error_message));
        }

      prm.leave_subsection();
    }
  } // namespace Parameters
} // namespace FSI
