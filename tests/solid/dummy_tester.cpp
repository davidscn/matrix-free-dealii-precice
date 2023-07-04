#include <math.h>
#include <precice/precice.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using Vector = std::vector<double>;

struct DataContainer
{
  void
  store_data(const Vector &vertices,
             const double &old_moment,
             const double &theta,
             const double &theta_dot)
  {
    old_vertices   = vertices;
    old_old_moment = old_moment;
    old_theta      = theta;
    old_theta_dot  = theta_dot;
  }

  void
  reload_data(Vector &vertices,
              double &old_moment,
              double &theta,
              double &theta_dot) const
  {
    vertices   = old_vertices;
    old_moment = old_old_moment;
    theta      = old_theta;
    theta_dot  = old_theta_dot;
  }

  Vector old_vertices;
  double old_old_moment;
  double old_theta;
  double old_theta_dot;
};

class Solver
{
public:
  Solver(const double moment_of_inertia)
    : moment_of_inertia(moment_of_inertia)
  {}

  void
  solve(const Vector &forces,
        Vector &      vertices,
        double &      old_moment,
        double &      theta,
        double &      theta_dot,
        const double  delta_t)
  {
    // Leapfrog update
    double moment = 0;
    for (uint i = 0; i < forces.size() / 2; ++i)
      moment += vertices[2 * i] * forces[2 * i + 1] +
                vertices[2 * i + 1] * forces[2 * i];

    // Compute angular acceleration
    double theta_acc_old = old_moment / moment_of_inertia;
    // Update angle
    theta =
      theta + theta_dot * delta_t +
      0.5 * (theta_acc_old + moment / moment_of_inertia) * std::pow(delta_t, 2);
    theta = -theta;
    // Update vertices according to rigid body rotation
    for (uint i = 0; i < vertices.size() / 2; ++i)
      {
        // compute length
        //      const double l       = std::sqrt(std::pow(vertices[2 * i], 2) +
        //      std::pow(vertices[2 * i + 1], 2));
        const double x_coord = vertices[2 * i];
        vertices[2 * i] =
          x_coord * std::cos(theta) + vertices[2 * i + 1] * std::sin(theta);
        vertices[2 * i + 1] =
          -x_coord * std::sin(theta) + vertices[2 * i + 1] * std::cos(theta);
      }
    // Compute recent moment on updated coordinates
    moment = 0;
    for (uint i = 0; i < forces.size() / 2; ++i)
      moment += vertices[2 * i] * forces[2 * i + 1] +
                vertices[2 * i + 1] * forces[2 * i];

    // Update recent angular velocity
    theta_dot = theta_dot + 0.5 * (moment + old_moment) * delta_t;
    //    theta_dot = -theta_dot;
    std::cout << "Theta: " << theta << " Theta dot: " << theta_dot
              << " Moment: " << moment << std::endl;
    // Update moment
    old_moment = moment;
  }

private:
  const double moment_of_inertia;
};

int
main()
{
  std::cout << "Dummy tester starting... \n";

  // Configuration settings
  const std::string config_file_name("precice-config.xml");
  const std::string solver_name("tester");
  const std::string mesh_name("tester-mesh");
  const std::string data_write_name("Stress");
  const std::string data_read_name("Displacement");

  // Mesh configuration
  constexpr int    vertical_refinement   = 3;
  constexpr int    horizontal_refinement = 6;
  constexpr double length                = 0.6 - 0.24899;
  constexpr double height                = 0.02;
  const Vector     rot_center({0.24899, 0.2});
  // Rotation centre is at (0,0)

  constexpr double density = 10;
  //***********************************************************************************//

  // Derived quantities
  // Substract common nodes at each corner
  constexpr int    n_vertical_nodes   = vertical_refinement * 2 + 1;
  constexpr int    n_horizontal_nodes = horizontal_refinement * 2 + 1;
  constexpr int    n_nodes = (n_vertical_nodes + n_horizontal_nodes - 2) * 2;
  constexpr double mass    = length * height * density;
  constexpr double inertia_moment =
    (1. / 12) * mass * (4 * std::pow(length, 2) + std::pow(height, 2));

  // Create Participant
  precice::Participant precice(solver_name,
                               config_file_name,
                               /*comm_rank*/ 0,
                               /*comm_size*/ 1);

  const int dim = precice.getMeshDimensions(mesh_name);

  // Set up data structures
  Vector           displacement(dim * n_nodes);
  Vector           vertices(dim * n_nodes);
  Vector           dummy_stress(dim * n_nodes);
  std::vector<int> vertex_ids(n_nodes);
  double           old_moment = 0.0;
  double           theta_dot  = 0.0;
  double           theta      = 0.0;

  {
    // Define a boundary mesh
    std::cout << "Dummy tester defining mesh: ";
    std::cout << n_nodes << " nodes " << std::endl;
    const double delta_y = height / (n_vertical_nodes - 1);
    const double delta_x = length / (n_horizontal_nodes - 1);

    // x planes
    for (int i = 0; i < n_vertical_nodes; ++i)
      {
        // negative x
        vertices[dim * i] = 0.0 + rot_center[0]; // fixed x
        vertices[dim * i + 1] =
          rot_center[1] - height * 0.5 + delta_y * i; // increasing y
        // positive x
        vertices[(2 * n_vertical_nodes) + dim * i] =
          length + rot_center[0]; // fixed x
        vertices[(2 * n_vertical_nodes) + dim * i + 1] =
          vertices[dim * i + 1]; // increasing y
      }

    // y planes
    const unsigned int of = 2 * dim * n_vertical_nodes; // static offset
    // Lower and upper bounds are already included
    const unsigned int n_remaining_nodes = n_horizontal_nodes - 2;
    for (int i = 0; i < n_remaining_nodes; ++i)
      {
        // negative y
        vertices[of + dim * i] =
          delta_x * (i + 1) + rot_center[0]; // increasing x
        vertices[of + dim * i + 1] = rot_center[1] - height * 0.5; // fixed y
        // positive y
        vertices[of + (2 * n_remaining_nodes) + dim * i] =
          vertices[of + dim * i]; // increasing x
        vertices[of + (2 * n_remaining_nodes) + dim * i + 1] =
          rot_center[1] + height * 0.5; // fixed y
      }
  }

  const Vector initial_vertices = vertices;
  // Pass the vertices to preCICE
  precice.setMeshVertices(mesh_name, vertices, vertex_ids);

  for (uint i = 0; i < dummy_stress.size() / dim; ++i)
    {
      dummy_stress[2 * i]     = 0;
      dummy_stress[2 * i + 1] = -4;
    }

  if (precice.requiresInitialData())
    precice.writeData(mesh_name, data_write_name, vertex_ids, dummy_stress);


  // initialize the Solverinterface
  precice.initialize();
  double dt = precice.getMaxTimeStepSize();

  std::cout << "Dummy tester reading initial data \n";
  precice.readData(mesh_name, data_read_name, vertex_ids, dt, displacement);

  DataContainer data_container;

  // Start time loop
  double time = 0;
  while (precice.isCouplingOngoing())
    {
      // Store
      if (precice.requiresWritingCheckpoint())
        data_container.store_data(vertices, old_moment, theta, theta_dot);

      // Solve system
      //      solver.solve(forces, vertices, old_moment, theta, theta_dot, dt);
      // Advance
      std::cout << "Dummy tester writing coupling data \n";
      precice.writeData(mesh_name, data_write_name, vertex_ids, dummy_stress);

      std::cout << "Dummy tester advancing in time\n";
      precice.advance(dt);
      dt = precice.getMaxTimeStepSize();

      std::cout << "Dummy tester reading coupling data \n";
      precice.readData(mesh_name, data_read_name, vertex_ids, dt, displacement);

      // Reload
      if (precice.requiresReadingCheckpoint())
        data_container.reload_data(vertices, old_moment, theta, theta_dot);

      // Increment
      if (precice.isTimeWindowComplete())
        time += dt;
    }

  // Print final read data
  for (uint i = 0; i < displacement.size() / dim; ++i)
    std::cout << std::setprecision(10) << "Vertex:" << std::setw(3) << i
              << " X: " << std::fixed << std::setw(13) << displacement[dim * i]
              << " Y: " << std::setw(13) << displacement[dim * i + 1] << "\n";
  std::cout << std::endl;

  std::cout << "Dummy tester closing...\n";

  return 0;
}
