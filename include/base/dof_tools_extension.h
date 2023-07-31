#pragma once

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q_generic.h>

DEAL_II_NAMESPACE_OPEN

namespace DoFTools
{
  /**
   * Similar to map_dofs_to_support_points, but restricted to the boundary of
   * the given boundary ID
   */
  template <int dim, int spacedim>
  void
  map_boundary_dofs_to_support_points(
    const Mapping<dim, spacedim>                       &mapping,
    const DoFHandler<dim, spacedim>                    &dof_handler,
    std::map<types::global_dof_index, Point<spacedim>> &support_points,
    const ComponentMask                                &in_mask,
    const types::boundary_id                            boundary_id)
  {
    const FiniteElement<dim, spacedim> &fe = dof_handler.get_fe();
    // check whether every fe in the collection has support points
    Assert(fe.has_support_points(),
           typename FiniteElement<dim>::ExcFEHasNoSupportPoints());

    Quadrature<dim - 1> quad(fe.get_unit_face_support_points());

    // Take care of components
    const ComponentMask mask =
      (in_mask.size() == 0 ? ComponentMask(fe.n_components(), true) : in_mask);

    // Now loop over all cells and enquire the support points on each
    // of these. we use dummy quadrature formulas where the quadrature
    // points are located at the unit support points to enquire the
    // location of the support points in real space.
    //
    // The weights of the quadrature rule have been set to invalid
    // values by the used constructor.
    FEFaceValues<dim, spacedim> fe_values(mapping,
                                          fe,
                                          quad,
                                          update_quadrature_points);

    std::vector<types::global_dof_index> local_dof_indices;
    for (const auto &cell : dof_handler.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() == true && face->boundary_id() == boundary_id &&
            cell->is_locally_owned())
          // only work on locally relevant cells
          {
            fe_values.reinit(cell, face);

            local_dof_indices.resize(fe.dofs_per_face);
            face->get_dof_indices(local_dof_indices);

            const std::vector<Point<spacedim>> &points =
              fe_values.get_quadrature_points();

            for (unsigned int i = 0; i < fe.n_dofs_per_face(); ++i)
              {
                const unsigned int dof_comp =
                  fe.face_system_to_component_index(i).first;

                // insert the values into the map if it is a valid component
                if (mask[dof_comp])
                  support_points[local_dof_indices[i]] = points[i];
              }
          }
  }
} // namespace DoFTools

DEAL_II_NAMESPACE_CLOSE
