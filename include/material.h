#pragma once

#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

using namespace dealii;

/**
 * Action of the geometric part of the 4-th order tangent tensor
 * on the $Grad N(x)$
 *
 * In index notation this tensor is $ [j e^{geo}]_{ijkl} = j \delta_{ik}
 * \sigma^{tot}_{jl} = \delta_{ik} \tau^{tot}_{jl} $.
 *
 */
template <int dim, typename NumberType>
inline Tensor<2, dim, NumberType>
egeo_grad(const Tensor<2, dim, NumberType> &grad_Nx,
          const Tensor<2, dim, NumberType> &tau_tot)
{
  // the product is actually  GradN * tau^T but due to symmetry of tau we can do
  // GradN * tau
  return grad_Nx * tau_tot;
}


template <typename number>
number
divide_by_dim(const number &x, const int dim)
{
  return x / dim;
}

template <typename number>
VectorizedArray<number>
divide_by_dim(const VectorizedArray<number> &x, const int dim)
{
  VectorizedArray<number> res(x);
  for (unsigned int i = 0; i < VectorizedArray<number>::n_array_elements; i++)
    res[i] *= 1.0 / dim;

  return res;
}

// As discussed in the literature and step-44, Neo-Hookean materials are a type
// of hyperelastic materials.  The entire domain is assumed to be composed of a
// compressible neo-Hookean material.  This class defines the behaviour of
// this material within a one-field formulation.  Compressible neo-Hookean
// materials can be described by a strain-energy function (SEF) $ \Psi =
// \Psi_{\text{iso}}(\overline{\mathbf{b}}) + \Psi_{\text{vol}}(J)
// $.
//
// The isochoric response is given by $
// \Psi_{\text{iso}}(\overline{\mathbf{b}}) = c_{1} [\overline{I}_{1} - 3] $
// where $ c_{1} = \frac{\mu}{2} $ and $\overline{I}_{1}$ is the first
// invariant of the left- or right-isochoric Cauchy-Green deformation tensors.
// That is $\overline{I}_1 :=\textrm{tr}(\overline{\mathbf{b}})$.  In this
// example the SEF that governs the volumetric response is defined as $
// \Psi_{\text{vol}}(J) = \kappa \frac{1}{4} [ J^2 - 1
// - 2\textrm{ln}\; J ]$,  where $\kappa:= \lambda + 2/3 \mu$ is
// the <a href="http://en.wikipedia.org/wiki/Bulk_modulus">bulk modulus</a>
// and $\lambda$ is <a
// href="http://en.wikipedia.org/wiki/Lam%C3%A9_parameters">Lame's first
// parameter</a>.
//
// The following class will be used to characterize the material we work with,
// and provides a central point that one would need to modify if one were to
// implement a different material model. For it to work, we will store one
// object of this type per quadrature point, and in each of these objects
// store the current state (characterized by the values or measures  of the
// displacement field) so that we can compute the elastic coefficients
// linearized around the current state.
template <int dim, typename NumberType>
class Material_Compressible_Neo_Hook_One_Field
{
public:
  Material_Compressible_Neo_Hook_One_Field(const double       mu,
                                           const double       nu,
                                           const unsigned int formulation)
    : kappa((2.0 * mu * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu)))
    , c_1(mu / 2.0)
    , mu(mu)
    , lambda((2.0 * mu * nu) / (1.0 - 2.0 * nu))
    , formulation(formulation)
  {
    Assert(kappa > 0, ExcInternalError());
    Assert(std::abs((lambda + 2.0 * mu / 3.0) - kappa) < 1e-6,
           ExcInternalError());
  }

  ~Material_Compressible_Neo_Hook_One_Field()
  {}

  // The first function is the total energy
  // $\Psi = \Psi_{\textrm{iso}} + \Psi_{\textrm{vol}}$.
  NumberType
  get_Psi(const NumberType &                         det_F,
          const SymmetricTensor<2, dim, NumberType> &b_bar,
          const SymmetricTensor<2, dim, NumberType> &b) const
  {
    if (formulation == 0)
      return get_Psi_vol(det_F) + get_Psi_iso(b_bar);
    else if (formulation == 1)
      {
        const NumberType ln_J = std::log(det_F);
        return mu / 2.0 * (trace(b) - dim - 2.0 * ln_J) - lambda * ln_J * ln_J;
      }
    else
      AssertThrow(false, ExcMessage("Unknown material formulation"));
  }

  // The second function determines the Kirchhoff stress $\boldsymbol{\tau}
  // = \boldsymbol{\tau}_{\textrm{iso}} + \boldsymbol{\tau}_{\textrm{vol}}$
  template <typename OutputType>
  void get_tau(SymmetricTensor<2, dim, OutputType> &      tau,
               const OutputType &                         det_F,
               const SymmetricTensor<2, dim, OutputType> &b_bar,
               const SymmetricTensor<2, dim, OutputType> &b)
  {
    tau = OutputType();

    if (formulation == 0)
      {
        // FIXME: combine with the act_Jc where we need tau_bar etc:

        // See Holzapfel p231 eq6.98 onwards

        // The following functions are used internally in determining the result
        // of some of the public functions above. The first one determines the
        // volumetric Kirchhoff stress $\boldsymbol{\tau}_{\textrm{vol}}$.
        // Note the difference in its definition when compared to step-44.
        const OutputType tmp = OutputType(get_dPsi_vol_dJ(det_F) * det_F);

        // Next, determine the isochoric Kirchhoff stress
        // $\boldsymbol{\tau}_{\textrm{iso}} =
        // \mathcal{P}:\overline{\boldsymbol{\tau}}$

        SymmetricTensor<2, dim, OutputType> tau_bar = b_bar * (2.0 * c_1);
        OutputType                          tr      = trace(tau_bar);
        for (unsigned int d = 0; d < dim; ++d)
          tau[d][d] = tmp - divide_by_dim(tr, dim);

        tau += tau_bar;
      }
    else if (formulation == 1)
      {
        // const NumberType tmp_1 = mu - 2.0*lambda*std::log(det_F);
        // tau  = mu*b;
        // tau -= tmp_1*Physics::Elasticity::StandardTensors<dim>::I;

        tau                  = mu * b;
        const OutputType tmp = mu - 2.0 * lambda * std::log(det_F);
        for (unsigned int d = 0; d < dim; ++d)
          tau[d][d] -= tmp;
      }
    else
      AssertThrow(false, ExcMessage("Unknown material formulation"));
  }

  // The action of the fourth-order material elasticity tensor in the spatial
  // setting on symmetric tensor.
  // $\mathfrak{c}$ is calculated from the SEF $\Psi$ as $ J
  // \mathfrak{c}_{ijkl} = F_{iA} F_{jB} \mathfrak{C}_{ABCD} F_{kC} F_{lD}$
  // where $ \mathfrak{C} = 4 \frac{\partial^2 \Psi(\mathbf{C})}{\partial
  // \mathbf{C} \partial \mathbf{C}}$
  SymmetricTensor<2, dim, NumberType>
  act_Jc(const NumberType &                         det_F,
         const SymmetricTensor<2, dim, NumberType> &b_bar,
         const SymmetricTensor<2, dim, NumberType> & /*b*/,
         const SymmetricTensor<2, dim, NumberType> &src) const
  {
    SymmetricTensor<2, dim, NumberType> res;

    if (formulation == 0)
      {
        const NumberType tr = trace(src);

        SymmetricTensor<2, dim, NumberType> dev_src(src);
        for (unsigned int i = 0; i < dim; ++i)
          dev_src[i][i] -= divide_by_dim(tr, dim);

        // 1) The volumetric part of the tangent $J
        // \mathfrak{c}_\textrm{vol}$. Again, note the difference in its
        // definition when compared to step-44. The extra terms result from two
        // quantities in $\boldsymbol{\tau}_{\textrm{vol}}$ being dependent on
        // $\boldsymbol{F}$.
        // See Holzapfel p265

        // the term with the 4-th order symmetric tensor which gives symmetric
        // part of the tensor it acts on
        res = src;
        res *= -det_F * (2.0 * get_dPsi_vol_dJ(det_F));

        // term with IxI results in trace of the tensor times I
        const NumberType tmp =
          det_F * (get_dPsi_vol_dJ(det_F) + det_F * get_d2Psi_vol_dJ2(det_F)) *
          tr;
        for (unsigned int i = 0; i < dim; ++i)
          res[i][i] += tmp;

        // 2) the isochoric part of the tangent $J
        // \mathfrak{c}_\textrm{iso}$:

        // trace of fictitious Kirchhoff stress
        // $\overline{\boldsymbol{\tau}}$:
        // 2.0 * c_1 * b_bar
        const NumberType tr_tau_bar = trace(b_bar) * 2.0 * c_1;

        // The isochoric Kirchhoff stress
        // $\boldsymbol{\tau}_{\textrm{iso}} =
        // \mathcal{P}:\overline{\boldsymbol{\tau}}$:
        SymmetricTensor<2, dim, NumberType> tau_iso(b_bar);
        tau_iso = tau_iso * (2.0 * c_1);
        for (unsigned int i = 0; i < dim; ++i)
          tau_iso[i][i] -= divide_by_dim(tr_tau_bar, dim);

        // term with deviatoric part of the tensor
        res += ((2.0 / dim) * tr_tau_bar) * dev_src;

        // term with tau_iso_x_I + I_x_tau_iso
        res -= ((2.0 / dim) * tr) * tau_iso;
        const NumberType tau_iso_src = tau_iso * src;
        for (unsigned int i = 0; i < dim; ++i)
          res[i][i] -= (2.0 / dim) * tau_iso_src;

        // c_bar==0 so we don't have a term with it.
      }
    else if (formulation == 1)
      {
        // SymmetricTensor<4,dim,NumberType> Jc;
        // const NumberType tmp_1 = mu - 2.0*lambda*std::log(det_F);
        // Jc += 2.0*tmp_1*Physics::Elasticity::StandardTensors<dim>::S;
        // Jc += 2.0*lambda*Physics::Elasticity::StandardTensors<dim>::IxI;
        // res = Jc*src;

        res = 2.0 * (mu - 2.0 * lambda * std::log(det_F)) * src;
        const NumberType tmp = 2.0 * lambda * trace(src);
        for (unsigned int i = 0; i < dim; ++i)
          res[i][i] += tmp;
      }
    else
      AssertThrow(false, ExcMessage("Unknown material formulation"));

    return res;
  }

  // Define constitutive model parameters $\kappa$ (bulk modulus) and the
  // neo-Hookean model parameter $c_1$:
  const double       kappa;
  const double       c_1;
  const double       mu;
  const double       lambda;
  const unsigned int formulation;

private:
  // Value of the volumetric free energy
  NumberType
  get_Psi_vol(const NumberType &det_F) const
  {
    return (kappa / 4.0) * (det_F * det_F - 1.0 - 2.0 * std::log(det_F));
  }

  // Value of the isochoric free energy
  NumberType
  get_Psi_iso(const SymmetricTensor<2, dim, NumberType> &b_bar) const
  {
    return c_1 * (trace(b_bar) - dim);
  }

  // Derivative of the volumetric free energy with respect to
  // $J$ return $\frac{\partial
  // \Psi_{\text{vol}}(J)}{\partial J}$
  template <typename OutputType>
  OutputType
  get_dPsi_vol_dJ(const OutputType &det_F) const
  {
    return (kappa / 2.0) * (det_F - 1.0 / det_F);
  }

  // Second derivative of the volumetric free energy wrt $J$. We
  // need the following computation explicitly in the tangent so we make it
  // public.  We calculate $\frac{\partial^2
  // \Psi_{\textrm{vol}}(J)}{\partial J \partial
  // J}$
  NumberType
  get_d2Psi_vol_dJ2(const NumberType &det_F) const
  {
    return ((kappa / 2.0) * (1.0 + 1.0 / (det_F * det_F)));
  }
};
