#pragma once

#include <deal.II/base/function.h>

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
template <int dim, typename Number>
inline Tensor<2, dim, Number>
egeo_grad(const Tensor<2, dim, Number> &grad_Nx,
          const Tensor<2, dim, Number> &tau_tot)
{
  // the product is actually  GradN * tau^T but due to symmetry of tau we can do
  // GradN * tau
  return grad_Nx * tau_tot;
}


template <typename Number>
Number
divide_by_dim(const Number &x, const int dim)
{
  return x / Number(dim);
}

template <typename Number,
          typename VectorizedArrayType = VectorizedArray<Number>>
VectorizedArrayType
divide_by_dim(const VectorizedArrayType &x, const int dim)
{
  VectorizedArrayType res(x);
  for (unsigned int i = 0; i < VectorizedArrayType::size(); i++)
    res[i] *= 1.0 / dim;

  return res;
}

// Helper function in order to evaluate the vectorized point
template <int dim, typename VectorizedArrayType, int n_components = dim>
Tensor<1, n_components, VectorizedArrayType>
evaluate_function(const Function<dim>                   &function,
                  const Point<dim, VectorizedArrayType> &p_vectorized)
{
  AssertDimension(function.n_components, n_components);
  Tensor<1, n_components, VectorizedArrayType> result;
  for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
    {
      Point<dim> p;
      for (unsigned int d = 0; d < dim; ++d)
        p[d] = p_vectorized[d][v];
      for (unsigned int d = 0; d < n_components; ++d)
        result[d][v] = function.value(p, d);
    }
  return result;
}


// Fallback case for general types
template <typename T, typename = void>
struct is_vectorized_array_type : std::false_type
{};

// Specialization for VectorizedArrayType
template <typename T>
struct is_vectorized_array_type<T,
                                std::void_t<decltype(std::declval<T>().size())>>
  : std::true_type
{};



// An outer product of a vector with itself a \bigO a
template <int dim, typename Number>
constexpr SymmetricTensor<2, dim, Number>
outer_product(const Tensor<1, dim, Number> &a)
{
  SymmetricTensor<2, dim, Number> result;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      result[i][j] = a[i] * a[j];
  return result;
}

// the fourth-order tensor Gab(Ma,Mb)
// TODO: we can probably explot symmetry here
template <int dim, typename Number>
constexpr SymmetricTensor<4, dim, Number>
get_G_ab(const SymmetricTensor<2, dim, Number> &Ma,
         const SymmetricTensor<2, dim, Number> &Mb)
{
  SymmetricTensor<4, dim, Number> G_ab{};
  for (unsigned int i = 0; i < dim; ++i)
    {
      // for (unsigned int j = i; j < dim; ++j) // Start from i to exploit
      // symmetry (1)
      for (unsigned int j = 0; j < dim; ++j)
        {
          for (unsigned int k = 0; k < dim; ++k)
            {
              // for (unsigned int l = k; l < dim; ++l) // Start from k to
              // exploit symmetry (2)
              for (unsigned int l = 0; l < dim; ++l)
                {
                  G_ab[i][j][k][l] = Ma[i][k] * Mb[j][l] + Ma[i][l] * Mb[j][k];
                }
            }
        }
    }
  return G_ab;
}


// the fourth-order tensor Gab(Ma,Mb)
// TODO: we can probably explot symmetry here
template <int dim, typename Number>
constexpr Tensor<4, dim, Number>
get_F_a_bc(const Tensor<2, dim, Number> &Ma,
           const Tensor<2, dim, Number> &MbTMc)
{
  Tensor<4, dim, Number> F_a_bc{};
  for (unsigned int i = 0; i < dim; ++i)
    {
      // for (unsigned int j = i; j < dim; ++j) // Start from i to exploit
      // symmetry (1)
      for (unsigned int j = 0; j < dim; ++j)
        {
          for (unsigned int k = 0; k < dim; ++k)
            {
              // for (unsigned int l = k; l < dim; ++l) // Start from k to
              // exploit symmetry (2)
              for (unsigned int l = 0; l < dim; ++l)
                {
                  F_a_bc[i][j][k][l] =
                    Ma[i][k] * MbTMc[j][l] + Ma[i][l] * MbTMc[j][k];
                }
            }
        }
    }
  return F_a_bc;
}

template <int dim, typename Number>
constexpr SymmetricTensor<4, dim, Number>
extractSymmetry(const Tensor<4, dim, Number> &in)
{
  SymmetricTensor<4, dim, Number> res{};
  for (unsigned int i = 0; i < dim; ++i)
    {
      // for (unsigned int j = i; j < dim; ++j) // Start from i to exploit
      // symmetry (1)
      for (unsigned int j = 0; j < dim; ++j)
        {
          for (unsigned int k = 0; k < dim; ++k)
            {
              // for (unsigned int l = k; l < dim; ++l) // Start from k to
              // exploit symmetry (2)
              for (unsigned int l = 0; l < dim; ++l)
                {
                  res[i][j][k][l] = in[i][j][k][l];
                }
            }
        }
    }
  return res;
}

// Fiber directions in reference coordinates (m_x)
template <int dim, typename Number>
class FiberDirections : public Functions::ConstantFunction<dim>
{
public:
  FiberDirections(const std::vector<Number> &direction_vector)
    : Functions::ConstantFunction<dim, Number>(direction_vector)
  {}
};

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
template <int dim, typename Number>
class Material_Compressible_Neo_Hook_One_Field
{
public:
  Material_Compressible_Neo_Hook_One_Field(const double       mu,
                                           const double       nu,
                                           const double       rho,
                                           const double       alpha_1,
                                           const unsigned int formulation)
    : kappa((2.0 * mu * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu)))
    , c_1(mu / 2.0)
    , mu(mu)
    , rho(rho)
    , rho_alpha(rho * alpha_1)
    , lambda((2.0 * mu * nu) / (1.0 - 2.0 * nu))
    , formulation(formulation)
    , m_x(std::vector<double>{-0.4523865912591927,
                              -0.05863679298792379,
                              -0.8898921836700103})
    , k1_tendon(92.779e2) //[N/cm^2 = 1e-2 MPa]
    , k2_tendon(305.87e2) //[N/cm^2 = 1e-2 MPa]
  {
    Assert(kappa > 0, ExcInternalError());
    Assert(std::abs((lambda + 2.0 * mu / 3.0) - kappa) < 1e-6,
           ExcInternalError());
  }

  template <typename T>
  T
  get_dPsi_f_dI_4(T I_4) const
  {
    T dPsi_dI = 0;
    if (I_4 >= 1)
      {
        dPsi_dI = 2 * k1_tendon * (I_4 - 1) +
                  3 * k2_tendon * Utilities::fixed_power<2>(I_4 - 1);
      }
    return dPsi_dI;
  }

  template <typename T>
  T
  get_d2Psi_f_dI2_4(T I_4) const
  {
    T d2Psi_dI2 = 0;
    if (I_4 >= 1)
      {
        d2Psi_dI2 = 2 * k1_tendon + 6 * k2_tendon * (I_4 - 1);
      }
    return d2Psi_dI2;
  }

  // The first function is the total energy
  // $\Psi = \Psi_{\textrm{iso}} + \Psi_{\textrm{vol}}$.
  Number
  get_Psi(const Number                          &det_F,
          const SymmetricTensor<2, dim, Number> &b_bar,
          const SymmetricTensor<2, dim, Number> &b) const
  {
    if (formulation == 0)
      return get_Psi_vol(det_F) + get_Psi_iso(b_bar);
    else if (formulation == 1)
      {
        const Number ln_J = std::log(det_F);
        return mu / 2.0 * (trace(b) - dim - 2.0 * ln_J) - lambda * ln_J * ln_J;
      }
    else
      AssertThrow(false, ExcMessage("Unknown material formulation"));
  }

  // The second function determines the Kirchhoff stress $\boldsymbol{\tau}
  // = \boldsymbol{\tau}_{\textrm{iso}} + \boldsymbol{\tau}_{\textrm{vol}}$
  template <typename OutputType>
  void
  get_tau(SymmetricTensor<2, dim, OutputType>       &tau,
          const OutputType                          &det_F,
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
  SymmetricTensor<2, dim, Number>
  act_Jc(const Number                          &det_F,
         const SymmetricTensor<2, dim, Number> &b_bar,
         const SymmetricTensor<2, dim, Number> & /*b*/,
         const SymmetricTensor<2, dim, Number> &src) const
  {
    SymmetricTensor<2, dim, Number> res;

    if (formulation == 0)
      {
        const Number tr = trace(src);

        SymmetricTensor<2, dim, Number> dev_src(src);
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
        const Number tmp =
          det_F * (get_dPsi_vol_dJ(det_F) + det_F * get_d2Psi_vol_dJ2(det_F)) *
          tr;
        for (unsigned int i = 0; i < dim; ++i)
          res[i][i] += tmp;

        // 2) the isochoric part of the tangent $J
        // \mathfrak{c}_\textrm{iso}$:

        // trace of fictitious Kirchhoff stress
        // $\overline{\boldsymbol{\tau}}$:
        // 2.0 * c_1 * b_bar
        const Number tr_tau_bar = trace(b_bar) * 2.0 * c_1;

        // The isochoric Kirchhoff stress
        // $\boldsymbol{\tau}_{\textrm{iso}} =
        // \mathcal{P}:\overline{\boldsymbol{\tau}}$:
        SymmetricTensor<2, dim, Number> tau_iso(b_bar);
        tau_iso = tau_iso * (2.0 * c_1);
        for (unsigned int i = 0; i < dim; ++i)
          tau_iso[i][i] -= divide_by_dim(tr_tau_bar, dim);

        // term with deviatoric part of the tensor
        res += ((2.0 / dim) * tr_tau_bar) * dev_src;

        // term with tau_iso_x_I + I_x_tau_iso
        res -= ((2.0 / dim) * tr) * tau_iso;
        const Number tau_iso_src = tau_iso * src;
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

        res              = 2.0 * (mu - 2.0 * lambda * std::log(det_F)) * src;
        const Number tmp = 2.0 * lambda * trace(src);
        for (unsigned int i = 0; i < dim; ++i)
          res[i][i] += tmp;
      }
    else
      AssertThrow(false, ExcMessage("Unknown material formulation"));

    return res;
  }


  static SymmetricTensor<4, dim, Number>
  get_Fung_A()
  {
    // init by zero
    SymmetricTensor<4, dim, Number> A{};
    A[0][0][0][0] = Number(fung_ca);

    A[1][1][1][1] = A[2][2][2][2] = Number(fung_ct);

    A[0][0][1][1] = A[0][0][2][2] = Number(fung_cat);

    A[1][1][2][2] = Number(fung_ctt);

    return A;
  }

  // the first derivative dPsi_Fung/d epsilon
  SymmetricTensor<2, dim, Number>
  get_T_Fung(const SymmetricTensor<2, dim, Number> &epsilon) const
  {
    Number                          C_exp_Q = this->get_C_times_exp_Q(epsilon);
    SymmetricTensor<4, dim, Number> coeff_Fung_A = C_exp_Q * get_Fung_A();
    return coeff_Fung_A * epsilon;
  }

  // the second derivative d2Psi_Fung/d epsilon2
  SymmetricTensor<4, dim, Number>
  get_E_Fung(const SymmetricTensor<2, dim, Number> &epsilon) const
  {
    Number                          C_exp_Q = this->get_C_times_exp_Q(epsilon);
    SymmetricTensor<4, dim, Number> coeff_Fung_A    = C_exp_Q * get_Fung_A();
    SymmetricTensor<2, dim, Number> A_times_epsilon = get_Fung_A() * epsilon;
    SymmetricTensor<4, dim, Number> outer_product_part =
      Number(2) * C_exp_Q * outer_product(A_times_epsilon, A_times_epsilon);
    return coeff_Fung_A + outer_product_part;
  }

  // the projection Tensor P (H in the paper)
  SymmetricTensor<4, dim, Number>
  get_H_Fung() const
  {
    // we have the case of m = 0
    SymmetricTensor<4, dim, Number> d_M_times_M{};
    // contributions from G
    SymmetricTensor<4, dim, Number> theta_G{};
    for (int i = 0; i < dim; ++i)
      {
        // d is 1 / lambda
        Number                          d = get_d_a(this->eigenvalues_[i]);
        SymmetricTensor<4, dim, Number> d_Mi_times_Mi =
          outer_product(d * this->eigenbase_[i], this->eigenbase_[i]);
        d_M_times_M += d_Mi_times_Mi;

        for (int j = 0; j < dim; ++j)
          {
            if (i != j)
              {
                SymmetricTensor<4, dim, Number> G_ij =
                  get_G_ab(this->eigenbase_[i], this->eigenbase_[j]);
                theta_G +=
                  get_theta(this->eigenvalues_[i], this->eigenvalues_[j]) *
                  G_ij;
              }
          }
      }

    return d_M_times_M + theta_G;
  }

  // double contraction of T with the rank six tensor L
  SymmetricTensor<4, dim, Number>
  get_T_L_Fung(const SymmetricTensor<2, dim, Number> &T_Fung,
               const SymmetricTensor<2, dim, Number> & /*C*/) const
  {
    SymmetricTensor<4, dim, Number> res{};

    SymmetricTensor<4, dim, Number> f_T_times_M{};
    for (int a = 0; a < dim; ++a)
      {
        // deal.II performs automatically the double contraction for symmetric
        // tensors
        Number f_T_M =
          get_f_a(this->eigenvalues_[a]) * (T_Fung * this->eigenbase_[a]);
        SymmetricTensor<2, dim, Number> f_T_M_M = f_T_M * this->eigenbase_[a];
        res += outer_product(f_T_M_M, this->eigenbase_[a]);
      }

    Tensor<4, dim, Number> tmp{};
    // eta doesn't depend on indices
    Number eta = get_eta();
    for (int a = 0; a < dim; ++a)
      {
        for (int b = 0; b < dim; ++b)
          {
            if (b != a)
              {
                for (int c = 0; c < dim; ++c)
                  {
                    if (c != a && c != b)
                      {
                        Tensor<4, dim, Number> sum{};
                        Tensor<2, dim, Number> M_a =
                          Tensor<2, dim, Number>(this->eigenbase_[a]);
                        Tensor<2, dim, Number> M_b =
                          Tensor<2, dim, Number>(this->eigenbase_[b]);
                        Tensor<2, dim, Number> M_c =
                          Tensor<2, dim, Number>(this->eigenbase_[c]);
                        sum += get_F_a_bc(M_a, M_b * T_Fung * M_c);
                        sum += get_F_a_bc(M_b * T_Fung * M_c, M_a);
                        tmp += Number(2) * eta * sum;
                      }
                  }
              }
          }
      }

    for (int a = 0; a < dim; ++a)
      {
        for (int b = 0; b < dim; ++b)
          {
            if (b != a)
              {
                Tensor<4, dim, Number> sum{};
                Tensor<2, dim, Number> M_a =
                  Tensor<2, dim, Number>(this->eigenbase_[a]);
                Tensor<2, dim, Number> M_b =
                  Tensor<2, dim, Number>(this->eigenbase_[b]);
                sum += get_F_a_bc(M_a, M_b * T_Fung * M_b);
                sum += get_F_a_bc(M_b * T_Fung * M_b, M_a);

                sum += get_F_a_bc(M_b, M_a * T_Fung * M_b);
                sum += get_F_a_bc(M_a * T_Fung * M_b, M_b);

                sum += get_F_a_bc(M_b, M_b * T_Fung * M_a);
                sum += get_F_a_bc(M_b * T_Fung * M_a, M_b);

                tmp += Number(2) *
                       get_xi(this->eigenvalues_[a], this->eigenvalues_[b]) *
                       sum;
              }
          }
      }
    res += extractSymmetry(tmp);
    return res;
  }

  // natural strain
  SymmetricTensor<2, dim, Number>
  get_epsilon(const SymmetricTensor<2, dim, Number> &C)
  {
    SymmetricTensor<2, dim, Number> res{};
    this->compute_eigenvalues_and_eigensystem(C);

    for (int a = 0; a < dim; ++a)
      {
        Number e_a = this->get_e_a(this->eigenvalues_[a]);
        res += e_a * this->eigenbase_[a];
      }
    return res;
  }



  // Define constitutive model parameters $\kappa$ (bulk modulus) and the
  // neo-Hookean model parameter $c_1$:
  const double                       kappa;
  const double                       c_1;
  const double                       mu;
  const double                       rho;
  const double                       rho_alpha;
  const double                       lambda;
  const unsigned int                 formulation;
  const FiberDirections<dim, double> m_x;
  const double                       k1_tendon;
  const double                       k2_tendon;

private:
  // fung model
  static constexpr double                 fung_c   = 0.998;
  static constexpr double                 fung_ca  = 0.1492;
  static constexpr double                 fung_ct  = 0.147;
  static constexpr double                 fung_cat = 0.964;
  static constexpr double                 fung_ctt = 0.1124;
  std::array<Number, dim>                 eigenvalues_;
  std::array<Tensor<1, dim, Number>, dim> eigenvectors_;
  // corresponding eigenbase
  std::array<SymmetricTensor<2, dim, Number>, dim> eigenbase_;

  void
  compute_eigenvalues_and_eigensystem(const SymmetricTensor<2, dim, Number> &C)
  {
    std::array<std::pair<Number, Tensor<1, dim, Number>>, dim> eigensystem =
      get_eigensystem(C);
    this->eigenbase_ = get_eigenvalue_bases(eigensystem);
    for (int d = 0; d < dim; ++d)
      {
        this->eigenvalues_[d]  = eigensystem[d].first;
        this->eigenvectors_[d] = eigensystem[d].second;
      }
  }

  // the eigenvectors function is unfortunately not instantiated for
  // vectorized types
  std::array<std::pair<Number, Tensor<1, dim, Number>>, dim>
  get_eigensystem(const SymmetricTensor<2, dim, Number> &C)
  {
    if constexpr (is_vectorized_array_type<Number>::value)
      {
        std::array<std::pair<Number, Tensor<1, dim, Number>>, dim> res;
        // first unvectorize the matrix
        for (std::size_t v = 0; v < Number::size(); ++v)
          {
            SymmetricTensor<2, dim, typename Number::value_type> C_v{};
            for (int i = 0; i < dim; ++i)
              for (int j = i; j < dim; ++j)
                {
                  {
                    C_v[i][j] = C[i][j][v];
                  }
                }
            // compute the eigenvectors
            std::array<std::pair<typename Number::value_type,
                                 Tensor<1, dim, typename Number::value_type>>,
                       dim>
              eigen_system = eigenvectors(C_v);
            // and vectorize again
            for (int d = 0; d < dim; ++d)
              {
                res[d].first[v] = eigen_system[d].first;
                for (int i = 0; i < dim; ++i)
                  {
                    res[d].second[i][v] = eigen_system[d].second[i];
                  }
              }
          }
        return res;
      }
    else
      {
        return eigenvectors(C);
      }
  }

  // the eigenvalue bases n x n, stored in an array
  std::array<SymmetricTensor<2, dim, Number>, dim>
  get_eigenvalue_bases(
    const std::array<std::pair<Number, Tensor<1, dim, Number>>, dim>
      &eigen_system) const
  {
    std::array<SymmetricTensor<2, dim, Number>, dim> bases;
    for (int i = 0; i < dim; ++i)
      bases[i] = outer_product(eigen_system[i].second);

    return bases;
  }

  Number
  get_eta() const
  {
    Number eta;
    if constexpr (is_vectorized_array_type<Number>::value)
      {
        for (std::size_t v = 0; v < Number::size(); ++v)
          {
            std::array<typename Number::value_type, dim> val;
            for (int d = 0; d < dim; ++d)
              {
                val[d] = this->eigenvalues_[d][v];
              }
            eta[v] = get_eta_impl(val);
          }
      }
    else
      {
        eta = get_eta_impl(this->eigenvalues_);
      }
    return eta;
  }

  // the coefficient eta
  template <typename T>
  T
  get_eta_impl(const std::array<T, dim> &eigenvalues) const
  {
    T eta{};
    if constexpr (dim == 2)
      {
        AssertThrow(false, ExcNotImplemented());
        return 1;
      }
    else
      {
        bool same_a_and_b = std::fabs(eigenvalues[0] - eigenvalues[1]) < 1e-12;
        bool same_a_and_c = std::fabs(eigenvalues[0] - eigenvalues[2]) < 1e-12;
        bool same_b_and_c = std::fabs(eigenvalues[1] - eigenvalues[2]) < 1e-12;
        bool all_same     = same_a_and_b && same_a_and_c && same_b_and_c;

        if (all_same)
          {
            eta = 0.125 * get_f_a(eigenvalues[0]);
          }
        else if (same_a_and_b)
          {
            eta = this->get_xi_impl(eigenvalues[2], eigenvalues[0]);
          }
        else if (same_a_and_c)
          {
            eta = this->get_xi_impl(eigenvalues[1], eigenvalues[0]);
          }
        else if (same_b_and_c)
          {
            eta = this->get_xi_impl(eigenvalues[0], eigenvalues[1]);
          }
        else
          {
            eta = 0;
            for (int a = 0; a < dim; ++a)
              {
                for (int b = 0; b < dim; ++b)
                  {
                    if (b != a)
                      {
                        for (int c = 0; c < dim; ++c)
                          {
                            if (c != a && c != b)
                              {
                                eta += get_e_a(eigenvalues[a]) /
                                       (2 * (eigenvalues[a] - eigenvalues[b]) *
                                        (eigenvalues[a] - eigenvalues[c]));
                              }
                          }
                      }
                  }
              }
          }
      }
    return eta;
  }

  Number
  get_xi(const Number &lambda_a, const Number &lambda_b) const
  {
    Number xi;
    if constexpr (is_vectorized_array_type<Number>::value)
      {
        for (std::size_t v = 0; v < Number::size(); ++v)
          {
            xi[v] = get_xi_impl(lambda_a[v], lambda_b[v]);
          }
      }
    else
      {
        xi = get_xi_impl(lambda_a, lambda_b);
      }
    return xi;
  }

  // the coefficient xi
  template <typename T>
  T
  get_xi_impl(const T &lambda_a, const T &lambda_b) const
  {
    T res;
    T lambda_diff = lambda_a - lambda_b;
    if (std::fabs(lambda_diff) > 1.0e-12)
      {
        res = (get_theta_impl(lambda_a, lambda_b) - 0.5 * get_d_a(lambda_b)) /
              lambda_diff;
      }
    else
      {
        res = 0.125 * get_f_a(lambda_a);
      }
    return res;
  }


  Number
  get_theta(const Number &lambda_a, const Number &lambda_b) const
  {
    Number theta;
    if constexpr (is_vectorized_array_type<Number>::value)
      {
        for (std::size_t v = 0; v < Number::size(); ++v)
          {
            theta[v] = get_theta_impl(lambda_a[v], lambda_b[v]);
          }
      }
    else
      {
        theta = get_theta_impl(lambda_a, lambda_b);
      }
    return theta;
  }

  // the coefficient theta
  template <typename T>
  T
  get_theta_impl(const T &lambda_a, const T &lambda_b) const
  {
    T res;
    T lambda_diff = lambda_a - lambda_b;
    if (std::fabs(lambda_diff) > 1.0e-12)
      {
        res = (get_e_a(lambda_a) - get_e_a(lambda_b)) / lambda_diff;
      }
    else
      {
        res = 0.5 * get_d_a(lambda_a);
      }
    return res;
  }

  // the eigenvalue coefficient e_a
  template <typename T>
  T
  get_e_a(const T &lambda_a) const
  {
    return 0.5 * std::log(lambda_a);
  }

  template <typename T>
  T
  get_d_a(const T &lambda_a) const
  {
    return 1. / lambda_a;
  }

  template <typename T>
  T
  get_f_a(const T &lambda_a) const
  {
    return -2. / (lambda_a * lambda_a);
  }

  // The coefficient for the stress contribution
  // of the Fung model C*exp(Q)
  Number
  get_C_times_exp_Q(const SymmetricTensor<2, dim, Number> &epsilon) const
  {
    Number Q = get_exponent_Q(epsilon);
    return fung_c * std::exp(Q);
  }

  // the exponential of the Fung model
  // eps : A : eps
  Number
  get_exponent_Q(const SymmetricTensor<2, dim, Number> &epsilon) const
  {
    SymmetricTensor<2, dim, Number> tmp = epsilon * get_Fung_A();
    return tmp * epsilon;
  }
  // Value of the volumetric free energy
  Number
  get_Psi_vol(const Number &det_F) const
  {
    return (kappa / 4.0) * (det_F * det_F - 1.0 - 2.0 * std::log(det_F));
  }

  // Value of the isochoric free energy
  Number
  get_Psi_iso(const SymmetricTensor<2, dim, Number> &b_bar) const
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
  Number
  get_d2Psi_vol_dJ2(const Number &det_F) const
  {
    return ((kappa / 2.0) * (1.0 + 1.0 / (det_F * det_F)));
  }
};
