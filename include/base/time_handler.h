#pragma once

#include <deal.II/base/config.h>

DEAL_II_NAMESPACE_OPEN

// A simple class to store time data. Its functioning is transparent so no
// discussion is necessary. For simplicity we assume a constant time step
// size.
class Time
{
public:
  Time(const double time_end, const double delta_t)
    : timestep(0)
    , time_current(0.0)
    , time_end(time_end)
    , delta_t(delta_t)
  {}

  virtual ~Time() = default;

  double
  current() const
  {
    return time_current;
  }
  double
  end() const
  {
    return time_end;
  }
  double
  get_delta_t() const
  {
    return delta_t;
  }
  unsigned int
  get_timestep() const
  {
    return timestep;
  }
  void
  increment()
  {
    time_current += delta_t;
    ++timestep;
  }

private:
  unsigned int timestep;
  double       time_current;
  const double time_end;
  const double delta_t;
};

DEAL_II_NAMESPACE_CLOSE
