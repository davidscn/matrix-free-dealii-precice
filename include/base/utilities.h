#pragma once

#include <deal.II/base/utilities.h>

#include <sys/stat.h>

DEAL_II_NAMESPACE_OPEN

namespace Utilities
{
  int
  create_directory(std::string  pathname,
                   const mode_t mode = (S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH |
                                        S_IXOTH))
  {
    // force trailing / so we can handle everything in loop
    if (pathname[pathname.size() - 1] != '/')
      {
        pathname += '/';
      }

    size_t pre = 0;
    size_t pos;

    while ((pos = pathname.find_first_of('/', pre)) != std::string::npos)
      {
        const std::string subdir = pathname.substr(0, pos++);
        pre                      = pos;

        // if leading '/', first string is 0 length
        if (subdir.size() == 0)
          continue;

        int mkdir_return_value;
        if ((mkdir_return_value = mkdir(subdir.c_str(), mode)) &&
            (errno != EEXIST))
          return mkdir_return_value;
      }

    return 0;
  }
} // namespace Utilities


DEAL_II_NAMESPACE_CLOSE
