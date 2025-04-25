#pragma once

#include <generic/common.hpp>
#include <map>
#include <cstdlib>

#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

namespace Ensign {

struct nc_writer {

    nc_writer(string filename, vector<long> extensions, vector<string> d_names); 
    ~nc_writer();

    void add_var(string name, vector<string> dims);
    
    void start_write_mode();
    void write(string name, double* data);

private:
    int ncid;
    std::map<string,int> dim_ids;
    std::map<string,int> var_ids;
};

#define NETCDF_CHECK(e)                                                                \
    {                                                                                  \
        int res = e;                                                                   \
        if (res != NC_NOERR) {                                                         \
            std::cout << "NetCDF Error " << __FILE__ << ":" << __LINE__ << ": "        \
                      << nc_strerror(res) << std::endl;                                \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
    }

} // namespace Ensign