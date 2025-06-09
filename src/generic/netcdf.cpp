#include <generic/netcdf.hpp>

namespace Ensign {

#include <netcdf.h>

#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

nc_writer::nc_writer(string filename, vector<long> extensions, vector<string> d_names) {
  int retval;
   
   if ((retval = nc_create(filename.c_str(), NC_CLOBBER, &ncid)))
      ERR(retval);

  for(size_t i=0;i<extensions.size();i++) {
    int dim_id;
    if ((retval = nc_def_dim(ncid, d_names[i].c_str(), extensions[i], &dim_id)))
      ERR(retval);
    dim_ids[d_names[i]] = dim_id;
  }
}

nc_writer::~nc_writer() {
  int retval;

  if ((retval = nc_close(ncid)))
    ERR(retval);
}

void nc_writer::add_var(string name, vector<string> dims) {
  int retval;

  vector<int> ids;
  for(size_t i=0;i<dims.size();i++)
    ids.push_back(dim_ids[dims[i]]);

  int varid;
  if ((retval = nc_def_var(ncid, name.c_str(), NC_DOUBLE, dims.size(), &ids[0], &varid)))
    ERR(retval);

  var_ids[name] = varid;
}

void nc_writer::start_write_mode() {
  int retval;

  if ((retval = nc_enddef(ncid)))
  ERR(retval);
}

void nc_writer::write(string name, double* data) {
  int retval;
  
  if ((retval = nc_put_var_double(ncid, var_ids[name], data)))
    ERR(retval);
}

} // namespace Ensign