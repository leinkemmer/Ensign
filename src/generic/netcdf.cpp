#include <generic/netcdf.hpp>

#ifdef __NETCDF__
#include <netcdf.h>

#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}
#endif

nc_writer::nc_writer(string filename, vector<long> extensions, vector<string> d_names) {
  #ifdef __NETCDF__
  int retval;
   
   if ((retval = nc_create(filename.c_str(), NC_CLOBBER, &ncid)))
      ERR(retval);

  for(size_t i=0;i<extensions.size();i++) {
    int dim_id;
    if ((retval = nc_def_dim(ncid, d_names[i].c_str(), extensions[i], &dim_id)))
      ERR(retval);
    dim_ids[d_names[i]] = dim_id;
  }
  #endif
}

nc_writer::~nc_writer() {
  #ifdef __NETCDF__
  int retval;

  if ((retval = nc_close(ncid)))
    ERR(retval);
  #endif
}

void nc_writer::add_var(string name, vector<string> dims) {
  #ifdef __NETCDF__
  int retval;

  vector<int> ids;
  for(size_t i=0;i<dims.size();i++)
    ids.push_back(dim_ids[dims[i]]);

  int varid;
  if ((retval = nc_def_var(ncid, name.c_str(), NC_DOUBLE, dims.size(), &ids[0], &varid)))
    ERR(retval);

  var_ids[name] = varid;
  #endif
}

void nc_writer::start_write_mode() {
  #ifdef __NETCDF__
  int retval;

  if ((retval = nc_enddef(ncid)))
  ERR(retval);
  #endif
}

void nc_writer::write(string name, double* data) {
  #ifdef __NETCDF__
  int retval;
  
  if ((retval = nc_put_var_double(ncid, var_ids[name], data)))
    ERR(retval);
  #endif
}
