#include <generic/tree.hpp>

namespace Ensign {

// TODO: Add cast functions `to_internal_node`, `to_external_node`

template <> void internal_node<double>::Initialize(int ncid)
{
    // read Q
    int id_Q;
    NETCDF_CHECK(nc_inq_varid(ncid, "Q", &id_Q));
    NETCDF_CHECK(nc_get_var_double(ncid, id_Q, Q.data()));
    return;
}

template <> void external_node<double>::Initialize(int ncid)
{
    // read X
    int id_X;
    NETCDF_CHECK(nc_inq_varid(ncid, "X", &id_X));
    NETCDF_CHECK(nc_get_var_double(ncid, id_X, X.data()));
    return;
}

template <>
void internal_node<double>::Write(int ncid, int id_r_in,
                                  std::array<int, 2> id_r_out) const
{
    int varid_Q;
    // NOTE: netCDF stores arrays in row-major order (but Ensign column-major order)
    int dimids_Q[3] = {id_r_in, id_r_out[1], id_r_out[0]};
    NETCDF_CHECK(nc_def_var(ncid, "Q", NC_DOUBLE, 3, dimids_Q, &varid_Q));
    NETCDF_CHECK(nc_put_var_double(ncid, varid_Q, Q.data()));
}

template <> void external_node<double>::Write(int ncid, int id_r_in, int id_dx) const
{
    int varid_X;
    // NOTE: netCDF stores arrays in row-major order (but Ensign column-major order)
    int dimids_X[2] = {id_r_in, id_dx};
    NETCDF_CHECK(nc_def_var(ncid, "X", NC_DOUBLE, 2, dimids_X, &varid_X));
    NETCDF_CHECK(nc_put_var_double(ncid, varid_X, X.data()));
}

} // namespace Ensign