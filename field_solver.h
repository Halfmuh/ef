#ifndef _FIELD_SOLVER_H_
#define _FIELD_SOLVER_H_

#include <petscksp.h>
#include <boost/multi_array.hpp>
#include "spatial_mesh.h"
#include "inner_region.h"

class Field_solver {
  public:
    Field_solver( Spatial_mesh &spat_mesh, Inner_region &inner_region );
    void eval_potential( Spatial_mesh &spat_mesh, Inner_region &inner_region );
    void eval_fields_from_potential( Spatial_mesh &spat_mesh );
    virtual ~Field_solver();
  private:
    Vec phi_vec, rhs;
    Mat A;
    KSP ksp;
    PC pc;
    void alloc_petsc_vector( Vec *x, PetscInt size, const char *name );
    void alloc_petsc_matrix( Mat *A, PetscInt nrow, PetscInt ncol, PetscInt nonzero_per_row );
    void construct_equation_matrix( Mat *A,
				    int nx, int ny, int nz,
				    double dx, double dy, double dz,
				    Inner_region &inner_region );
    void construct_equation_matrix_in_full_domain( Mat *A,
						   int nx, int ny, int nz,
						   double dx, double dy, double dz );
    void cross_out_nodes_occupied_by_objects( Mat *A,
					      int nx, int ny, int nz,
					      Inner_region &inner_region );
    void modify_equation_near_object_boundaries( Mat *A,
						 int nx, int ny, int nz,
						 double dx, double dy, double dz,
						 Inner_region &inner_region );
    void create_solver_and_preconditioner( KSP *ksp, PC *pc, Mat *A );
    void construct_d2dx2_in_3d( Mat *d2dx2_3d, int nx, int ny, int nz );
    void construct_d2dy2_in_3d( Mat *d2dy2_3d, int nx, int ny, int nz );
    void construct_d2dz2_in_3d( Mat *d2dz2_3d, int nx, int ny, int nz );
    void multiply_pattern_along_diagonal( Mat *result, Mat *pattern, int pt_size, int n_times );
    void construct_d2dx2_in_2d( Mat *d2dx2, int nx, int ny );
    void construct_d2dy2_in_2d( Mat *d2dy2, int nx, int ny );
    // Solve potential
    void solve_poisson_eqn( Spatial_mesh &spat_mesh, Inner_region &inner_region );
    void init_rhs_vector( Spatial_mesh &spat_mesh, Inner_region &inner_region );
    void init_rhs_vector_in_full_domain( Spatial_mesh &spat_mesh );
    void set_rhs_at_nodes_occupied_by_objects( Spatial_mesh &spat_mesh, Inner_region &inner_region );
    void modify_rhs_near_object_boundaries( Spatial_mesh &spat_mesh, Inner_region &inner_region );
    void indicies_of_near_boundary_nodes_and_rhs_modifications(
	std::vector<PetscInt> indices_of_nodes_near_boundaries,
	std::vector<PetscScalar> rhs_modification_for_nodes_near_boundaries,
	int nx, int ny, int nz,
	double dx, double dy, double dz,
	Inner_region &inner_region );
    int kronecker_delta( int i,  int j );
    void transfer_solution_to_spat_mesh( Spatial_mesh &spat_mesh );
    // Eval fields from potential
    double boundary_difference( double phi1, double phi2, double dx );
    double central_difference( double phi1, double phi2, double dx );
};

#endif /* _FIELD_SOLVER_H_ */