#include "FieldSolver.cuh"




__constant__ double3 d_cell_size[1];
__constant__ int3 d_n_nodes[1];

__constant__ double dev_dxdxdydy[1];
__constant__ double dev_dxdxdzdz[1];
__constant__ double dev_dydydzdz[1];
__constant__ double dev_dxdxdydydzdz[1];

__constant__ int dev_end[1];

__device__ int GetIdx() {
	//int xStepthread = 1;
	int xStepBlock = blockDim.x;
	int yStepThread = d_n_nodes[0].x;
	int yStepBlock = yStepThread * blockDim.y;
	int zStepThread = d_n_nodes[0].x * d_n_nodes[0].y;
	int zStepBlock = zStepThread * blockDim.z;
	return (threadIdx.x + blockIdx.x * xStepBlock) +
		(threadIdx.y * yStepThread + blockIdx.y * yStepBlock) +
		(threadIdx.z * zStepThread + blockIdx.z * zStepBlock);
}

__device__ double GradientComponent(double phi1, double phi2, double cell_side_size) {
	return ((phi2 - phi1) / cell_side_size);
}

__global__ void SetPhiNextAsCurrent(double* d_phi_current, double* d_phi_next) {
	int idx = GetIdx();
	d_phi_current[idx] = d_phi_next[idx];
}

__global__ void ComputePhiNext(const double* d_phi_current, const double* d_charge, double* d_phi_next) {
	int idx = GetIdx();
	int offset_Dx = 1;
	//todo rewrite usind device n_nodes.x/y/z
	int offset_Dy = d_n_nodes[0].x;
	int offset_Dz = d_n_nodes[0].x*d_n_nodes[0].y;

	int prev_neighbour_idx;
	int next_neighbour_idx;

	double denom = (double)2 * (dev_dxdxdydy[0] + dev_dxdxdzdz[0] + dev_dydydzdz[0]);

	prev_neighbour_idx = max(idx + offset_Dx, 0);
	next_neighbour_idx = min(idx + offset_Dx, dev_end[0]);
	d_phi_next[idx] =
		(d_phi_current[next_neighbour_idx] + d_phi_current[prev_neighbour_idx])*dev_dydydzdz[0];

	prev_neighbour_idx = max(idx + offset_Dy, 0);
	next_neighbour_idx = min(idx + offset_Dy, dev_end[0]);
	d_phi_next[idx] +=
		(d_phi_current[next_neighbour_idx] + d_phi_current[prev_neighbour_idx])*dev_dxdxdzdz[0];

	prev_neighbour_idx = max(idx + offset_Dz, 0);
	next_neighbour_idx = min(idx + offset_Dz, dev_end[0]);
	d_phi_next[idx] +=
		(d_phi_current[next_neighbour_idx] + d_phi_current[prev_neighbour_idx])*dev_dxdxdydy[0];

	d_phi_next[idx] += 4.0 * CUDART_PI * d_charge[idx] * dev_dxdxdydydzdz[0];
	d_phi_next[idx] /= denom;

}

__global__ void EvaluateFields(const double* dev_potential, double3* dev_el_field) {
	int idx = GetIdx();

	double3 e = make_double3(0, 0, 0);
	//assuming true=1, false =0 
	//this method is hard to read due avoidance of if-else constructions on device code
	bool is_on_up_border;
	bool is_on_low_border;
	bool is_inside_borders;
	int offset;

	offset = 1;
	is_on_up_border = ((threadIdx.x == 0) && (blockIdx.x == 0));
	is_on_low_border = ((threadIdx.x == (blockDim.x - 1)) && (blockIdx.x == (gridDim.x - 1)));
	is_inside_borders = !(is_on_low_border || is_on_up_border);

	e.x = -(1 / (1 + is_inside_borders)) * GradientComponent(
		dev_potential[idx + (offset*is_on_up_border) - (offset*is_inside_borders)],
		dev_potential[idx - (offset*is_on_low_border) + (offset*is_inside_borders)],
		d_cell_size[0].x);

	offset = d_n_nodes[0].x;
	is_on_up_border = ((threadIdx.x == 0) && (blockIdx.x == 0));
	is_on_low_border = ((threadIdx.x == (blockDim.x - 1)) && (blockIdx.x == (gridDim.x - 1)));
	is_inside_borders = !(is_on_low_border || is_on_up_border);

	e.y = -(1 / (1 + is_inside_borders)) * GradientComponent(
		dev_potential[idx + (offset*is_on_up_border) - (offset*is_inside_borders)],
		dev_potential[idx - (offset*is_on_low_border) + (offset*is_inside_borders)],
		d_cell_size[0].y);

	offset = d_n_nodes[0].y*d_n_nodes[0].x;
	is_on_up_border = ((threadIdx.x == 0) && (blockIdx.x == 0));
	is_on_low_border = ((threadIdx.x == (blockDim.x - 1)) && (blockIdx.x == (gridDim.x - 1)));
	is_inside_borders = !(is_on_low_border || is_on_up_border);

	e.z = -(1 / (1 + is_inside_borders)) * GradientComponent(
		dev_potential[idx + (offset*is_on_up_border) - (offset*is_inside_borders)],
		dev_potential[idx - (offset*is_on_low_border) + (offset*is_inside_borders)],
		d_cell_size[0].z);

	dev_el_field[idx] = e;

}

__global__ void AssertConvergence(const double* d_phi_current, const double* d_phi_next) {
	double rel_diff;
	double abs_diff;
    double abs_tolerance = 1.0e-5;
    double rel_tolerance = 1.0e-12;
	int idx = GetIdx();
	abs_diff = fabs(d_phi_next[idx] - d_phi_current[idx]);
	rel_diff = abs_diff / fabs(d_phi_current[idx]);
	bool converged =((abs_diff <= abs_tolerance) || (rel_diff <= rel_tolerance));

	assert(converged==true);
}

FieldSolver::FieldSolver(SpatialMeshCu &mesh, Inner_regions_manager &inner_regions) :mesh(mesh)
{
	allocate_next_phi();
	copy_constants_to_device();
}

void FieldSolver::allocate_next_phi()
{
	size_t dim = mesh.n_nodes.x * mesh.n_nodes.y * mesh.n_nodes.z;
	cudaError_t cuda_status;

	cuda_status = cudaMalloc<double>(&dev_phi_next, dim);

}

void FieldSolver::copy_constants_to_device() {
	cudaError_t cuda_status;

	cuda_status = cudaMemcpyToSymbol(d_n_nodes, (void*)&mesh.n_nodes, sizeof(dim3),
		cudaMemcpyHostToDevice);
	cuda_status = cudaMemcpyToSymbol(d_cell_size, (void*)&mesh.cell_size, sizeof(double3),
		cudaMemcpyHostToDevice);

	double dxdxdydy = mesh.cell_size.x*mesh.cell_size.x*
		mesh.cell_size.y*mesh.cell_size.y;
	cuda_status = cudaMemcpyToSymbol(dev_dxdxdydy, (void*)&dxdxdydy, sizeof(double),
		cudaMemcpyHostToDevice);

	double dxdxdzdz = mesh.cell_size.x*mesh.cell_size.x*
		mesh.cell_size.z*mesh.cell_size.z;
	cuda_status = cudaMemcpyToSymbol(dev_dxdxdzdz, (void*)&dxdxdzdz, sizeof(double),
		cudaMemcpyHostToDevice);

	double dydydzdz = mesh.cell_size.y*mesh.cell_size.y*
		mesh.cell_size.z*mesh.cell_size.z;
	cuda_status = cudaMemcpyToSymbol(dev_dydydzdz, (void*)&dydydzdz, sizeof(double),
		cudaMemcpyHostToDevice);

	double dxdxdydydzdz = mesh.cell_size.x*mesh.cell_size.x*
		mesh.cell_size.y*mesh.cell_size.y*
		mesh.cell_size.z*mesh.cell_size.z;
	cuda_status = cudaMemcpyToSymbol(dev_dxdxdydydzdz, (void*)&dxdxdydydzdz, sizeof(double),
		cudaMemcpyHostToDevice);

	int end = mesh.n_nodes.x*mesh.n_nodes.y*mesh.n_nodes.z - 1;
	cuda_status = cudaMemcpyToSymbol(dev_end, (void*)&end, sizeof(int),
		cudaMemcpyHostToDevice);
}

void FieldSolver::eval_potential(Inner_regions_manager &inner_regions)
{
	solve_poisson_eqn_Jacobi(inner_regions);
}

void FieldSolver::solve_poisson_eqn_Jacobi(Inner_regions_manager &inner_regions)
{
	max_Jacobi_iterations = 150;
	int iter;

	for (iter = 0; iter < max_Jacobi_iterations; ++iter) {
		single_Jacobi_iteration(inner_regions);
		if (iterative_Jacobi_solutions_converged()) {
			break;
		}
		set_phi_next_as_phi_current();
	}
	if (iter == max_Jacobi_iterations) {
		printf("WARING: potential evaluation did't converge after max iterations!\n");
	}
	set_phi_next_as_phi_current();

	//return;
}

void FieldSolver::single_Jacobi_iteration(Inner_regions_manager &inner_regions)
{
	set_phi_next_at_boundaries();
	compute_phi_next_at_inner_points();
	set_phi_next_at_inner_regions(inner_regions);
}

void FieldSolver::set_phi_next_at_boundaries()
{
	mesh.set_boundary_conditions(dev_phi_next);
}

void FieldSolver::compute_phi_next_at_inner_points()
{
	dim3 threads = mesh.GetThreads();
	dim3 blocks = mesh.GetBlocks(threads);
	cudaError_t cuda_status;

	ComputePhiNext << <blocks, threads >> > (mesh.dev_potential, mesh.dev_charge_density, dev_phi_next);
	cuda_status = cudaDeviceSynchronize();
}

void FieldSolver::set_phi_next_at_inner_regions(Inner_regions_manager &inner_regions)
{
	//for (auto &reg : inner_regions.regions) {
	//	for (auto &node : reg.inner_nodes) {
	//		// todo: mark nodes at edge during construction
	//		// if (!node.at_domain_edge( nx, ny, nz )) {
	//		phi_next[node.x][node.y][node.z] = reg.potential;
	//		// }
	//	}
	//}
}


bool FieldSolver::iterative_Jacobi_solutions_converged()
{
	//// todo: bind tol to config parameters
	cudaError_t status;
	dim3 threads = mesh.GetThreads();
	dim3 blocks = mesh.GetBlocks(threads);
	AssertConvergence << <blocks, threads >> > (mesh.dev_potential,dev_phi_next);
	status = cudaDeviceSynchronize();
	if (status == cudaErrorAssert) {
		return false;
	}
	if (status == cudaSuccess) {
		return true;
	}

	std::cout << "Cuda error: " << cudaGetErrorString(status) << std::endl;
	return false;
}


void FieldSolver::set_phi_next_as_phi_current()
{
	dim3 threads = mesh.GetThreads();
	dim3 blocks = mesh.GetBlocks(threads);
	cudaError_t cuda_status;
	SetPhiNextAsCurrent << <blocks, threads >> > (mesh.dev_potential, dev_phi_next);
	cuda_status = cudaDeviceSynchronize();
}


void FieldSolver::eval_fields_from_potential()
{
	dim3 threads = mesh.GetThreads();
	dim3 blocks = mesh.GetBlocks(threads);
	cudaError_t cuda_status;

	EvaluateFields << <blocks, threads >> > (mesh.dev_potential, mesh.dev_electric_field);

	cuda_status = cudaDeviceSynchronize();
	return;
}




FieldSolver::~FieldSolver()
{
	// delete phi arrays?
	cudaFree((void*)dev_phi_next);
	cudaFree((void*)d_n_nodes);
	cudaFree((void*)d_cell_size);
}
