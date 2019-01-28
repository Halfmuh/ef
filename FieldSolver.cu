#include "FieldSolver.cuh"



#define FULL_MASK 0xffffffff
//mask for __all_sync used in convergence method

__constant__ double3 d_cell_size[1];
__constant__ int3 d_n_nodes[1];

__constant__ double dev_dxdxdydy[1];
__constant__ double dev_dxdxdzdz[1];
__constant__ double dev_dydydzdz[1];
__constant__ double dev_dxdxdydydzdz[1];

__device__ int GetIdx() {
	//int xStepthread = 1;
	int mesh_x = threadIdx.x + blockIdx.x * blockDim.x;
	int mesh_y = threadIdx.y + blockIdx.y * blockDim.y;
	int mesh_z = threadIdx.z + blockIdx.z * blockDim.z;
	return mesh_x +
			mesh_y * d_n_nodes[0].x +
			mesh_z * d_n_nodes[0].x * d_n_nodes[0].y;
}
__device__ int GetIdx(int x, int y, int z) {
	return 
		x +
		y * d_n_nodes[0].x +
		z * d_n_nodes[0].x * d_n_nodes[0].y;

}
__device__ double GradientComponent(double phi1, double phi2, double cell_side_size) {
	return ((phi2 - phi1) / cell_side_size);
}

__global__ void SetPhiNextAsCurrent(double* d_phi_current, const double* d_phi_next) {
	int idx = GetIdx();
	d_phi_current[idx] = d_phi_next[idx];
}

__global__ void ComputePhiNext(const double* d_phi_current, const double* d_charge, double* d_phi_next) {
	int mesh_x = threadIdx.x + blockIdx.x * blockDim.x;
	int mesh_y = threadIdx.y + blockIdx.y * blockDim.y;
	int mesh_z = threadIdx.z + blockIdx.z * blockDim.z;

	int idx = GetIdx(mesh_x, mesh_y, mesh_z);

	int prev_x = max(mesh_x - 1, 0);
	int prev_y = max(mesh_y - 1, 0);
	int prev_z = max(mesh_z - 1, 0);

	int next_x = min(mesh_x + 1, d_n_nodes[0].x - 1);
	int next_y = min(mesh_y + 1, d_n_nodes[0].y - 1);
	int next_z = min(mesh_z + 1, d_n_nodes[0].z - 1);
	
	int prev_neighbour_idx;
	int next_neighbour_idx;

	double denom = 2.0 * (dev_dxdxdydy[0] + dev_dxdxdzdz[0] + dev_dydydzdz[0]);

	prev_neighbour_idx = GetIdx(prev_x, mesh_y, mesh_z);
	next_neighbour_idx = GetIdx(next_x, mesh_y, mesh_z);

	d_phi_next[idx] =
		(d_phi_current[next_neighbour_idx] + d_phi_current[prev_neighbour_idx]) * dev_dydydzdz[0];

	prev_neighbour_idx = GetIdx(mesh_x, prev_y, mesh_z);
	next_neighbour_idx = GetIdx(mesh_x, next_y, mesh_z);
	d_phi_next[idx] +=
		(d_phi_current[next_neighbour_idx] + d_phi_current[prev_neighbour_idx]) * dev_dxdxdzdz[0];

	prev_neighbour_idx = GetIdx(mesh_x, mesh_y, prev_z);
	next_neighbour_idx = GetIdx(mesh_x, mesh_y, next_z);
	d_phi_next[idx] +=
		(d_phi_current[next_neighbour_idx] + d_phi_current[prev_neighbour_idx]) * dev_dxdxdydy[0];

	d_phi_next[idx] += 4.0 * CUDART_PI * d_charge[idx] * dev_dxdxdydydzdz[0];
	d_phi_next[idx] /= denom;

}

__global__ void EvaluateFields(const double* dev_potential, double3* dev_el_field) {
	int idx = GetIdx();

	double3 e = make_double3(0, 0, 0);
	//assuming true = 1, false = 0 
	//this method is hard to read due avoidance of if-else constructions on device code
	bool is_on_up_border;
	bool is_on_low_border;
	bool is_inside_borders;
	int offset;

	offset = 1;
	is_on_up_border = (threadIdx.x == 0) && (blockIdx.x == 0);
	is_on_low_border = (threadIdx.x == (blockDim.x - 1)) && (blockIdx.x == (gridDim.x - 1));
	is_inside_borders = !(is_on_low_border || is_on_up_border);

	e.x = -(1.0 / (1.0 + is_inside_borders)) * GradientComponent(
		dev_potential[idx + (offset*is_on_up_border) - (offset*is_inside_borders)],
		dev_potential[idx - (offset*is_on_low_border) + (offset*is_inside_borders)],
		d_cell_size[0].x);

	offset = d_n_nodes[0].x;
	is_on_up_border = (threadIdx.y == 0) && (blockIdx.y == 0);
	is_on_low_border = (threadIdx.y == (blockDim.y - 1)) && (blockIdx.y == (gridDim.y - 1));
	is_inside_borders = !(is_on_low_border || is_on_up_border);

	e.y = -(1.0 / (1.0 + is_inside_borders)) * GradientComponent(
		dev_potential[idx + (offset*is_on_up_border) - (offset*is_inside_borders)],
		dev_potential[idx - (offset*is_on_low_border) + (offset*is_inside_borders)],
		d_cell_size[0].y);

	offset = d_n_nodes[0].y*d_n_nodes[0].x;
	is_on_up_border = (threadIdx.z == 0) && (blockIdx.z == 0);
	is_on_low_border = (threadIdx.z == (blockDim.z - 1)) && (blockIdx.z == (gridDim.z - 1));
	is_inside_borders = !(is_on_low_border || is_on_up_border);

	e.z = -(1.0 / (1.0 + is_inside_borders)) * GradientComponent(
		dev_potential[idx + (offset * is_on_up_border) - (offset * is_inside_borders)],
		dev_potential[idx - (offset * is_on_low_border) + (offset * is_inside_borders)],
		d_cell_size[0].z);

	dev_el_field[idx] = e;

}

//__global__ void AssertConvergence(const double* d_phi_current, const double* d_phi_next) {
//	double rel_diff;
//	double abs_diff;
//	double abs_tolerance = 1.0e-5;
//	double rel_tolerance = 1.0e-12;
//	int idx = GetIdx();
//	abs_diff = fabs(d_phi_next[idx] - d_phi_current[idx]);
//	rel_diff = abs_diff / fabs(d_phi_current[idx]);
//	bool converged = ((abs_diff <= abs_tolerance) || (rel_diff <= rel_tolerance));
//
//	assert(converged==true);
//}

template<int nwarps>
__global__ void Convergence(const double* d_phi_current, const double* d_phi_next, unsigned int *d_convergence)
{
	__shared__ int w_convegence[nwarps];
	unsigned int laneid = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) % warpSize;
	unsigned int warpid = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) / warpSize;

	double rel_diff;
	double abs_diff;
	double abs_tolerance = 1.0e-5;
	double rel_tolerance = 1.0e-12;

	int idx = GetIdx();

	abs_diff = fabs(d_phi_next[idx] - d_phi_current[idx]);
	rel_diff = abs_diff / fabs(d_phi_current[idx]);

	unsigned int converged = ((abs_diff <= abs_tolerance) || (rel_diff <= rel_tolerance));

	converged = __all_sync(FULL_MASK, converged == 1 );

	if (laneid == 0) {
		w_convegence[warpid] = converged;
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		int b_convergence = 0;
#pragma unroll
		for (int i = 0; i<nwarps; i++) {
			b_convergence &= w_convegence[i];
		}
		if (b_convergence == 0 ) {
			atomicAdd(d_convergence, 1);
		}
	}
}

FieldSolver::FieldSolver(SpatialMeshCu &mesh, Inner_regions_manager &inner_regions) : mesh(mesh)
{
	allocate_next_phi();
	//std::cout << "solver memory allocation ";
	copy_constants_to_device();
	//std::cout << " solver copy constants ";
}

void FieldSolver::allocate_next_phi()
{
	size_t dim = mesh.n_nodes.x * mesh.n_nodes.y * mesh.n_nodes.z;
	cudaError_t cuda_status;

	cuda_status = cudaMalloc<double>(&dev_phi_next, dim * sizeof(double) );

}

void FieldSolver::copy_constants_to_device() {
	cudaError_t cuda_status;
	std::string cudaErrorMessage = "const copy to device";
	cuda_status = cudaMemcpyToSymbol(d_n_nodes, (const void*)&mesh.n_nodes, sizeof(int3));
	cuda_status = cudaMemcpyToSymbol(d_cell_size, (const void*)&mesh.cell_size, sizeof(double3));

	double dxdxdydy = mesh.cell_size.x * mesh.cell_size.x *
		mesh.cell_size.y * mesh.cell_size.y;
	cuda_status = cudaMemcpyToSymbol(dev_dxdxdydy, (const void*)&dxdxdydy, sizeof(double));
	cuda_status_check(cuda_status, cudaErrorMessage);

	double dxdxdzdz = mesh.cell_size.x * mesh.cell_size.x *
		mesh.cell_size.z * mesh.cell_size.z;
	cuda_status = cudaMemcpyToSymbol(dev_dxdxdzdz, (const void*)&dxdxdzdz, sizeof(double));
	cuda_status_check(cuda_status, cudaErrorMessage);

	double dydydzdz = mesh.cell_size.y * mesh.cell_size.y *
		mesh.cell_size.z * mesh.cell_size.z;
	cuda_status = cudaMemcpyToSymbol(dev_dydydzdz, (const void*)&dydydzdz, sizeof(double));
	cuda_status_check(cuda_status, cudaErrorMessage);

	double dxdxdydydzdz = mesh.cell_size.x * mesh.cell_size.x *
		mesh.cell_size.y * mesh.cell_size.y *
		mesh.cell_size.z * mesh.cell_size.z;
	cuda_status = cudaMemcpyToSymbol(dev_dxdxdydydzdz, (const void*)&dxdxdydydzdz, sizeof(double));
	cuda_status_check(cuda_status, cudaErrorMessage);

}

void FieldSolver::eval_potential(Inner_regions_manager &inner_regions)
{
	solve_poisson_eqn_Jacobi(inner_regions);
}

void FieldSolver::solve_poisson_eqn_Jacobi(Inner_regions_manager &inner_regions)
{
	max_Jacobi_iterations = 1;
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
	compute_phi_next_at_inner_points();
	set_phi_next_at_boundaries();
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
	std::string cudaErrorMessage = "compute fi";

	ComputePhiNext<<<blocks, threads>>>(mesh.dev_potential, mesh.dev_charge_density, dev_phi_next);
	cuda_status = cudaDeviceSynchronize();
	cuda_status_check(cuda_status, cudaErrorMessage);
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

	unsigned int *convergence, *d_convergence;//host,device  flags
	status = cudaHostAlloc((void **)&convergence, sizeof(unsigned int), cudaHostAllocMapped);
	status = cudaHostGetDevicePointer((void **)&d_convergence, convergence, 0);

	const int nwarps = 2;
	std::string cudaErrorMessage = "convergence";

	Convergence<nwarps><<<blocks, threads>>>(mesh.dev_potential, dev_phi_next, d_convergence);
	status = cudaDeviceSynchronize();
	cuda_status_check(status, cudaErrorMessage);
	//if (status == cudaErrorAssert) {
	//	return false;
	//}
	//if (status == cudaSuccess) {
	//	return true;
	//}

	//std::cout << "Cuda error: " << cudaGetErrorString(status) << std::endl;
	return *convergence == 0 ;
}


void FieldSolver::set_phi_next_as_phi_current()
{
	dim3 threads = mesh.GetThreads();
	dim3 blocks = mesh.GetBlocks(threads);
	cudaError_t cuda_status;
	SetPhiNextAsCurrent<<<blocks, threads>>>(mesh.dev_potential, dev_phi_next);
	cuda_status = cudaDeviceSynchronize();
}


void FieldSolver::eval_fields_from_potential()
{
	dim3 threads = mesh.GetThreads();
	dim3 blocks = mesh.GetBlocks(threads);
	cudaError_t cuda_status;

	EvaluateFields<<<blocks, threads>>>(mesh.dev_potential, mesh.dev_electric_field);

	cuda_status = cudaDeviceSynchronize();
	return;
}

void FieldSolver::cuda_status_check(cudaError_t status, std::string &sender)
{
	if (status > 0) {
		std::cout << "Cuda error at" << sender << ": " << cudaGetErrorString(status) << std::endl;
		exit(EXIT_FAILURE);
	}
}



FieldSolver::~FieldSolver()
{
	// delete phi arrays?
	cudaFree((void*)dev_phi_next);
	cudaFree((void*)d_n_nodes);
	cudaFree((void*)d_cell_size);
}
