#include "particle_source.h"

void check_and_warn_if_not( const bool &should_be, const std::string &message );

Single_particle_source::Single_particle_source( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_correctness_of_related_config_fields( conf, src_conf );
    set_parameters_from_config( src_conf );
    generate_initial_particles();
}

void Single_particle_source::check_correctness_of_related_config_fields( 
    Config &conf, 
    Source_config_part &src_conf )
{
    particle_source_initial_number_of_particles_gt_zero( conf, src_conf );
    particle_source_particles_to_generate_each_step_ge_zero( conf, src_conf );
    particle_source_x_left_ge_zero( conf, src_conf );
    particle_source_x_left_le_particle_source_x_right( conf, src_conf );
    particle_source_x_right_le_grid_x_size( conf, src_conf );
    particle_source_y_bottom_ge_zero( conf, src_conf );
    particle_source_y_bottom_le_particle_source_y_top( conf, src_conf );
    particle_source_y_top_le_grid_y_size( conf, src_conf );
    particle_source_z_near_ge_zero( conf, src_conf );
    particle_source_z_near_le_particle_source_z_far( conf, src_conf );
    particle_source_z_far_le_grid_z_size( conf, src_conf );
    particle_source_temperature_gt_zero( conf, src_conf );
    particle_source_mass_gt_zero( conf, src_conf );
}

void Single_particle_source::set_parameters_from_config( Source_config_part &src_conf )
{
    name = src_conf.particle_source_name;
    initial_number_of_particles = src_conf.particle_source_initial_number_of_particles;
    particles_to_generate_each_step = 
	src_conf.particle_source_particles_to_generate_each_step;
    xleft = src_conf.particle_source_x_left;
    xright = src_conf.particle_source_x_right;
    ytop = src_conf.particle_source_y_top;
    ybottom = src_conf.particle_source_y_bottom;
    znear = src_conf.particle_source_z_near;
    zfar = src_conf.particle_source_z_far;
    mean_momentum = vec3d_init( src_conf.particle_source_mean_momentum_x, 
				src_conf.particle_source_mean_momentum_y,
				src_conf.particle_source_mean_momentum_z );
    temperature = src_conf.particle_source_temperature;
    charge = src_conf.particle_source_charge;
    mass = src_conf.particle_source_mass;
    // Random number generator
    // Simple approach: use different seed for each proccess.
    // Other way would be to synchronize the state of the rnd_gen
    //    between each processes after each call to it.    
    int mpi_process_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );    
    unsigned seed = 0 + 1000000*mpi_process_rank;
    rnd_gen = std::default_random_engine( seed );
    // Initial id
    max_id = 0;
}

void Single_particle_source::generate_initial_particles()
{
    //particles.reserve( initial_number_of_particles );
    generate_num_of_particles( initial_number_of_particles );
}

void Single_particle_source::generate_each_step()
{
    //particles.reserve( particles.size() + particles_to_generate_each_step );
    generate_num_of_particles( particles_to_generate_each_step );
}
    
void Single_particle_source::generate_num_of_particles( int num_of_particles )
{
    Vec3d pos, mom;
    std::vector<int> vec_of_ids;
    int num_of_particles_for_this_proc;

    int mpi_n_of_proc, mpi_process_rank;
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_n_of_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );    

    num_of_particles_for_each_process( &num_of_particles_for_this_proc,
				       num_of_particles );
    populate_vec_of_ids( vec_of_ids, num_of_particles_for_this_proc ); 
    for ( int i = 0; i < num_of_particles_for_this_proc; i++ ) {
	pos = uniform_position_in_cube( xleft, ytop, znear,
					xright, ybottom, zfar,
					rnd_gen );
	mom = maxwell_momentum_distr( mean_momentum, temperature, mass, rnd_gen );
	particles.emplace_back( vec_of_ids[i], charge, mass, pos, mom );
    }
}

void Single_particle_source::num_of_particles_for_each_process(
    int *num_of_particles_for_this_proc,
    int num_of_particles )
{
    int rest;
    int mpi_n_of_proc, mpi_process_rank;
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_n_of_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );    
    
    *num_of_particles_for_this_proc = num_of_particles / mpi_n_of_proc;
    rest = num_of_particles % mpi_n_of_proc;
    if( mpi_process_rank < rest ){
	(*num_of_particles_for_this_proc)++;
	// Processes with lesser ranks will accumulate
	// more particles.
	// This seems unessential.
    }    
}

void Single_particle_source::populate_vec_of_ids(
    std::vector<int> &vec_of_ids, int num_of_particles_for_this_proc )
{
    int mpi_n_of_proc, mpi_process_rank;
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_n_of_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );    

    vec_of_ids.reserve( num_of_particles_for_this_proc );
    
    for( int proc = 0; proc < mpi_n_of_proc; proc++ ){
	if( mpi_process_rank == proc ){
	    for( int i = 0; i < num_of_particles_for_this_proc; i++ ){
		vec_of_ids.push_back( max_id++ );
	    }	    
	}
	MPI_Bcast( &max_id, 1, MPI_INT, proc, MPI_COMM_WORLD );
    }
}

// int Single_particle_source::generate_particle_id( const int number, const int proc )
// {
//     max_id++;
//     MPI_Bcast( &max_id, 1, MPI_UNSIGNED, proc, MPI_COMM_WORLD );
//     return max_id;     
// }

Vec3d Single_particle_source::uniform_position_in_cube( 
    const double xleft,  const double ytop, const double znear,
    const double xright, const double ybottom, const double zfar,
    std::default_random_engine &rnd_gen )
{
    return vec3d_init( random_in_range( xleft, xright, rnd_gen ), 
		       random_in_range( ybottom, ytop, rnd_gen ),
		       random_in_range( znear, zfar, rnd_gen ) );
}

double Single_particle_source::random_in_range( 
    const double low, const double up, 
    std::default_random_engine &rnd_gen )
{
    std::uniform_real_distribution<double> uniform_distr( low, up );
    return uniform_distr( rnd_gen );
}

Vec3d Single_particle_source::maxwell_momentum_distr(
    const Vec3d mean_momentum, const double temperature, const double mass, 
    std::default_random_engine &rnd_gen )
{    
    double maxwell_gauss_std_mean_x = vec3d_x( mean_momentum );
    double maxwell_gauss_std_mean_y = vec3d_y( mean_momentum );
    double maxwell_gauss_std_mean_z = vec3d_z( mean_momentum );
    double maxwell_gauss_std_dev = sqrt( mass * temperature );
    std::normal_distribution<double> 
	normal_distr_x( maxwell_gauss_std_mean_x, maxwell_gauss_std_dev );
    std::normal_distribution<double> 
	normal_distr_y( maxwell_gauss_std_mean_y, maxwell_gauss_std_dev );
    std::normal_distribution<double> 
	normal_distr_z( maxwell_gauss_std_mean_z, maxwell_gauss_std_dev );

    Vec3d mom;
    mom = vec3d_init( normal_distr_x( rnd_gen ),
		      normal_distr_y( rnd_gen ),
		      normal_distr_z( rnd_gen ) );		     
    mom = vec3d_times_scalar( mom, 1.0 ); // recheck
    return mom;
}

void Single_particle_source::update_particles_position( double dt )
{
    for ( auto &p : particles )
	p.update_position( dt );
}


void Single_particle_source::print_particles()
{
    std::cout << "Source name: " << name << std::endl;
    for ( auto& p : particles  ) {	
	p.print_short();
    }
    return;
}

void Single_particle_source::write_to_file( std::ofstream &output_file )
{
    std::cout << "Source name = " << name << ", "
	      << "number of particles = " << particles.size() 
	      << std::endl;
    output_file << "Source name = " << name << std::endl;
    output_file << "Total number of particles = " << particles.size() << std::endl;
    output_file << "id, charge, mass, position(x,y,z), momentum(px,py,pz)" << std::endl;
    output_file.fill(' ');
    output_file.setf( std::ios::scientific );
    output_file.precision( 3 );    
    output_file.setf( std::ios::right );
    for ( auto &p : particles ) {	
	output_file << std::setw(10) << std::left << p.id
		    << std::setw(12) << p.charge
		    << std::setw(12) << p.mass
		    << std::setw(12) << vec3d_x( p.position )
		    << std::setw(12) << vec3d_y( p.position )
		    << std::setw(12) << vec3d_z( p.position )
		    << std::setw(12) << vec3d_x( p.momentum )
		    << std::setw(12) << vec3d_y( p.momentum )
		    << std::setw(12) << vec3d_z( p.momentum )
		    << std::endl;
    }
    return;
}

void Single_particle_source::write_to_file_particles_only( std::ofstream &output_file )
{
    output_file.fill(' ');
    output_file.setf( std::ios::scientific );
    output_file.precision( 3 );    
    output_file.setf( std::ios::right );
    for ( auto &p : particles ) {	
	output_file << std::setw(10) << std::left << p.id
		    << std::setw(12) << p.charge
		    << std::setw(12) << p.mass
		    << std::setw(12) << vec3d_x( p.position )
		    << std::setw(12) << vec3d_y( p.position )
		    << std::setw(12) << vec3d_z( p.position )
		    << std::setw(12) << vec3d_x( p.momentum )
		    << std::setw(12) << vec3d_y( p.momentum )
		    << std::setw(12) << vec3d_z( p.momentum )
		    << std::endl;
    }
    return;
}


void Single_particle_source::particle_source_initial_number_of_particles_gt_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_initial_number_of_particles > 0,
	"particle_source_initial_number_of_particles <= 0" );
}

void Single_particle_source::particle_source_particles_to_generate_each_step_ge_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_particles_to_generate_each_step >= 0,
	"particle_source_particles_to_generate_each_step < 0" );
}

void Single_particle_source::particle_source_x_left_ge_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_x_left >= 0,
	"particle_source_x_left < 0" );
}

void Single_particle_source::particle_source_x_left_le_particle_source_x_right( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_x_left <= src_conf.particle_source_x_right,
	"particle_source_x_left > particle_source_x_right" );
}

void Single_particle_source::particle_source_x_right_le_grid_x_size( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_x_right <= conf.mesh_config_part.grid_x_size,
	"particle_source_x_right > grid_x_size" );
}

void Single_particle_source::particle_source_y_bottom_ge_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_y_bottom >= 0,
	"particle_source_y_bottom < 0" );
}

void Single_particle_source::particle_source_y_bottom_le_particle_source_y_top( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_y_bottom <= src_conf.particle_source_y_top,
	"particle_source_y_bottom > particle_source_y_top" );
}

void Single_particle_source::particle_source_y_top_le_grid_y_size( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_y_top <= conf.mesh_config_part.grid_y_size,
	"particle_source_y_top > grid_y_size" );
}

void Single_particle_source::particle_source_z_near_ge_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_z_near >= 0,
	"particle_source_z_near < 0" );
}

void Single_particle_source::particle_source_z_near_le_particle_source_z_far( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_z_near <= src_conf.particle_source_z_far,
	"particle_source_z_near > particle_source_z_far" );
}

void Single_particle_source::particle_source_z_far_le_grid_z_size( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_z_far <= conf.mesh_config_part.grid_z_size,
	"particle_source_z_far > grid_z_size" );
}

void Single_particle_source::particle_source_temperature_gt_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_temperature >= 0,
	"particle_source_temperature < 0" );
}

void Single_particle_source::particle_source_mass_gt_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_mass >= 0,
	"particle_source_mass < 0" );
}

void check_and_warn_if_not( const bool &should_be, const std::string &message )
{
    if( !should_be ){
	std::cout << "Warning: " + message << std::endl;
    }
    return;
}


Particle_sources::Particle_sources( Config &conf )
{
    for( auto &src_conf : conf.sources_config_part ) {
	sources.emplace_back( conf, src_conf );
    }
}
