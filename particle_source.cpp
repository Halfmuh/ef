#include "particle_source.h"

void check_and_warn_if_not( const bool &should_be, const std::string &message );

Particle_source::Particle_source( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_correctness_of_related_config_fields( conf, src_conf );
    set_parameters_from_config( src_conf );
    generate_initial_particles();
}

void Particle_source::check_correctness_of_related_config_fields( 
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

void Particle_source::set_parameters_from_config( Source_config_part &src_conf )
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
    unsigned seed = 0;
    rnd_gen = std::default_random_engine( seed );
}

void Particle_source::generate_initial_particles()
{
    //particles.reserve( initial_number_of_particles );
    generate_num_of_particles( initial_number_of_particles );
}

void Particle_source::generate_each_step()
{
    //particles.reserve( particles.size() + particles_to_generate_each_step );
    generate_num_of_particles( particles_to_generate_each_step );
}
    
void Particle_source::generate_num_of_particles( int num_of_particles )
{
    Vec3d pos, mom;
    int id = 0;
                
    for ( int i = 0; i < num_of_particles; i++ ) {
	id = generate_particle_id( i );
	pos = uniform_position_in_cube( xleft, ytop, znear, xright, ybottom, zfar, rnd_gen );
	mom = maxwell_momentum_distr( mean_momentum, temperature, mass, rnd_gen );
	particles.emplace_back( id, charge, mass, pos, mom );
    }

}

int Particle_source::generate_particle_id( const int number )
{    
    // Preserve max id between calls to generator.
    static int last_id = 0;
    
    return last_id++;
}

Vec3d Particle_source::uniform_position_in_cube( 
    const double xleft,  const double ytop, const double znear,
    const double xright, const double ybottom, const double zfar,
    std::default_random_engine &rnd_gen )
{
    return vec3d_init( random_in_range( xleft, xright, rnd_gen ), 
		       random_in_range( ybottom, ytop, rnd_gen ),
		       random_in_range( znear, zfar, rnd_gen ) );
}

double Particle_source::random_in_range( 
    const double low, const double up, 
    std::default_random_engine &rnd_gen )
{
    std::uniform_real_distribution<double> uniform_distr( low, up );
    return uniform_distr( rnd_gen );
}

Vec3d Particle_source::maxwell_momentum_distr(
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

void Particle_source::update_particles_position( double dt )
{
    for ( auto &p : particles )
	p.update_position( dt );
}


void Particle_source::print_particles()
{
    std::cout << "Source name: " << name << std::endl;
    for ( auto& p : particles  ) {	
	p.print_short();
    }
    return;
}

void Particle_source::write_to_file_iostream( std::ofstream &output_file )
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

void Particle_source::write_to_file_hdf5( hid_t group_id )
{
    std::cout << "Source name = " << name << ", "
	      << "number of particles = " << particles.size() 
	      << std::endl;
    std::string table_of_particles_name = name;

    write_hdf5_particles( group_id, table_of_particles_name );
    write_hdf5_source_parameters( group_id, table_of_particles_name );
    
    return;
}

void Particle_source::write_hdf5_particles( hid_t group_id, std::string table_of_particles_name )
{
    herr_t status;
    int n_of_particles = particles.size();
    int nfields = 6; // id, charge, mass, position, momentum, mpi_proc_rank
    int nrecords = n_of_particles;

    // todo: dst_buf should be removed.
    // currently it is used to avoid any problems of
    // working with Particles class, which is a C++ class
    // and not a plain C datastructure
    HDF5_buffer_for_Particle *dst_buf = new HDF5_buffer_for_Particle[n_of_particles];

    //size_t dst_size =  sizeof( Particle );
    size_t dst_size = sizeof( HDF5_buffer_for_Particle );

    size_t dst_offset[nfields];
    // dst_offset[0] = HOFFSET( Particle, id );
    // dst_offset[1] = HOFFSET( Particle, charge );
    // dst_offset[2] = HOFFSET( Particle, mass );
    // dst_offset[3] = HOFFSET( Particle, position );
    // dst_offset[4] = HOFFSET( Particle, momentum );
    dst_offset[0] = HOFFSET( HDF5_buffer_for_Particle, id );
    dst_offset[1] = HOFFSET( HDF5_buffer_for_Particle, charge );
    dst_offset[2] = HOFFSET( HDF5_buffer_for_Particle, mass );
    dst_offset[3] = HOFFSET( HDF5_buffer_for_Particle, position );
    dst_offset[4] = HOFFSET( HDF5_buffer_for_Particle, momentum );
    dst_offset[5] = HOFFSET( HDF5_buffer_for_Particle, mpi_proc_rank );

    const char *field_names[nfields];
    field_names[0] = "id";
    field_names[1] = "charge";
    field_names[2] = "mass";
    field_names[3] = "position";
    field_names[4] = "momentum";
    field_names[5] = "mpi_proc";

    hid_t vec3d_compound_type_for_mem;
    vec3d_compound_type_for_mem = vec3d_hdf5_compound_type_for_memory();

    hid_t field_type[nfields];
    field_type[0] = H5T_NATIVE_INT;
    field_type[1] = H5T_NATIVE_DOUBLE;
    field_type[2] = H5T_NATIVE_DOUBLE;
    field_type[3] = vec3d_compound_type_for_mem;
    field_type[4] = vec3d_compound_type_for_mem;
    field_type[5] = H5T_NATIVE_INT;

    // todo: will become unnecessary when dst_buf is removed.
    int mpi_process_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );    
    for( unsigned int i = 0; i < particles.size(); i++ ){
	dst_buf[i].id = particles[i].id;
	dst_buf[i].charge = particles[i].charge;
	dst_buf[i].mass = particles[i].mass;
	dst_buf[i].position = particles[i].position;
	dst_buf[i].momentum = particles[i].momentum;
	dst_buf[i].mpi_proc_rank = mpi_process_rank;
    }	
    
    hsize_t    chunk_size = 10;
    int        *fill_data = NULL;
    int        compress  = 0;
    
    H5TBmake_table( table_of_particles_name.c_str(), group_id, table_of_particles_name.c_str(),
		    nfields, nrecords,
		    dst_size, field_names, dst_offset, field_type,
		    chunk_size, fill_data, compress, dst_buf );

    status = H5Tclose( vec3d_compound_type_for_mem );
    delete[] dst_buf;
}

void Particle_source::write_hdf5_source_parameters( hid_t group_id,
						    std::string table_of_particles_name )
{
    int single_element = 1;
    double mean_mom_x = vec3d_x( mean_momentum );
    double mean_mom_y = vec3d_y( mean_momentum );
    double mean_mom_z = vec3d_z( mean_momentum );

    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(), "xleft", &xleft, single_element );
    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(), "xright", &xright, single_element );
    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(), "ytop", &ytop, single_element );
    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(), "ybottom", &ybottom, single_element );
    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(), "zfar", &zfar, single_element );
    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(), "znear", &znear, single_element );
    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(), "temperature", &temperature, single_element );
    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(),
			      "mean_momentum_x", &mean_mom_x, single_element );
    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(),
			      "mean_momentum_y", &mean_mom_y, single_element );
    H5LTset_attribute_double( group_id, table_of_particles_name.c_str(),
			      "mean_momentum_z", &mean_mom_z, single_element );    
}


void Particle_source::particle_source_initial_number_of_particles_gt_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_initial_number_of_particles > 0,
	"particle_source_initial_number_of_particles <= 0" );
}

void Particle_source::particle_source_particles_to_generate_each_step_ge_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_particles_to_generate_each_step >= 0,
	"particle_source_particles_to_generate_each_step < 0" );
}

void Particle_source::particle_source_x_left_ge_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_x_left >= 0,
	"particle_source_x_left < 0" );
}

void Particle_source::particle_source_x_left_le_particle_source_x_right( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_x_left <= src_conf.particle_source_x_right,
	"particle_source_x_left > particle_source_x_right" );
}

void Particle_source::particle_source_x_right_le_grid_x_size( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_x_right <= conf.mesh_config_part.grid_x_size,
	"particle_source_x_right > grid_x_size" );
}

void Particle_source::particle_source_y_bottom_ge_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_y_bottom >= 0,
	"particle_source_y_bottom < 0" );
}

void Particle_source::particle_source_y_bottom_le_particle_source_y_top( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_y_bottom <= src_conf.particle_source_y_top,
	"particle_source_y_bottom > particle_source_y_top" );
}

void Particle_source::particle_source_y_top_le_grid_y_size( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_y_top <= conf.mesh_config_part.grid_y_size,
	"particle_source_y_top > grid_y_size" );
}

void Particle_source::particle_source_z_near_ge_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_z_near >= 0,
	"particle_source_z_near < 0" );
}

void Particle_source::particle_source_z_near_le_particle_source_z_far( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_z_near <= src_conf.particle_source_z_far,
	"particle_source_z_near > particle_source_z_far" );
}

void Particle_source::particle_source_z_far_le_grid_z_size( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_z_far <= conf.mesh_config_part.grid_z_size,
	"particle_source_z_far > grid_z_size" );
}

void Particle_source::particle_source_temperature_gt_zero( 
    Config &conf, 
    Source_config_part &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_temperature >= 0,
	"particle_source_temperature < 0" );
}

void Particle_source::particle_source_mass_gt_zero( 
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


Particle_sources_manager::Particle_sources_manager( Config &conf )
{
    for( auto &src_conf : conf.sources_config_part ) {
	sources.emplace_back( conf, src_conf );
    }
}
