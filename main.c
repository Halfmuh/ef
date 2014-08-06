#include "config.h"
#include "domain.h"
#include "parse_args.h"

void pic_simulation( Config *conf );

int main(int argc, char *argv[])
{

    char *config_file = NULL;
    Config conf;

    // prepare everything
    //// Parse command line
    parse_args( argc, argv, &config_file );
    printf( "CONFIG = %s \n", config_file );
    //// Read config
    config_read( config_file, &conf );
    config_print( &conf );
    // run simulation
    pic_simulation( &conf );
    // finalize_whatever_left
    return 0;
}

void pic_simulation( Config *conf )
{
  Domain dom;

  domain_prepare( &dom, conf );
  domain_write( &dom, conf );
  domain_run_pic( &dom, conf );
  domain_free( &dom );

  return;
}
