//
// Created by paul on 18/05/2020.
//

#include "daemon.h"

#include <boost/program_options.hpp>
#include <iostream>

namespace boost_po = boost::program_options;

int main(int argc, char* argv[]) {
    boost_po::options_description desc("Options");
    desc.add_options()
            ("stay-in-foreground",
            "Do not turn this process in a background process and do not redirect stdout and stderr to /dev/null");
    boost_po::variables_map vm;
    boost_po::store(boost_po::parse_command_line(argc, argv, desc), vm);
    boost_po::notify(vm);

    if (vm.count("stay-in-foreground") == 0) daemon(0, 0);
    plasticity::Daemon plasticd;
    plasticd.start();
    return EXIT_SUCCESS;
}
