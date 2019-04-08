#!/bin/bash

# Copyright (c) 1999,2000,2002-2007
# Utrecht University (The Netherlands),
# ETH Zurich (Switzerland),
# INRIA Sophia-Antipolis (France),
# Max-Planck-Institute Saarbruecken (Germany),
# and Tel-Aviv University (Israel).  All rights reserved.
#
# This file is part of CGAL (www.cgal.org); you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation; version 3 of the License,
# or (at your option) any later version.
#
# Licensees holding a valid commercial license may use this file in
# accordance with the commercial license agreement provided with the software.
#
# This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
# WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#
# $URL$
# $Id$
# SPDX-License-Identifier: LGPL-3.0+
#
# Author(s)     : various

# This script creates a CGAL cmake script with entries for files with a common
# C++ file extension (as mentioned in the g++ man page) in the current directory.
#
# Usage: cgal_create_cmake_script TYPE
#
#  echo "  TYPE can be any of "demo", "example" or "test".. any other value is ignored"

#VERSION=2.0

create_cmake_script()
{
  # print makefile header
  cat <<EOF
# Created by the script cgal_create_cmake_script
# This is the CMake script for compiling a CGAL application.


project( ${PROJECT}_${TYPE} )
EOF
  cat <<'EOF'

cmake_minimum_required(VERSION 2.8.10)

set(CMAKE_CXX_STANDARD 14)

EOF
  
    cat <<'EOF'
find_package(CGAL QUIET COMPONENTS Core )

if ( CGAL_FOUND )

EOF
    if [ -d "${SOURCE_DIR}" ] ; then
      echo "  set(CGAL_CURRENT_SOURCE_DIR \"${SOURCE_DIR}\")"
      echo
    fi
    if [ -d "${SOURCE_DIR}../include" ] ; then
      echo "  include_directories (BEFORE \"${SOURCE_DIR}../include\")"
      echo
    fi
    if [ -d "${SOURCE_DIR}include" ] ; then
      echo "    include_directories (BEFORE \"${SOURCE_DIR}include\")"
      echo
    fi
    
    for file in `ls "$SOURCE_DIR"*.cc "$SOURCE_DIR"*.cp "$SOURCE_DIR"*.cxx "$SOURCE_DIR"*.cpp "$SOURCE_DIR"*.CPP "$SOURCE_DIR"*.c++ "$SOURCE_DIR"*.C 2>/dev/null | sort` ; do
      # Create an executable for each cpp that  contains a function "main()"
      BASE=`basename $file .cc`
      BASE=`basename $BASE .cp`
      BASE=`basename $BASE .cxx`
      BASE=`basename $BASE .cpp`
      BASE=`basename $BASE .CPP`
      BASE=`basename $BASE .c++`
      BASE=`basename $BASE .C`
      egrep '\bmain[ \t]*\(' $file >/dev/null 2>&1
      if [ $? -eq 0 ]; then
        echo "  create_single_source_cgal_program( \"$file\" )"
        echo "Adding a target ${BASE}..." >&3
      fi
    done
    
    cat <<'EOF'

else()
  
    message(STATUS "This program requires the CGAL library, and will not be compiled.")
  
endif()
EOF
  
  echo

}


usage()
{
  echo "Usage: cgal_create_cmake_script [--source_dir <source directory>]"
  echo
  echo "  Create a CMakeLists.txt file in the current working directory."
  echo
  echo "  TYPE must be any of example or test. The default is example."
  echo
  echo "  If the option --source_dir is specified with a directory, the "
  echo "  CMakeLists.txt uses source files from that directory, otherwise "
  echo "  the source directory is supposed to be the current directory."
}

SOURCE_DIR=

while [ $1 ]; do
    case "$1" in
        -h|-help|--h|--help)
            usage; exit
        ;;
        example) 
            if [ -z "$TYPE" ]; then TYPE=$1; shift; else usage; exit 1; fi
        ;;
        demo) 
            if [ -z "$TYPE" ]; then TYPE=$1; shift; else usage; exit 1; fi
        ;;
        test) 
            if [ -z "$TYPE" ]; then TYPE=$1; shift; else usage; exit 1; fi
        ;;
        --source_dir)
            if [ -d "$2" ]; then 
                SOURCE_DIR=$2; 
                shift;
                shift;
            else
                if [ -z "$2" ]; then
                    echo "Error: you must specify a directory after the --source_dir option!"
                    echo
                else
                    echo "Error: \"$2\" is not a directory!"
                    echo
                fi
                usage; exit 1; 
            fi
        ;;
        *) 
            echo "Unknown option: $1"
            usage; exit 1
        ;;
    esac
done

OUTPUTFILE=CMakeLists.txt
if [ -n "$SOURCE_DIR" ]; then
    PROJECT=`basename $SOURCE_DIR`
    SOURCE_DIR=$SOURCE_DIR/
else
    PROJECT=`basename $PWD`
fi

if [ -f ${OUTPUTFILE} ] ; then
  echo "moving $OUTPUTFILE to ${OUTPUTFILE}.bak ..."
  mv -f $OUTPUTFILE ${OUTPUTFILE}.bak
fi
create_cmake_script 3>&1 > $OUTPUTFILE
echo "created $OUTPUTFILE in $PWD ..."
