#include <iostream>
#include <string>
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/draw_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <fstream>
#include <time.h>       /* time */


typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_2<int, K> Vb; // add some user-defined variable per vertex
typedef CGAL::Triangulation_face_base_with_info_2<int, K> Fb; // add some user-defined variable per face
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Delaunay_triangulation_2<K, Tds> Delaunay;
typedef Delaunay::Point Point;

int main()
{
	std::vector<std::pair<Point, int>> points;
	points.push_back( std::make_pair( Point(0.3, 0), 0 ) );
	points.push_back( std::make_pair( Point(0.111233131, 1), 1 ) );
	points.push_back( std::make_pair( Point(0, 2.44112112), 2 ) );
	points.push_back( std::make_pair( Point(1, 0), 3 ) );
	points.push_back( std::make_pair( Point(1, 1), 4 ) );
	points.push_back( std::make_pair( Point(1, 2), 5 ) );

	Delaunay dt;
	dt.insert( points.begin(), points.end() );
	std::cout << "#Vertices in Delaunay: " << dt.number_of_vertices() << std::endl;

	/// Traverse all finite vertices
	Delaunay::Finite_vertices_iterator vit;
	for (vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit)
	{
		if ( points[ vit->info() ].first != vit->point() )
		{	// info() holds the original point index
			std::cerr << "Error different info" << std::endl;
			return false;
		}
		std::cout << "Vertex " << vit->info() << ": " << vit->point() << std::endl;

		// Traverse all adjacent vertices for each vertex
		std::cout << "Neighbors: ";
		Delaunay::Vertex_handle vd = vit;
		// A vertex circulator will start at an arbitrary vertex adjacent to vd,
		// and visit circular sequences and finally return to the start vertex.
		Delaunay::Vertex_circulator vc = dt.incident_vertices(vd), vc_end(vc);
		while (vc != 0) // check if the vertex has adjacent neighbors
		{
			if (!dt.is_infinite(vc))
				std::cout << vc->info() << " ";
			if (++vc == vc_end)
				break;
		}
		std::cout << std::endl;
	}

	/// Traverse all faces
	Delaunay::Finite_faces_iterator fit;
	int fidx = 0;
	for (fit = dt.finite_faces_begin() ; fit != dt.finite_faces_end(); ++fit)
	{
		fit->info() = fidx++; // assign a face index for each face
	}
	fidx = 0;
	for (fit = dt.finite_faces_begin() ; fit != dt.finite_faces_end(); ++fit)
	{
		std::cout << "Face " << fit->info() << std::endl;

		Delaunay::Face_handle fd = fit;
		std::cout << "Face Vertices: ";
		for (int i = 0; i < 3; ++i)
		{
			std::cout << fd->vertex(i)->info() << " ";
		}

		std::cout << "Neighbors faces: ";
		for (int i = 0; i < 3; ++i)
		{
			Delaunay::Face_handle fd_nbr = fd->neighbor(i); // Get ith neighbor, where i (0,1,2) is the index of face vertex
			if (dt.is_infinite(fd_nbr)) // skip infinite face
				continue;
			std::cout << fd_nbr->info() << "(";
			std::cout << fd->index(fd_nbr) << ") "; // which neighbor of original face that this neighbor is
		}

		std::cout << "Opposite vertices: ";
		for (int i = 0; i < 3; ++i)
		{
			std::cout << dt.mirror_vertex(fd, i)->info() << " ";
		}
		std::cout << std::endl;
	}

	std::cout << "DONE!" << std::endl;

	// /// Try to change the coordinates of some points
	// std::cout << "Change vertex coordinates after triangulation" << std::endl;
	// srand (time(NULL));
	// for (vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit)
	// {
	// 	Point pt(vit->point().x() + double(rand() % 10) / 10, vit->point().y() + double(rand() % 10) / 10);
	// 	vit->set_point(pt);
	// 	std::cout << "Vertex " << vit->info() << ": " << vit->point() << std::endl;
	// } 

	// Draw the triangulation result (need to use CGAL with Qt5)
	CGAL::draw(dt);

	return 0;
}
