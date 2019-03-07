#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/convex_hull_3.h>
#include <vector>
#include <fstream>
#include <string>
#include <CGAL/draw_surface_mesh.h> // used to draw a surface mesh (have to compile CGAL from source with Qt5)

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Polyhedron_3<K> Polyhedron_3;
typedef K::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Surface_mesh;

int main(int argc, char *argv[])
{
     if (argc < 2)
     {
          std::cout << "Usage: 3dhull input_xyz" << std::endl;
          return -1;
     }
     std::ifstream in(argv[1]);
     std::vector<Point_3> points;
     Point_3 p;
     while (in >> p)
     {
          points.push_back(p);
     }
     // define polyhedron to hold convex hull
     Polyhedron_3 poly;

     // compute convex hull of non-collinear points
     CGAL::convex_hull_3(points.begin(), points.end(), poly);
     std::cout << "The convex hull contains " << poly.size_of_vertices() << " vertices" << std::endl;
     Surface_mesh sm;

     CGAL::convex_hull_3(points.begin(), points.end(), sm);
     std::cout << "The convex hull contains " << num_vertices(sm) << " vertices" << std::endl;


     // Output off mesh
     std::string out_file(argv[1]);
     out_file = out_file.substr(0, out_file.length() - 4) + "_hull.off";
     std::ofstream writeout(out_file, std::ios::trunc);
     if (!CGAL::write_off(writeout, sm))
     // if (!CGAL::write_off(out_file))
     {
          std::cout << "ERROR: cannot write convex hull mesh into " << out_file << std::endl;
          return -1;
     }

     CGAL::draw(sm);

     return 0;
}