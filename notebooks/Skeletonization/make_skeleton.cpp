
#define CGAL_EIGEN3_ENABLED
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <CGAL/boost/graph/split_graph_into_polylines.h>
#include <fstream>
#include <boost/foreach.hpp>
typedef CGAL::Simple_cartesian<double>                        Kernel2;
typedef Kernel2::Point_3                                       Point;
typedef CGAL::Polyhedron_3<Kernel2>                            Polyhedron;
typedef boost::graph_traits<Polyhedron>::vertex_descriptor    vertex_descriptor;
typedef CGAL::Mean_curvature_flow_skeletonization<Polyhedron> Skeletonization;
typedef Skeletonization::Skeleton                             Skeleton;
typedef Skeleton::vertex_descriptor                           Skeleton_vertex;
typedef Skeleton::edge_descriptor                             Skeleton_edge;

#if 1
#include <CGAL/IO/OFF_reader.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/mesh_segmentation.h>
#include <CGAL/property_map.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#endif

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;



//only needed for the display of the skeleton as maximal polylines
struct Display_polylines{
    const Skeleton& skeleton;
    std::ofstream& out;
    int polyline_size;
    std::stringstream sstr;
    Display_polylines(const Skeleton& skeleton, std::ofstream& out)
    : skeleton(skeleton), out(out)
    {}
    void start_new_polyline(){
        polyline_size=0;
        sstr.str("");
        sstr.clear();
    }
    void add_node(Skeleton_vertex v){
        ++polyline_size;
        sstr << " " << skeleton[v].point;
    }
    void end_polyline()
    {
        out << polyline_size << sstr.str() << "\n";
    }
};
// This example extracts a medially centered skeleton from a given mesh.
int make_skeleton(){
    //std::ifstream input((argc>1)?argv[1]:"data/elephant.off");

    //WORKED FOR ELEPHANT EXAMPLE BELOW
    /*
    std::string location = "/Users/brendancelii/Documents/C++_code/mesh_skeleton/";
    std::string filename = "elephant";
    */

    //std::string location = "/Users/brendancelii/Google Drive/Xaq Lab/Final_Blender/Blender_Annotations_Tool/Automated_Pipeline/temp/";
    //std::string filename = "neuron_648518346341366885"; //generted by mesh lab and doesn't work

    std::string location = "/Users/brendancelii/Documents/C++_code/mesh_skeleton/";
    std::string filename = "filled_cleaned_up"; //generted by mesh lab and doesn't work
    
    std::ifstream input(location + filename + ".off");

    /* Old way of importing mesh
    Polyhedron mesh;
    input >> mesh;
    if (!CGAL::is_triangle_mesh(mesh))
    {
        std::cout << "Input geometry is not triangulated." << std::endl;
        return EXIT_FAILURE;
    }
    */

    //New way of importing mesh
    Polyhedron mesh;
    if (!input)
    {
        std::cerr << "Cannot open file " << std::endl;
        return 2;
    }
    std::vector<K::Point_3> points;
    std::vector< std::vector<std::size_t> > polygons;
    if (!CGAL::read_OFF(input, points, polygons))
    {
        std::cerr << "Error parsing the OFF file " << std::endl;
        return 3;
    }
    CGAL::Polygon_mesh_processing::orient_polygon_soup(points, polygons);
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, polygons, mesh);
    if(!CGAL::is_closed(mesh)){
        std::cerr << "Not closed mesh"<<std::endl;
        return 4;
    }
    if (CGAL::is_closed(mesh) && (!CGAL::Polygon_mesh_processing::is_outward_oriented(mesh)))
        CGAL::Polygon_mesh_processing::reverse_face_orientations(mesh);

    //end of new way



    Skeleton skeleton;
    CGAL::extract_mean_curvature_flow_skeleton(mesh, skeleton);
    std::cout << "Number of vertices of the skeleton: " << boost::num_vertices(skeleton) << "\n";
    std::cout << "Number of edges of the skeleton: " << boost::num_edges(skeleton) << "\n";
    // Output all the edges of the skeleton.
    //std::ofstream output("skel-poly.cgal");

    std::string output_location = "/Users/brendancelii/Documents/C++_code/mesh_skeleton/";
    std::ofstream output(output_location + filename + "edges-.cgal");
    Display_polylines display(skeleton,output);
    CGAL::split_graph_into_polylines(skeleton, display);
    output.close();
    // Output skeleton points and the corresponding surface points
    //output.open("correspondance-poly.cgal");
    output.open(output_location + filename + "correspondance-poly-.cgal");
    BOOST_FOREACH(Skeleton_vertex v, vertices(skeleton))
    BOOST_FOREACH(vertex_descriptor vd, skeleton[v].vertices)
    output << "2 " << skeleton[v].point << " "
    << get(CGAL::vertex_point, mesh, vd)  << "\n";
    return EXIT_SUCCESS;
}
