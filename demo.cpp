
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>

#include <boost/iterator/function_output_iterator.hpp>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Heat_method_3/Surface_mesh_geodesic_distances_3.h>
#include <CGAL/IO/write_ply_points.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Point_3.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/clip.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/fair.h>
#include <CGAL/Polygon_mesh_processing/locate.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/refine.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/smooth_shape.h>
#include <CGAL/Polygon_mesh_processing/surface_Delaunay_remeshing.h>
#include <CGAL/Qt/Basic_viewer_qt.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_shortest_path.h>
#include <CGAL/Vector_3.h>
#include <CGAL/draw_point_set_3.h>
#include <CGAL/draw_surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Mean_curvature_flow_skeletonization.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <CGAL/mesh_segmentation.h>
#include <CGAL/draw_polyhedron.h>
#include <CGAL/draw_point_set_3.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h> 
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//#pragma comment( linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"" ) //todo 调用exe时隐藏控制台窗口，编译时不能够显示打印消息

// CGAL 约定类型
typedef CGAL::Exact_predicates_inexact_constructions_kernel   K;
typedef CGAL::Surface_mesh<K::Point_3>                        Mesh;
namespace PMP = CGAL::Polygon_mesh_processing;
typedef boost::graph_traits<Mesh>::vertex_descriptor      vertex_descriptor;
typedef boost::graph_traits<Mesh>::halfedge_descriptor        halfedge_descriptor;
typedef boost::graph_traits<Mesh>::edge_descriptor            edge_descriptor;
typedef boost::graph_traits<Mesh>::face_descriptor           face_descriptor;
typedef boost::graph_traits<Mesh>::faces_size_type           faces_size_type;
typedef Mesh::Property_map<vertex_descriptor, double> Vertex_distance_map;
typedef CGAL::Surface_mesh_shortest_path_traits<K, Mesh> Traits;
typedef CGAL::Surface_mesh_shortest_path<Traits> Surface_mesh_shortest_path;
typedef CGAL::Search_traits_3<K> TreeTraits;
typedef CGAL::Orthogonal_k_neighbor_search<TreeTraits> Neighbor_search;
typedef Neighbor_search::Tree Tree;

typedef CGAL::Simple_cartesian<double>                               Kernel;
typedef Kernel::Point_3                                              Point;
typedef CGAL::Polyhedron_3<Kernel, CGAL::Polyhedron_items_with_id_3> Polyhedron;
typedef boost::graph_traits<Polyhedron>::vertex_descriptor           vertex_descriptorPolyh;
typedef boost::graph_traits<Polyhedron>::halfedge_descriptor         halfedge_descriptorPolyh;
typedef boost::graph_traits<Polyhedron>::face_descriptor             face_descriptorPolyh;
typedef CGAL::Mean_curvature_flow_skeletonization<Polyhedron>        Skeletonization;
typedef Skeletonization::Skeleton                                    Skeleton;
typedef Skeleton::vertex_descriptor                                  Skeleton_vertex;

typedef Mesh::Property_map<vertex_descriptor, Kernel::Vector_3> VNMap;
//typedef K::Vector_3                                               Vector;

// 自定义类型
typedef std::vector<vertex_descriptor> border_vertices_type;
typedef std::vector<edge_descriptor> border_edges_type;
typedef std::vector<K::Point_3> border_points_type;
typedef std::unordered_map<vertex_descriptor, std::list<vertex_descriptor>> isoline_topology_type;
typedef std::unordered_map<vertex_descriptor, std::list<face_descriptor>> isoline_location_type;
typedef Mesh::Property_map<vertex_descriptor, face_descriptor> shortest_path_map_face;   //点到最近的边界面
typedef std::vector<std::vector<std::pair<K::Point_3, K::Vector_3>>> isoline_points_type_part;
typedef std::vector<isoline_points_type_part> isoline_points_type_path;
typedef std::vector<std::pair<std::vector<std::pair<K::Point_3, K::Vector_3>>, float>> line_width;

bool is_local_test = false;

struct halfedge2edge
{
    halfedge2edge(const Mesh& m, std::vector<edge_descriptor>& edges)
        : m_mesh(m), m_edges(edges)
    {}
    void operator()(const halfedge_descriptor& h) const
    {
        m_edges.push_back(edge(h, m_mesh));
    }
    const Mesh& m_mesh;
    std::vector<edge_descriptor>& m_edges;
};

struct Display_polylines {
    const Skeleton& skeleton;
    std::ofstream& out;
    int polyline_size;
    std::stringstream sstr;
    Display_polylines(const Skeleton& skeleton, std::ofstream& out)
        : skeleton(skeleton), out(out)
    {}
    void start_new_polyline() {
        polyline_size = 0;
        sstr.str("");
        sstr.clear();
    }
    void add_node(Skeleton_vertex v) {
        ++polyline_size;
        sstr << " " << skeleton[v].point;
    }
    void end_polyline()
    {
        out << polyline_size << sstr.str() << "\n";
    }
};

template<class ValueType>
struct Face_with_id_pmap
    : public boost::put_get_helper<ValueType&,
    Face_with_id_pmap<ValueType> >
{
    typedef face_descriptor key_type;
    typedef ValueType value_type;
    typedef value_type& reference;
    typedef boost::lvalue_property_map_tag category;
    Face_with_id_pmap(
        std::vector<ValueType>& internal_vector
    ) : internal_vector(internal_vector) { }
    reference operator[](key_type key) const
    {
        return internal_vector[key->id()];
    }
private:
    std::vector<ValueType>& internal_vector;
};

struct Vector_pmap_wrapper
{
    std::vector<bool>& vect;
    Vector_pmap_wrapper(std::vector<bool>& v) : vect(v) {}
    friend bool get(const Vector_pmap_wrapper& m, face_descriptor f)
    {
        return m.vect[f];
    }
    friend void put(const Vector_pmap_wrapper& m, face_descriptor f, bool b)
    {
        m.vect[f] = b;
    }
};

struct PathPlannerConfig
{
    double gap_border = 0.04;                 //首笔到边界距离
    double gap_inner = 0.08;                  // 笔宽
    double epsilon_global = 0.08;              // 改这个参数，越小运行时间越长，路径质量越好,形成的网格越密集
    double epsilon_border = 0.08;             // 边界上撒点间距，越小距离计算越准，避免边界上的点扭曲
    int near_border_iters = 3;                //边界迭代次数
    double min_lineswidth = 0.1;              //最小线宽
    bool use_heat_method = false;
    double second_epsilon = 0.04;             //骨骼细化密度
    bool startpointinner = false;             //首笔点是否在内部
    bool datasample = true;                  //是否对数据进行采样
    bool skeletonization = false;             //是否骨架化
    bool ismuitlpenwidth = true;             //是否多笔宽

    float minlines = 0.04;//离散后最小线间距
    float maxprintwidth = 0.8;//打印笔最大宽度 0.8
    float minprintwidth = 0.08;//打印笔最小宽度 0.08
};
static bool firstCircleFalg = true;
// 定义点结构体，包含法向量  
struct PointVex {
    float x, y, z; // 坐标  
    float nx, ny, nz; // 法向量  

    PointVex(float x, float y, float z, float nx, float ny, float nz)
        : x(x), y(y), z(z), nx(nx), ny(ny), nz(nz) {}
};

// 定义体素结构体，包含法向量信息  
struct Voxel {
    std::vector<PointVex> points; // 存放在这个体素中的点  
};


// 哈希函数用于体素索引  
struct VoxelHash {
    size_t operator()(const std::tuple<int, int, int>& voxel) const {
        return std::hash<int>()(std::get<0>(voxel)) ^
            std::hash<int>()(std::get<1>(voxel)) ^
            std::hash<int>()(std::get<2>(voxel));
    }
};

// Property map associating a facet with an integer as id to an
// element in a vector stored internally
template<class ValueType>
struct Facet_with_id_pmap
    : public boost::put_get_helper<ValueType&,
    Facet_with_id_pmap<ValueType> >
{
    typedef face_descriptorPolyh key_type;
    typedef ValueType value_type;
    typedef value_type& reference;
    typedef boost::lvalue_property_map_tag category;
    Facet_with_id_pmap(
        std::vector<ValueType>& internal_vector
    ) : internal_vector(internal_vector) { }
    reference operator[](key_type key) const
    {
        return internal_vector[key->id()];
    }
private:
    std::vector<ValueType>& internal_vector;
};


/// <summary>
/// 构建当前轮廓线数据结构
/// </summary>
struct ContourLinePoint
{
    float forwarddis = -1; //顺时针外部间距
    float backwarddis = -1;//顺时针内部间距
    std::vector<K::Point_3> points;
    std::vector<K::Vector_3> pointnormals;
};



struct PathPlanner
{
    PathPlannerConfig config;
    PathPlanner(PathPlannerConfig config) : config(config) {
        allpath.reserve(1024);
    }
    ~PathPlanner() {}
    isoline_points_type_path allpath;
    std::vector<std::pair<std::vector<std::vector<K::Point_3>>,int>> allpathboudrytype;

    std::vector<int> merwgerpos;

    std::vector<line_width> linesclassify;
    std::vector<std::vector<std::pair<K::Point_3, K::Vector_3>>> evecircledata; //一笔画路径，如果有较远距离就代表跳线


    /*std::vector<K::Point_3>*/std::vector<std::pair<K::Point_3, K::Vector_3>> plan(Mesh& mesh)
    {
        std::vector<K::Point_3> path;
        std::vector<std::pair<K::Point_3, K::Vector_3>> path_normals;
        
        auto t1 = std::chrono::system_clock::now();
        {
            // 第一步：重新生成网格
            std::cout << "remeshing...";
            remesh(mesh, config);
            repair_mesh(mesh);
            if (is_local_test)
                CGAL::draw(mesh);
            // 第二步：计算各顶点到边界的距离
            std::cout << "done.\n first computing geodesic distance...";
           
            Vertex_distance_map vertex_distance = mesh.add_property_map<vertex_descriptor, double>("v:distance", 0).first;
            shortest_path_map_face vertexFace = mesh.add_property_map<vertex_descriptor,face_descriptor>("vertex_Face", face_descriptor()).first;
            compute_geodesic_distance_to_border(mesh, vertex_distance, vertexFace);

            if (config.skeletonization)
            {
                std::cout << "done.\n Extract Skeleton...";
                getSkeleton(mesh, vertex_distance, vertexFace);
                if(is_local_test)
                    CGAL::draw(mesh);
                std::cout << "Skeleton done.\n second computing geodesic distance...";
                compute_geodesic_distance_to_border(mesh, vertex_distance, vertexFace);
            }

            //
           
            //refine_nearborder(mesh, vertex_distance);
            // 第三步：检测等距线
            std::cout << "done.\n detecting isolines on refined mesh...";
            auto [firstCircle, isoline_topology] = detect_isolines(mesh, vertex_distance, vertexFace);
            std::cout << "done.\n start route...";
            if (is_local_test)
                CGAL::draw(mesh);

            // 第四步：导出规划的路径
            path_normals.reserve(mesh.num_vertices());
            std::cout << "done.\n dumping point set..." << std::endl;

            //调整起始点
            vertex_descriptor start_point;
            start_point = distance_to_boundary(mesh, isoline_topology, config.startpointinner);

            /*path*/ path_normals = dump_path(mesh, isoline_topology, start_point /*isoline_topology.begin()->first*/);
        }
        auto t2 = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        std::cout << "done.\n time cost: " << duration.count() * 1e-3 << std::endl;
        return /*path*/path_normals;
    }

    vertex_descriptor distance_to_boundary(const Mesh& mesh, std::unordered_map<vertex_descriptor, std::list<vertex_descriptor>> isoline_topology,bool inorout)
    {
        vertex_descriptor start_pointregion;
        std::unordered_map<K::Point_3, vertex_descriptor> unmap(2 * isoline_topology.size());
        for (auto [v, nb] : isoline_topology)
        {
            unmap[mesh.point(v)] = v;  
        }

        std::vector<edge_descriptor> edges_border;
        edges_border.reserve(mesh.number_of_edges());
        PMP::border_halfedges(faces(mesh), mesh, boost::make_function_output_iterator(halfedge2edge(mesh, edges_border)));

        double min_distance = std::numeric_limits<double>::max(); // 初始化最小距离为最大值
        double max_distance = std::numeric_limits<double>::min(); // 初始化最小距离为最大值
        for (auto vertexpoint :unmap)
        {
            double min_curdispointroborder = std::numeric_limits<double>::max();
            double max_curdispointroborder = std::numeric_limits<double>::min(); 
            for (const auto& border_ele : edges_border)
            {
                auto sourcepoint = source(border_ele, mesh);
                auto targetpoint = target(border_ele, mesh);

                // 计算点到这条边的距离  
                double disource = CGAL::to_double(CGAL::squared_distance(vertexpoint.first, mesh.point(sourcepoint)));
                double distarget = CGAL::to_double(CGAL::squared_distance(vertexpoint.first, mesh.point(targetpoint)));
                min_curdispointroborder = std::min(min_curdispointroborder, distarget / 2 + disource / 2);
                max_curdispointroborder = std::max(max_curdispointroborder, distarget / 2 + disource / 2);
            }
            if (inorout)
            {
                 //内部
                if (max_distance < max_curdispointroborder)
                {
                    max_distance = max_curdispointroborder;
                    start_pointregion = vertexpoint.second;
                }
            }
            else
            {
                if (min_distance > min_curdispointroborder)
                {
                    min_distance = min_curdispointroborder;
                    start_pointregion = vertexpoint.second;
                }
            }
        }

        return start_pointregion; // 返回最小距离  
    }



    int remesh(Mesh& mesh, const PathPlannerConfig& config) const
    {
        // 查找并粗加细边界
        std::vector<edge_descriptor> border; border.reserve(mesh.number_of_edges());
        PMP::border_halfedges(faces(mesh), mesh, boost::make_function_output_iterator(halfedge2edge(mesh, border)));
        PMP::split_long_edges(border, config.epsilon_global / 2, mesh);

        // 重做各向同性网格
        PMP::isotropic_remeshing(faces(mesh), config.epsilon_global, mesh,
            CGAL::parameters::number_of_iterations(5)
            .protect_constraints(true)); //i.e. protect border, 

        // 精加细边界
        border.clear();
        PMP::border_halfedges(faces(mesh), mesh, boost::make_function_output_iterator(halfedge2edge(mesh, border)));
        //PMP::split_long_edges(border, config.epsilon_border, mesh);
        std::unordered_set<edge_descriptor> near_border_edges(border.begin(), border.end());
        std::unordered_set<vertex_descriptor> near_border_vertices;
        for (int i = 0; i < config.near_border_iters; ++i)
        {

            for (auto e : near_border_edges) 
            {
                near_border_vertices.insert(source(e, mesh));
                near_border_vertices.insert(target(e, mesh));
            }
            for (auto v : near_border_vertices)
            {
                for (auto h : halfedges_around_target(mesh.halfedge(v), mesh))
                {
                    near_border_edges.insert(edge(h, mesh));
                }
            }

        }
        PMP::split_long_edges(near_border_edges, config.epsilon_border, mesh);
        return EXIT_SUCCESS;
    }



    void getSkeleton(Mesh& mesh, const Vertex_distance_map& vertex_distance,const shortest_path_map_face& vertex_distance_face)
    {
        std::unordered_set<face_descriptor> skeleton_faces;
        skeleton_faces.reserve(num_faces(mesh));
        std::vector<edge_descriptor> edgesValid(mesh.number_of_edges());
        //create a property on edges to indicate whether they are constrained
        Mesh::Property_map<edge_descriptor, bool> is_constrained_map =
            mesh.add_property_map<edge_descriptor, bool>("e:is_constrained", true).first;
        std::vector<bool> is_selected(num_faces(mesh), false);

        // 检测并记录等距线与各边交点
        for (auto ed : edges(mesh)) 
        {
            if (is_constrained_map[ed])
            {
                auto vs = source(ed, mesh);
                auto vt = target(ed, mesh);
                auto ds = (vertex_distance[vs] - config.gap_border) / config.gap_inner;
                auto dt = (vertex_distance[vt] - config.gap_border) / config.gap_inner;
                auto facevs = vertex_distance_face[vs];
                auto facevt = vertex_distance_face[vt];
                auto btwPointDis = distance_between_faces(mesh, facevs, facevt);

                // 当边的两个端点到边界的距离跨过等距线时，插值得到等距线上点  //todo 示意图
                if (btwPointDis > config.min_lineswidth)
                {
                    // insert all faces incident to the target vertex
                    for (halfedge_descriptor h : halfedges_around_target(halfedge(ed, mesh), mesh))
                    {
                        if (!is_border(h, mesh))
                        {
                            face_descriptor f = face(h, mesh);
                            if (!is_selected[f])
                            {
                                skeleton_faces.insert(f);
                                is_selected[f] = true;
                                edgesValid.emplace_back(ed);
                            }
                        }
                    }
                }
            }
        }
        std::vector<face_descriptor> skeleton_faces_vec(skeleton_faces.begin(), skeleton_faces.end());
        // increase the face selection 扩展范围
        CGAL::expand_face_selection(skeleton_faces_vec, mesh, 1,
            Vector_pmap_wrapper(is_selected), std::back_inserter(skeleton_faces_vec));
        std::cout << skeleton_faces_vec.size()
            << " faces were selected for the remeshing step\n";
        // remesh the region around the intersection polylines
        PMP::isotropic_remeshing(skeleton_faces_vec, 0.05, mesh,
            CGAL::parameters::/*edge_is_constrained_map(is_constrained_map).*/protect_constraints(true));
        //PMP::split_long_edges(edge_descriptors, 0.01, mesh);
    }

    void getSkeletonCenter(Mesh& mesh, std::vector<std::vector<K::Point_3>>& skeleton)
    {
    
    
    
    }
    double distance_between_faces(const Mesh& mesh, face_descriptor f1, face_descriptor f2) {
        // 获取面 f1 和 f2 的顶点  
        std::vector<K::Point_3> points_f1, points_f2;

        //for (auto face : faces(mesh))
        //{
        //    if (face == f1)
        //    {
        //        auto hf = mesh.halfedge(f1);
        //        for (auto hi : CGAL::halfedges_around_face(hf, mesh))
        //        {
        //            auto vi = target(hi, mesh);
        //            points_f1.push_back(mesh.point(vi));
        //        }
        //    }
        //    else if (face == f2)
        //    {
        //        auto hf = mesh.halfedge(f2);
        //        for (auto hi : CGAL::halfedges_around_face(hf, mesh))
        //        {
        //            auto vi = target(hi, mesh);
        //            points_f2.push_back(mesh.point(vi));
        //        }
        //    }
        //    if (points_f1.size() >= 1 && points_f2.size() >= 1)
        //    {
        //        break;
        //    }
        //}

        for (auto ss : CGAL::vertices_around_face(mesh.halfedge(f1), mesh))
        {
            points_f1.push_back(mesh.point(ss));
        }
        for (auto tt : CGAL::vertices_around_face(mesh.halfedge(f2), mesh))
        {
            points_f2.push_back(mesh.point(tt));
        }

        double min_distance = std::numeric_limits<double>::max();
        // 计算面 f1 和 f2 的所有顶点之间的最小距离 
        for (const auto& p1 : points_f1) {
            for (const auto& p2 : points_f2) {
                double dist = CGAL::sqrt(CGAL::to_double(CGAL::squared_distance(p1, p2)));
                if (dist < min_distance) {
                    min_distance = dist;
                }
            }
        }

        return min_distance;
    }


    int refine_nearborder(Mesh &mesh, Vertex_distance_map &vertex_distance)
    {
        const double th = 2.0;
        std::vector<vertex_descriptor> near_border_vertices;
        for (auto v : vertices(mesh)) {
            if (vertex_distance[v] < th) {
                near_border_vertices.push_back(v);
            }
        }
        std::unordered_set<face_descriptor> near_border_faces;
        for (auto v : near_border_vertices) {
            for (auto f : faces_around_target(mesh.halfedge(v), mesh)) {
                near_border_faces.insert(f);
            }
        }

        std::vector<face_descriptor> new_facets;
        std::vector<vertex_descriptor> new_vertices;
        repair_mesh(mesh);
        PMP::isotropic_remeshing(near_border_faces, 0.05, mesh,
            CGAL::parameters::number_of_iterations(5)
            .protect_constraints(true)); //i.e. protect border, 
        return EXIT_SUCCESS;
    }


    void repair_mesh(Mesh& mesh) const
    {
        CGAL::Polygon_mesh_processing::remove_isolated_vertices(mesh);
        CGAL::Polygon_mesh_processing::remove_connected_components_of_negligible_size(mesh);
        CGAL::Polygon_mesh_processing::remove_almost_degenerate_faces(mesh);
    }

    // 提取边界
    std::tuple<border_edges_type, border_vertices_type> extract_border(Mesh& mesh) const
    {
        std::vector<edge_descriptor> border_ed; border_ed.reserve(mesh.number_of_edges());
        std::vector<vertex_descriptor> border_vd; border_vd.reserve(mesh.number_of_vertices());
        std::vector<K::Point_3> border_points; border_points.reserve(mesh.number_of_vertices());
        PMP::border_halfedges(faces(mesh), mesh, boost::make_function_output_iterator(halfedge2edge(mesh, border_ed)));
        for (auto e : border_ed)
        {
            border_vd.push_back(source(e, mesh));  //todo 每个面片的起点？
        }
        return std::make_tuple(border_ed, border_vd);
    }

    //计算点到边界的最短距离
    void compute_geodesic_distance_to_border(Mesh& mesh, Vertex_distance_map& vertex_distance, shortest_path_map_face& rtnShortestFace)
    {
        std::vector<CGAL::Surface_mesh_shortest_path<Traits>::Shortest_path_result> shortest_path_results;
        auto [border_ed, border_vd] = extract_border(mesh);
        if (config.use_heat_method) {
            CGAL::Heat_method_3::estimate_geodesic_distances(mesh, vertex_distance, border_vd);
        }
        else {
            Surface_mesh_shortest_path shortest_paths(mesh);
            std::vector<Surface_mesh_shortest_path> shortDisFace;
            shortest_paths.add_source_points(border_vd.begin(), border_vd.end());
            for (auto v : vertices(mesh))
            {
                auto shortPathVerFace = shortest_paths.shortest_distance_to_source_points(v);
                vertex_distance[v] = shortPathVerFace.first;
                rtnShortestFace[v] = shortPathVerFace.second->first;
            }
        }
    }

    std::tuple<vertex_descriptor, isoline_topology_type> detect_isolines(Mesh& mesh, const Vertex_distance_map& vertex_distance, shortest_path_map_face& vertex_to_face)
    {
        std::unordered_map<face_descriptor, std::vector<std::pair<int, vertex_descriptor>>> isoline_faces(mesh.number_of_faces());
        std::unordered_map<vertex_descriptor, std::list<vertex_descriptor>> isoline_topology(mesh.number_of_vertices());
        // 检测并记录等距线与各边交点
        for (auto ed : edges(mesh)) {
            auto vs = source(ed, mesh);
            auto vt = target(ed, mesh);

            //判断该边是否为边界边
            //todo 此处可以增加判断，如果当前距离非常小时，再判断与天线骨骼的最短距离，且此时的骨骼距离边界也小于笔宽，如果当前距离与最短距离之差小于笔宽，则认为该边为窄边，然后直接提取
            //骨骼点作为等距线上的点；同时也可判断是不是骨骼点，定位到的骨骼点到边界的距离是否超过笔宽，如果超过，将该区域增加一笔。
            auto ds = (vertex_distance[vs] - config.gap_border) / config.gap_inner;
            auto dt = (vertex_distance[vt] - config.gap_border) / config.gap_inner;

            vertex_descriptor anti_vertex;
            //int is_centerLine = thinLines(mesh, vertex_distance, vertex_to_face, vs, vt, ed, anti_vertex);
            halfedge_descriptor hda = halfedge(ed, mesh);
            halfedge_descriptor hdb = opposite(hda, mesh);
            face_descriptor fda = face(hda, mesh);
            face_descriptor fdb = face(hdb, mesh);
            //if (is_centerLine==0)
            //{ 
                std::cout << "normal lines start" << std::endl;
                //是否反向，中间部分
                if (ds > dt) {
                    std::swap(ds, dt);
                    std::swap(vs, vt);
                }
                // 当边的两个端点到边界的距离跨过等距线时，插值得到等距线上点  //todo 示意图
                if (std::ceil(dt) - std::floor(ds) > 1) {
                    K::Point_3 ps = mesh.point(vs);
                    K::Point_3 pt = mesh.point(vt);

                    for (int i = std::ceil(ds); i <= std::floor(dt); ++i) {
                        auto t = (i - ds) / (dt - ds);
                        auto p = ps + t * (pt - ps);
                        auto vd = mesh.add_vertex(p);
                        isoline_faces[fda].emplace_back(i, vd);
                        isoline_faces[fdb].emplace_back(i, vd);
                    }
                }
                std::cout << "normal lines end" << std::endl;
            //}
            ////窄线的处理
            //else
            //{
            //    std::cout << "thin linrs start" << std::endl;
            //    auto dis_boundry = 0;
            //    if (is_centerLine == 1)
            //    {
            //        //比较窄的线短的边
            //        dis_boundry  = vertex_distance[vs] + vertex_distance[vt];
            //    }
            //    else if(is_centerLine == 2)
            //    {
            //        dis_boundry = vertex_distance[vs] / 2 + vertex_distance[vt] / 2 + vertex_distance[anti_vertex];
            //    }

            //    auto points_center = K::Point_3(mesh.point(vs).x() / 2 + mesh.point(vt).x() / 2,
            //        mesh.point(vs).y() / 2 + mesh.point(vt).y() / 2,
            //        mesh.point(vs).z() / 2 + mesh.point(vt).z() / 2);
            //    auto vd = mesh.add_vertex(points_center);
            //    //窄线
            //    if (dis_boundry <= 2 * config.gap_border)
            //    {
            //        isoline_faces[fda].emplace_back(0, vd);
            //        isoline_faces[fdb].emplace_back(0, vd);
            //        std::cout << "thin lines \n" << std::endl;
            //    }
            //    //线比较宽,中间位置
            //    else
            //    {
            //        int pos = std::ceil((dt + ds) / 2);
            //        isoline_faces[fda].emplace_back(pos, vd);
            //        isoline_faces[fdb].emplace_back(pos, vd);
            //        std::cout << "center lines \n" <<std::endl;
            //    }
            //}
        }
        std::cout << "isoline_faces.size():" << isoline_faces.size() << std::endl;
        vertex_descriptor rtnVertex;
        // 添加等距线上的边
        for (auto& [key, candidates] : isoline_faces) {
            std::sort(candidates.begin(), candidates.end());
            int np = candidates.size();
            for (int i = 0; i < np; ++i) {
                for (int j = i + 1; j < np && candidates[i].first == candidates[j].first; ++j) {
                    if (candidates[i].first < 0) continue;
                    vertex_descriptor vs = candidates[i].second;
                    vertex_descriptor vt = candidates[j].second;
                    isoline_topology[vs].push_back(vt);
                    isoline_topology[vt].push_back(vs);
                }
            }
        }
        
        return std::make_tuple(rtnVertex, isoline_topology);
    }

    /// <summary>
    /// 判断当前边是否为窄边
    /// </summary>
    /// <param name="mesh"></param>
    /// <param name="vertex_distance"></param>
    /// <param name="vertex_to_face"></param>
    /// <param name="vs"></param>
    /// <param name="vt"></param>
    /// <returns></returns>
    int thinLines(Mesh& mesh, const Vertex_distance_map& vertex_distance, shortest_path_map_face& vertex_to_face,
        vertex_descriptor vs, vertex_descriptor vt,CGAL::SM_Edge_index ed, vertex_descriptor& anti_vertex)
    {
        int is_center_tri = 0;
        auto ds_face = vertex_to_face[vs];
        auto dt_face = vertex_to_face[vt];
        std::vector<K::Point_3> points_f1;
        std::vector<K::Point_3> points_f2;

        for (auto ss : CGAL::vertices_around_face(mesh.halfedge(ds_face), mesh))
        {
            points_f1.push_back(mesh.point(ss));
        }
        for (auto tt : CGAL::vertices_around_face(mesh.halfedge(dt_face), mesh))
        {
            points_f2.push_back(mesh.point(tt));
        }

        std::cout <<"halfedge1" << std::endl;
        auto point1_center = CGAL::centroid(points_f1.begin(), points_f1.end());
        auto point2_center = CGAL::centroid(points_f2.begin(), points_f2.end());

        auto vs_point = mesh.point(vs);
        auto vt_point = mesh.point(vt);
        auto vs_vt_center = K::Point_3(vs_point.x() / 2 + vt_point.x() / 2,
            vs_point.y() / 2 + vt_point.y() / 2, vs_point.z() / 2 + vt_point.z() / 2);
        double angle = get_angle(point1_center, point2_center, vs_vt_center);
        if (angle > M_PI * 3 / 4)
        {
            //认为该边所属的face为中间face，但还需要进一步判断当前
            is_center_tri = 1;  //两边本身反向
            return is_center_tri;
        }
        else
        {
            //判读 vs、vt 共面的另外一个点的方向
            // 获取与边ed相关的一个半边
            auto he = halfedge(ed, mesh);
            // 获取与he相对的另一个半边
            auto opposite_he = opposite(he, mesh);
            // 获取两个半边所在的面
            auto face1 = face(he, mesh);
            auto face2 = face(opposite_he, mesh);

            K::Point_3 he_point;
            K::Point_3 opposite_he_point;  
            {
                //只有有一个相邻的三角面片角度不满足都说明不在一边
                face_descriptor temp_face = face1;
                face_descriptor temp_face2 = face2;
                std::cout << "halfedge2" << std::endl;
                for (int i = 0; i < 2; i++)
                {
                    std::cout << "halfedge3" << std::endl;
                    if(i == 0)
                        temp_face = face1;
                    else
                        temp_face = face2;
                    for (auto ss : CGAL::vertices_around_face(mesh.halfedge(temp_face), mesh))
                    {
                        std::cout << "halfedge4" << std::endl;
                        if (!(mesh.point(ss) == vs_point || mesh.point(ss) == vt_point))
                        {
                            auto ss_face = vertex_to_face[ss];
                            std::vector<K::Point_3> points_f;
                            for (auto ss_1 : CGAL::vertices_around_face(mesh.halfedge(ss_face), mesh))
                            {
                                points_f.push_back(mesh.point(ss_1));
                            }
                            auto point_center = CGAL::centroid(points_f.begin(), points_f.end());
                            auto pp_point = mesh.point(ss);
                            auto tri_angle_center = K::Point_3((vs_vt_center.x() / 2 + pp_point.x()) / 3,
                                (vs_vt_center.y() / 2 + pp_point.y()) / 3, (vs_vt_center.z() / 2 + pp_point.z()) / 3);
                            angle = get_angle(point_center, point1_center, tri_angle_center);
                            if (angle > M_PI * 3 / 4)
                            {
                                anti_vertex = ss; //临边三角型反向
                                is_center_tri = 2;
                                return is_center_tri;
                                //break;
                            }
                        }
                    }
                }
            }
        }
        return 0;
    }


    /// <summary>
    /// 获取夹角
    /// </summary>
    /// <param name="vs_point"></param>
    /// <param name="vt_point"></param>
    /// <param name="point_center"></param>
    /// <returns></returns>
    double get_angle(K::Point_3 vs_point, K::Point_3 vt_point, K::Point_3 point_center)
    {
        auto vini = vs_point - point_center;
        auto vfin = vt_point - point_center;
        auto dot_product = vini * vfin;
        double length_1 = sqrt(vini.squared_length());
        double length_2 = sqrt(vfin.squared_length());
        double cos_angle = dot_product / (length_1 * length_2);
        // 做边界检查  
        if (cos_angle < -1.0) cos_angle = -1.0;
        if (cos_angle > 1.0) cos_angle = 1.0;    
        return std::acos(cos_angle);
    }



    
    std::vector<std::pair<K::Point_3, K::Vector_3>> dump_path(Mesh& mesh, isoline_topology_type& isoline_topology, vertex_descriptor vini)
    {
        auto vnormals = mesh.add_property_map<vertex_descriptor, K::Vector_3>("v:normals", CGAL::NULL_VECTOR).first;
        CGAL::Polygon_mesh_processing::compute_vertex_normals(mesh, vnormals);

        repair_mesh(mesh);
        // 建搜索树 
        std::unordered_map<K::Point_3, vertex_descriptor> unmap(2 * isoline_topology.size());
        std::vector<K::Point_3> points; points.reserve(isoline_topology.size());
        for (auto [v, nb] : isoline_topology) {
            points.push_back(mesh.point(v));
            unmap[mesh.point(v)] = v;   
        }

        Tree tree(points.begin(), points.end());
        std::unordered_map<K::Point_3, K::Vector_3> isoline_topology_point_normal = getUnorderNormal(mesh, isoline_topology);

        // 逐圈建路径
        std::vector<K::Point_3> path;
        std::vector<std::pair<K::Point_3, K::Vector_3>> pointNormal;
        pointNormal.reserve(isoline_topology.size());
        std::unordered_set<vertex_descriptor> vis(isoline_topology.size());
        vertex_descriptor v = vini;
        std::vector<std::pair<K::Point_3, K::Vector_3>> circlepath;  //一个轮廓线
        int njump = 0;
        std::vector<std::pair<K::Point_3, K::Vector_3>> rtnPath; rtnPath.reserve(isoline_topology.size());
        static int circlenum = 0;
        
        std::vector<std::pair<K::Point_3, K::Vector_3>> valildeOnepath;  //一笔画有效数据
        //evecircledata.reserve();

        while (vis.size() < isoline_topology.size()) {
            // 沿等距线深度优先搜索
            std::stack<vertex_descriptor> frontier;
            vis.insert(v);
            frontier.push(v); frontier.push(v);//todo 
            path.push_back(mesh.point(v));

            auto ttpoint = mesh.point(v);
            circlepath.push_back(std::make_pair(mesh.point(v), isoline_topology_point_normal[ttpoint]));
            pointNormal.push_back(std::make_pair(mesh.point(v), isoline_topology_point_normal[ttpoint]));
            valildeOnepath.push_back(std::make_pair(mesh.point(v), isoline_topology_point_normal[ttpoint]));

            while (!frontier.empty()) {
                // 插入当前点到下一个点的路径
                v = frontier.top(); frontier.pop();
                auto pv = mesh.point(v);
                auto pw = path.back();
                auto dist = std::sqrt((pv - pw).squared_length());
                int ninterp = /*std::ceil(*/dist / config.gap_inner/*)*/;
                for (int k = 1; k <= ninterp; ++k) 
                {
                    K::Point_3 p = pw + k / double(ninterp) * (pv - pw);
                    auto vv = mesh.point(v);
                    path.push_back(p);
                    circlepath.push_back(std::make_pair(p, isoline_topology_point_normal[vv]));
                    pointNormal.push_back(std::make_pair(p, isoline_topology_point_normal[vv]));
                    valildeOnepath.push_back(std::make_pair(p, isoline_topology_point_normal[vv]));
                }
                // 搜索
                for (auto w : isoline_topology[v]) {
                    if (vis.find(w) == vis.end()) {
                        vis.insert(w);
                        frontier.push(w);
                    }
                }
            }

            ////精简数据
            std::vector<std::pair<K::Point_3, K::Vector_3>> tempPath = voxelGridFilter(circlepath, config.gap_inner / 2);
            if (config.ismuitlpenwidth)
            {
                btwLinesNearestMerger(tempPath);
            }
            rtnPath.insert(rtnPath.end(), tempPath.begin(), tempPath.end());
            tempPath.clear();


            K::Point_3 nextPoint;
            // 寻找最近的下一圈路径
            if (vis.size() == isoline_topology.size()) break;
            size_t k = 5;
            bool acc = false;
            do {
                k = std::min(2 * k, isoline_topology.size());
                Neighbor_search search(tree, mesh.point(v), k);

                ////保证第一个下一圈的点就是最近的
                std::vector<K::Point_3> searchPoint;
                for (auto ele : search)
                {
                    searchPoint.emplace_back(ele.first);
                }
                CGAL::Epick::Point_3 tempPoint = mesh.point(v); 
                std::sort(searchPoint.begin(), searchPoint.end(), [&tempPoint](const K::Point_3 & a, const K::Point_3 & b)
                {
                    return std::sqrt((tempPoint - a).squared_length()) < std::sqrt((tempPoint - b).squared_length());
                });

                int cnt = 0;
                for (auto& item : searchPoint)
                {
                    ++cnt;
                    auto u = unmap[item];
                    if (vis.find(u) == vis.end()) {
                        acc = true;
                        v = u;
                        nextPoint = item;
                        break;
                    }
                }
               
            } while (!acc);
            circlepath.clear();
            float disbtwroute = std::sqrt((nextPoint - mesh.point(v)).squared_length());
            if (disbtwroute > config.gap_inner)
            {
                evecircledata.emplace_back(valildeOnepath);
                valildeOnepath.clear();
            }
        }
        if (!config.datasample)
        {
            return pointNormal;
        }
        else
        {
            return rtnPath;
        }
    }

    /// <summary>
    /// 路径分类,先形成单独的分类，再合并
    /// </summary>
    /// <param name="evecircledata"></param>
    void btwLinesNearestMerger(std::vector<std::pair<K::Point_3, K::Vector_3>>& evecircledata)
    {
        double mindis = DBL_MAX;
        int minindex = -1;
        double mindis0 = DBL_MAX;
        int minindex0 = -1;
        std::vector<K::Point_3>  temppoindata;
        temppoindata.reserve(evecircledata.size());
        for (const auto& path : evecircledata)
        {
            temppoindata.push_back(path.first);
        }

        for (int i = 0; i < allpath.size(); ++i)
        {
            for (auto& path : allpath[i])
            {
                std::vector<K::Point_3> pathpoints;
                pathpoints.reserve(path.size());
                for (auto np : path)
                    pathpoints.push_back(np.first);
                if (isFinishTouch(temppoindata, pathpoints))
                {
                    path.insert(path.end(), evecircledata.begin(), evecircledata.end());
                    return;
                } 
            }
        }

        std::vector<std::vector<std::pair<K::Point_3, K::Vector_3>>> newallpath;
        newallpath.push_back(evecircledata);
        allpath.emplace_back(newallpath);
    }

    void mergerLinesType(isoline_points_type_path inallpath)
    {
        isoline_points_type_part allpathpart;
        allpathpart.reserve(inallpath.size());
        for (size_t i = 0; i < inallpath.size(); i++)
        {
            allpathpart.emplace_back(*inallpath[i].begin());
        }
        allpath.clear();
        isoline_points_type_part newallpath;
        newallpath.emplace_back(*allpathpart.begin());
        allpath.emplace_back(newallpath);

        for (size_t i = 1; i < allpathpart.size(); i++)
        {
            std::vector<K::Point_3> dd;
            dd.reserve(allpathpart[i].size());
            for (auto &aa : allpathpart[i])
            {
                dd.emplace_back(aa.first);
            }
            bool isflag = false;
            for (auto &line : allpath)
            {
                auto evelastpath = line.back();

                //先收尾判断，这是最易判断是同一种类型的
                std::vector<K::Point_3> tempgetevelastpath;
                tempgetevelastpath.reserve(evelastpath.size());
                for (auto np : evelastpath) 
                    tempgetevelastpath.push_back(np.first);
                float dis1 = btwLinesWidth(dd, tempgetevelastpath);
                float dis2 = btwLinesWidth(tempgetevelastpath, dd);
                if (dis1 < config.gap_inner * config.gap_inner * 1.5 && dis2 < config.gap_inner * config.gap_inner * 1.5)
                {
                    line.emplace_back(allpathpart[i]);
                    isflag = true;
                    break;
                }
            }
            if (!isflag)
            {
                isoline_points_type_part newpath;
                newpath.emplace_back(allpathpart[i]);
                allpath.emplace_back(newpath);
            }
        }
    }



    float btwLinesWidth(std::vector<K::Point_3> line0, std::vector<K::Point_3> line1)
    {
        float mindis = 0;
        double minnumpointdis = 0;
        int minnumpoint = line0.size();
        Tree tree(line1.begin(), line1.end());
        for (size_t j = 0; j < minnumpoint;)
        {
            int searchnum = 5;
            Neighbor_search search(tree, line0[j], searchnum);
            ////保证第一个下一圈的点就是最近的
            minnumpointdis = minnumpointdis +  search.advanced_begin()->second;
            j = j + 3;
        }
        mindis = minnumpointdis / minnumpoint * 3;
        return mindis;
    }

    /// <summary>
    /// 两个路径是否非常相近
    /// </summary>
    /// <param name="line0"></param>
    /// <param name="line1"></param>
    /// <returns></returns>
    bool isFinishTouch(std::vector<K::Point_3> line0, std::vector<K::Point_3> line1)
    {
        bool isfinishtouch = false;
        //求最近的点
        Tree tree(line1.begin(), line1.end());
        int searchnum = 3;
        for (size_t i = 0; i < line0.size(); i++)
        {
            Neighbor_search search(tree, line0[i], searchnum);
            if (search.advanced_begin()->second < config.gap_inner * config.gap_inner * 0.5)
            {
                isfinishtouch = true;
                return isfinishtouch;
            }
        }
        return isfinishtouch;
    }

    /// <summary>
    /// 按照最小离散0.04，按照不同笔宽整个路径 需要配置前后间距,体素滤波的方式
    /// </summary>
    /// <param name="mesh"></param>
    /// <param name="allpath"></param>
    /*std::vector<line_width>*/ void mergerLinesDiffWidth(isoline_points_type_path& allpath)
    {
        line_width onepart;
        std::cout << "allpath" << allpath.size() << std::endl;
        //所有不同路径组成的集合
        for (auto pathi : allpath)
        {
            //只有一条路径时
            if (pathi.size() == 1)
            {
                line_width tempele;
                tempele.push_back(std::pair(pathi[0], 0.04));
                linesclassify.emplace_back(tempele);
            }
            else
            {
                int bigpennums = (pathi.size() + 1) / (config.maxprintwidth / config.minprintwidth);
                int minpennums = (pathi.size() + 1) % (int)(config.maxprintwidth / config.minprintwidth);
                std::vector<std::pair<K::Point_3, K::Vector_3>> tempmergerroute;
                for (int i = 0; i < bigpennums; i++)
                {
                    tempmergerroute.clear();
                    for (size_t j = 0; j < (int)(config.maxprintwidth / config.minprintwidth); j++)
                    {
                        tempmergerroute.insert(tempmergerroute.end(),pathi[i * (int)(config.maxprintwidth / config.minprintwidth) + j].begin(), pathi[i * (int)(config.maxprintwidth / config.minprintwidth) + j].end());
                    }
                    std::vector<std::pair<K::Point_3, K::Vector_3>> merdoneroutdata = voxelGridFilter(tempmergerroute, config.maxprintwidth + 0.01);
                    onepart.push_back(std::pair(merdoneroutdata, config.maxprintwidth));
                }

                tempmergerroute.clear();
                if (minpennums != 0)
                {
                     for (size_t j = 0; j < minpennums - 1; j++)
                    {
                        tempmergerroute.insert(tempmergerroute.end(),pathi[bigpennums * (int)(config.maxprintwidth / config.minprintwidth) + j].begin(), pathi[bigpennums * (int)(config.maxprintwidth / config.minprintwidth) + j].end());
                    }
                     
                    std::vector<std::pair<K::Point_3, K::Vector_3>> merdoneroutdata = voxelGridFilter(tempmergerroute, minpennums *config.minlines + 0.01);
                    onepart.push_back(std::pair(merdoneroutdata, minpennums * config.minlines / 2));
                }
                linesclassify.push_back(onepart);
            }
        }
    }
#if 1  

    void rangeLinespre()
    {
        std::vector<std::unordered_set<int>> allpathdivisors;
        allpathdivisors.reserve(allpath.size());
        for (const auto& ele : allpath)
        {
            allpathdivisors.emplace_back(getLinesWidthPrintWidthComDivi(ele.size()));
        }
        std::vector<std::pair<int, int>> penwidthmap = makeBestPenWidth(allpathdivisors);
        rangeLines(penwidthmap);
        mergerLines();
    }

    /// <summary>
    /// 整理不同笔宽的路径，将同一笔宽且相邻的路径合并
    /// </summary>
    void mergerLines()
    {
        std::vector<line_width> tempLines;
        tempLines.reserve(linesclassify.size());
        tempLines.emplace_back(*linesclassify.begin());
        for (size_t i = 1; i < linesclassify.size(); i++)
        {
            auto tp = linesclassify[i];
            if (tp.size() != 0 && abs( tp.at(tp.size() - 1).second - tempLines.back().at(tempLines.back().size() - 1).second )< 0.001)
            {
                std::vector<std::pair<K::Point_3, K::Vector_3>> tp1;
                tp1 = tp.back().first;
                tempLines.back().back().first.insert(tempLines.back().back().first.end(), tp1.begin(), tp1.end());
                continue;
            }
            tempLines.emplace_back(tp);
        }
        linesclassify.clear();
        linesclassify.insert(linesclassify.end(), tempLines.begin(), tempLines.end());
    }

    /// <summary>
    /// 较小换笔次数，还是只用最大笔宽
    /// </summary> kl/
    /// <param name="num"></param>
    /// <param name="mode"></param>
    /// <returns></returns>
    std::unordered_set<int> getLinesWidthPrintWidthComDivi(int num,int mode = 0)
    {
        std::unordered_set<int> divisors;
        divisors.insert(1);

        //没有达到最大笔宽，把所有公约数加入进去
        if (num < (int)std::floor(config.maxprintwidth / config.minprintwidth))
        {
            divisors.insert(num);
            for (size_t i = 2; i <= num / 2; i++) 
            {
                if (num % i == 0)
                    divisors.insert(i);
            }
        }
        else
        {  
            //todo 这样处理会出现一个问题，就是最大笔宽的数量不能够确认 都是用最大笔宽下，时间以及换笔次数是否有个比较好的平衡
            int maxprintnum = (int)std::floor(config.maxprintwidth / config.minprintwidth);
            for (size_t i = 2; i <= maxprintnum; i++)
            {
                if (num % i == 0 && maxprintnum % i == 0)
                {
                    divisors.insert(i);
                }
            }
        }
        return divisors;
    }


    std::vector<std::pair<int, int>> makeBestPenWidth(std::vector<std::unordered_set<int>>& linesclassify)
    {
        std::vector<int> indexnum((int)std::floor(config.maxprintwidth / config.minprintwidth),0);
        for (auto ele : linesclassify)
        {
            for (auto ele2 : ele)
            {
                indexnum[ele2 - 1]++;
            }
        }
        std::unordered_map<int,int> indexnum_map;
        for (size_t i = 0; i < indexnum.size(); i++)
        {
            if (indexnum[i] == 0)
            {
                continue;
            }
            indexnum_map[i] = indexnum[i];
        }

        std::vector<std::pair<int, int>> vec(indexnum_map.begin(), indexnum_map.end());

        // 根据值进行排序  
        std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
            return a.second > b.second; 
            });

        return vec;
    }

    /// <summary>
    /// 更新每一组最佳打印笔宽以及组合
    /// </summary>
    void rangeLines(std::vector<std::pair<int, int>> lineswidthmap)
    {
        for (auto pathi : allpath)
        {
            //只有一条路径时
            if (pathi.size() == 1)
            {
                line_width tempele;
                tempele.push_back(std::pair(pathi[0], 0.08));
                linesclassify.emplace_back(tempele);
            }
            else
            {
                if (pathi.size() < (int)std::floor(config.maxprintwidth / config.minprintwidth))
                {

                    for (auto lineswidth : lineswidthmap)
                    {
                        if (lineswidth.first + 1 == 1)
                            continue;
                        else
                        {
                            if (pathi.size() % (lineswidth.first + 1) == 0)
                            {
                                //笔宽合并，合并之后添加
                                //剩余线宽处理
                                std::vector<std::pair<K::Point_3, K::Vector_3>> reminLinerPoint = looppointsroute(pathi, 0, pathi.size());
                                line_width reminLiner;
                                reminLiner.push_back(std::pair(reminLinerPoint, pathi.size() * 0.08));
                                linesclassify.emplace_back(reminLiner);
                                break;
                            }
                        }
                    }
                }
                else
                {
                    //将操作最大笔宽得线段划分为出现的最多线段
                    for (auto lineswidth : lineswidthmap)
                    {
                        if (lineswidth.first + 1 == 1)
                            continue;
                        else
                        {
                            //std::cout << "2pathi.size():" << pathi.size() << std::endl;
                            int optipenwidthnum =(int) pathi.size() / (lineswidth.first + 1);
                            for (size_t i = 0; i < optipenwidthnum; i++)
                            {
                                //当前最佳线宽处理(0-lineswidth.second)得线段
                                std::vector<std::pair<K::Point_3, K::Vector_3>> afterrengerLiner = looppointsroute(pathi,i * lineswidth.second,(i + 1)* lineswidth.second);
                                line_width tempeleone;
                                tempeleone.push_back(std::pair(afterrengerLiner, (lineswidth.first + 1) * 0.08));
                                linesclassify.emplace_back(tempeleone);

                            }
                            //剩余线宽处理
                            if(optipenwidthnum * lineswidth.second == pathi.size())
                                break;
                            std::vector<std::pair<K::Point_3, K::Vector_3>> reminLinerPoint = looppointsroute(pathi, optipenwidthnum * lineswidth.second, pathi.size());
                            line_width reminLiner;
                            reminLiner.push_back(std::pair(reminLinerPoint, (pathi.size() - optipenwidthnum * (lineswidth.first + 1)) * 0.08));
                            linesclassify.emplace_back(reminLiner);
                            break;
                        }
                    }
                }
            }
        }
    }

    std::vector<std::pair<K::Point_3, K::Vector_3>> looppointsroute(isoline_points_type_part typeroute, int stratpos, int endpos)
    {
        //数据重组，用于查询
        std::unordered_map<K::Point_3, K::Vector_3> pointmap;
        for (size_t i = stratpos; i < endpos; i++)
            for (size_t j = 0; j < typeroute[i].size(); j++)
            {
                pointmap[typeroute[i][j].first] = typeroute[i][j].second;
            }
        std::vector<std::vector<K::Point_3>> result;
        result.reserve(typeroute.size());

        for (size_t i = stratpos; i < endpos; i++)
        {
            std::vector<K::Point_3> temp;
            temp.reserve(typeroute[i].size());
            for (size_t j = 0; j < typeroute[i].size(); j++)
                temp.emplace_back(typeroute[i][j].first);
            result.emplace_back(temp);
        }

        std::vector<std::vector<K::Point_3>> pointnearests;
        std::vector<K::Point_3> firstLinesPloy;
        firstLinesPloy.insert(firstLinesPloy.end(), result[0].begin(), result[0].end());
        int num  = firstLinesPloy.size();
        pointnearests.reserve(result.size());
        pointnearests.emplace_back(firstLinesPloy);

        
        for (size_t j = 1; j < result.size(); j++)
        {
            std::vector<K::Point_3> tempPoints;
            {
                tempPoints.reserve(pointnearests.back().size());
                Tree tree(result[j].begin(), result[j].end());
                for (size_t i = 0; i < pointnearests.back().size(); i++)
                {
                    //K::Point_3 nearpoint = searchShortPoints(pointnearests.back()[i], tree,3);
                    K::Point_3 result;
                    Neighbor_search search(tree, pointnearests.back()[i], 3);
                    result = *search.advanced_begin()->first;
                    tempPoints.emplace_back(result);
                }
            }
            pointnearests.emplace_back(tempPoints);
        }

        std::vector<std::pair<K::Point_3, K::Vector_3>> resultrtn;
        {
            int endnum = pointnearests.begin()->size();
            resultrtn.reserve(endnum);
            int pointnearestssize = pointnearests.size();
            for (size_t i = 0; i < endnum; i++)
            {
                K::Point_3 finallypoint;
                K::Vector_3 normal;
                float tempx = 0;
                float tempy = 0;
                float tempz = 0;
                float normalx = 0;
                float normaly = 0;
                float normalz = 0;

                for (size_t j = 0; j < pointnearestssize; j++)
                {
                    tempx = tempx + pointnearests[j][i].x();
                    tempy = tempy + pointnearests[j][i].y();
                    tempz = tempz + pointnearests[j][i].z();
                    normalx = normalx + pointmap[pointnearests[j][i]].x();
                    normaly = normaly + pointmap[pointnearests[j][i]].y();
                    normalz = normalz + pointmap[pointnearests[j][i]].z();
                }
                normal = K::Vector_3(normalx / pointnearestssize, normaly / pointnearestssize, normalz / pointnearestssize);
                finallypoint = K::Point_3(tempx / pointnearestssize, tempy / pointnearestssize, tempz / pointnearestssize);
                resultrtn.emplace_back(std::pair(finallypoint, normal));
            }
        }
        return resultrtn;
    }

    K::Point_3 searchShortPoints(K::Point_3 typeroutepoint,Tree intree, int knum = 3)
    {
        K::Point_3 result;
        Neighbor_search search(intree, typeroutepoint, knum);
        result = *search.advanced_begin()->first;
        return result;
    }

#endif
    std::unordered_map<K::Point_3, K::Vector_3> getUnorderNormal(Mesh& mesh, isoline_topology_type& isoline_topology)
    {
        std::unordered_map<K::Point_3, K::Vector_3> normalPoint_map;
        normalPoint_map.reserve(isoline_topology.size());

        //获取正常vertex_descriptor的法向
        auto vnormals = mesh.add_property_map<vertex_descriptor, K::Vector_3>("v:normals", CGAL::NULL_VECTOR).first;
        CGAL::Polygon_mesh_processing::compute_vertex_normals(mesh, vnormals);

        std::unordered_map<K::Point_3, K::Vector_3> unpointnormal;
        unpointnormal.reserve(mesh.number_of_vertices());

        //获取正常的vertex的坐标
        std::vector<K::Point_3> orginalpoints; orginalpoints.reserve(mesh.number_of_vertices());
        for (auto elelpoint : vertices(mesh)) 
        { 
            auto vertextpoint = mesh.point(elelpoint);
            orginalpoints.push_back(vertextpoint);
            unpointnormal[vertextpoint] = vnormals[elelpoint];  //vertexpoint与normal是一一对应的
        }
        Tree tree(orginalpoints.begin(), orginalpoints.end());

        //获取交点vertex的坐标
        std::vector<K::Point_3> points; points.reserve(isoline_topology.size());
        for (auto [v, nb] : isoline_topology) {
            points.push_back(mesh.point(v));
        }

        int neibernum = 3;
        for (auto unorderpoint : points)
        {
            Neighbor_search search(tree,unorderpoint, neibernum);
            normalPoint_map[unorderpoint] = unpointnormal[search.begin()->first];
        }
        return normalPoint_map;
    }
   // 体素滤波函数  
    std::vector<std::pair<K::Point_3, K::Vector_3>> voxelGridFilter(std::vector<std::pair<K::Point_3, K::Vector_3>> inputCloud, float voxelSize)
    {
       std::unordered_map<std::tuple<int, int, int>, Voxel, VoxelHash> voxelMap;
       // 对点进行体素化  
       for (const auto& point : inputCloud) {
           int voxelX = static_cast<int>(std::floor(point.first.x() / voxelSize));
           int voxelY = static_cast<int>(std::floor(point.first.y() / voxelSize));
           int voxelZ = static_cast<int>(std::floor(point.first.z() / voxelSize));

           voxelMap[{voxelX, voxelY, voxelZ}].points.push_back(PointVex(point.first.x(), point.first.y(), point.first.z(), point.second.x()
           ,point.second.y(), point.second.z()));
       }

       std::vector<std::pair<K::Point_3, K::Vector_3>> filteredPoints;

       // 对每个体素的点进行处理，简单取均值作为该体素的代表点，由于是hashmap所以所有数据都是无须的
       for (const auto& pair : voxelMap) {
           const Voxel& voxel = pair.second;
           if (!voxel.points.empty()) {
               float sumX = 0, sumY = 0, sumZ = 0;
               float sumNX = 0, sumNY = 0, sumNZ = 0;
               for (const auto& point : voxel.points) {
                   sumX += point.x;
                   sumY += point.y;
                   sumZ += point.z;
                   sumNX += point.nx;
                   sumNY += point.ny;
                   sumNZ += point.nz;
               }
               float avgX = sumX / voxel.points.size();
               float avgY = sumY / voxel.points.size();
               float avgZ = sumZ / voxel.points.size();
               float avgNX = sumNX / voxel.points.size();
               float avgNY = sumNY / voxel.points.size();
               float avgNZ = sumNZ / voxel.points.size();
               K::Point_3 arepoint(avgX, avgY, avgZ);
               K::Vector_3 aretnormal(avgNX, avgNY, avgNZ);

               filteredPoints.emplace_back(arepoint, aretnormal);
           }
       }
       std::vector<std::pair<K::Point_3, K::Vector_3>> rtnfilteredPoints;
       rtnfilteredPoints.reserve(filteredPoints.size());
       K::Point_3 firstpoint;
       K::Vector_3 firstnormal;
       double mindist = std::numeric_limits<double>::max();
       std::pair<K::Point_3, K::Vector_3> tempele;
       for (const auto& point : filteredPoints) 
       {
           double dist = CGAL::squared_distance(inputCloud.begin()->first, point.first);
           if (dist < mindist)
           {
               mindist = dist;
               firstpoint = point.first;
               firstnormal = point.second;
               tempele = point;
           }
       }
       filteredPoints.erase(std::find(filteredPoints.begin(), filteredPoints.end(), tempele));

       rtnfilteredPoints.emplace_back(firstpoint, firstnormal);
       int size = filteredPoints.size();

       if (size == 0)
       {
          return rtnfilteredPoints;
       }
       std::vector<bool> isVisited(size,false);
       //重新排序
       while (true)
       {
            mindist = std::numeric_limits<double>::max();
            K::Point_3 backpoint = rtnfilteredPoints.back().first;
            int pos = 0;
            for (size_t i = 0; i < size; i++)
            {
                if (isVisited[i])
                {
                    continue;
                }
                double dist = CGAL::squared_distance(backpoint, filteredPoints[i].first);
                if (dist < mindist)
                {
                    mindist = dist;
                    firstpoint = filteredPoints[i].first;
                    firstnormal = filteredPoints[i].second;
                    pos = i;
                }
            }
            isVisited[pos] = true;
            rtnfilteredPoints.emplace_back(firstpoint, firstnormal);
            if (rtnfilteredPoints.size() == size + 1)
            {
                break;
            }
       }
       return rtnfilteredPoints;
    }

};


struct PointUser {
    double x, y, z;

    PointUser operator-(const PointUser& other) const {
        return { x - other.x, y - other.y, z - other.z };
    }

    double distance(const PointUser& other) const {
        return std::sqrt(std::pow(x - other.x, 2) + std::pow(y - other.y, 2) + std::pow(z - other.z, 2));
    }

    bool isCollinear(const PointUser& p1, const PointUser& p2, double epsilon) const {
        double v1x = p1.x - x;
        double v1y = p1.y - y;
        double v1z = p1.z - z;
        double v2x = p2.x - x;
        double v2y = p2.y - y;
        double v2z = p2.z - z;

        double crossProdZ = v1x * v2y - v1y * v2x;
        double collinearity = std::abs(crossProdZ);
        return collinearity < epsilon;
    }
    PointUser(double xx, double yy, double zz): x(xx), y(yy), z(zz) {}

};

std::vector<PointUser> processPoints(const std::vector<PointUser>& points, double spacing) {
    if (points.size() < 2) {
        return points;  // Not enough points to process  
    }

    std::vector<PointUser> processedPoints;
    processedPoints.push_back(points[0]);

    for (size_t i = 1; i < points.size(); ++i) {
        size_t j = i;
        while (j < points.size() - 1 && points[j].isCollinear(points[j - 1], points[j + 1], 0.00001)) {
            j++;
        }
        processedPoints.push_back(points[j]); 
        if (j > i) {
            PointUser start = points[i - 1];
            PointUser end = points[j];
            double totalDistance = start.distance(end);
            int numSamples = std::max(1, static_cast<int>(std::floor(totalDistance / spacing)));

            for (int k = 1; k <= numSamples; ++k) {
                double t = static_cast<double>(k) / (numSamples + 1);
                PointUser newPoint = { start.x + t * (end.x - start.x),
                                  start.y + t * (end.y - start.y),
                                  start.z + t * (end.z - start.z) };
                processedPoints.push_back(newPoint);
            }
        }
        i = j;    // Move to the next non-collinear segment  
    }

    return processedPoints;
}
/// <summary>
/// 获取关键点
/// </summary>
/// <param name="ordata"></param>
/// <param name="deltadis"></param>
/// <returns></returns>
std::vector<K::Point_3> getkeypoint(std::vector<K::Point_3>& ordata ,float deltadis = 0.001)
{
    std::vector<K::Point_3> pointsrtn;
    if (ordata.size() <= 1)
    {
        return ordata;
    }
    pointsrtn.push_back(ordata.at(0));
    K::Point_3 flagpoint = ordata.at(0);
    K::Point_3 lastpoint = flagpoint;
    double current_total_distance = 0.0;
    double current_distance = 0.0;
    for (auto point : ordata)
    {
       float tempdis = std::sqrt( CGAL::squared_distance(lastpoint, point));
       current_total_distance = current_total_distance + tempdis;
       current_distance = std::sqrt(CGAL::squared_distance(flagpoint, point));  //控制点的直线距离
       if (current_total_distance - current_distance > deltadis)
       {
           pointsrtn.push_back(lastpoint);
           flagpoint = lastpoint;
           current_total_distance = tempdis;
       }
       lastpoint = point;
    }
    pointsrtn.push_back(lastpoint);
    ordata.clear();
    ordata = pointsrtn;
    return pointsrtn;
}

/// <summary>
/// 存放笔宽以及对应笔宽的点与法线
/// </summary>
/// <param name="muitlpenwidth"></param>
/// <param name="filepath"></param>
void outfiles(std::vector<line_width> muitlpenwidth, std::string filepath)
{
    std::string penwidth = filepath + "penwidth" + std::to_string(0) + ".txt";
    std::ofstream foutnormalepen(penwidth, std::ios::trunc);
    if (!foutnormalepen.is_open())
    {
        std::cout << "fail" << endl;
        std::cerr << "未能创建路径文件!" << std::endl;
        return;
    }
    for (auto ele : muitlpenwidth)
    {
        foutnormalepen << ele.begin()->second << '\n';
    }
    foutnormalepen.close();

    for (auto ele : muitlpenwidth)
    {
        static int sss = 0;
        std::string everoutpath = filepath + "P" + std::to_string(sss) + "_P" + ".txt";
        std::ofstream routespoints(everoutpath, std::ios::trunc);
        if (!routespoints.is_open())
        {
            std::cout << "fail" << endl;
            std::cerr << "未能创建点路径文件!" << std::endl;
            return;
        }
        for (auto p : ele)
        {
            for (auto pp : p.first)
            {
                if (pp.first.x() == 0 && pp.first.y() == 0 && pp.first.z() == 0)
                {
                    continue;
                }
                routespoints << pp.first << '\n';
            }
        }
        routespoints.close();


        std::string everoutpathnormal = filepath + "P" + std::to_string(sss) + "_N" + ".txt";
        std::ofstream routesnormals(everoutpathnormal, std::ios::trunc);
        if (!routesnormals.is_open())
        {
            std::cout << "fail" << endl;
            std::cerr << "未能创建法向路径文件!" << std::endl;
            return;
        }
        for (auto p : ele)
        {
            for (auto pp : p.first)
            {
                if (pp.second.x() == 0 && pp.second.y() == 0 && pp.second.z() == 0)
                {
                    continue;
                }
                routesnormals << pp.second << '\n';
            }
        }
        routesnormals.close();
        sss++;
    }
}


int main(int argc, char* argv[])
{
    PathPlannerConfig config;
    std::string modelfullpath;
    std::string modelName;
    std::string path_routes;
    modelfullpath.clear();
    modelName.clear();
    path_routes.clear();
    if (argc == 1)
    {
        modelfullpath = R"(D:\Desktop\STL\1111111.stl)";
        is_local_test = true;
    }
    else if (argc == 11)
    {
        modelName = argv[1];
        path_routes = argv[2];
        config.gap_border = std::stod(argv[3]) / 2;
        config.gap_inner = std::stod(argv[3]) - std::stod(argv[4]); //argv[4]相当于重叠的部分
        config.epsilon_border = std::stod(argv[5]);
        config.near_border_iters = std::stod(argv[6]);
        config.min_lineswidth = std::stod(argv[7]);
        config.startpointinner = (bool)std::stoi(argv[8]);
        //config.datasample = (bool)std::stoi(argv[9]);
        config.skeletonization = (bool)std::stoi(argv[9]);
        config.ismuitlpenwidth = (bool)std::stoi(argv[10]);
        modelfullpath = path_routes + modelName + ".stl";
        is_local_test = false;
    }
    else
    {
        return EXIT_FAILURE;
    }
    Mesh mesh;
    const std::string filename = CGAL::data_file_path(modelfullpath);
    std::vector<K::Point_3> points;
    std::vector<std::vector<std::size_t> > polygons;
    //保证处理的stl是标准的流行表面
    if (CGAL::IO::read_polygon_soup(filename, points, polygons) || points.size() == 0 || polygons.size() == 0)
    {
        PMP::orient_polygon_soup(points, polygons);
        PMP::repair_polygon_soup(points, polygons);
        PMP::polygon_soup_to_polygon_mesh(points, polygons, mesh);
    }
    else
    {
        if (!CGAL::IO::read_polygon_mesh(filename, mesh)) {
            std::cerr << "文件路径不对！！！！" << std::endl;
            return EXIT_FAILURE;
        }
    }
    
    std::cout << "original:\n"
        << "\t#vertices " << mesh.number_of_vertices() << std::endl
        << "\t#edges " << mesh.number_of_edges() << std::endl
        << "\t#faces " << mesh.number_of_faces() << std::endl;

    if(is_local_test)
        CGAL::draw(mesh);

    // 寻找连通区域，保留天线的数量由给定的参数决定
    std::vector<Mesh> cc;
    PMP::keep_largest_connected_components(mesh,1);
    PMP::split_connected_components(mesh, cc);
    std::cout << "#CC found: " << cc.size() << std::endl;
    std::vector<K::Point_3> path;
    std::vector<std::pair<K::Point_3, K::Vector_3>> normalspath;
    
    std::vector<K::Vector_3> v_normals;
    auto t1 = std::chrono::system_clock::now();

    int index = 0;
    for (auto& mesh : cc) 
    {
        // 规划路径
        PathPlanner planner(config);
        normalspath = planner.plan(mesh);
        path.reserve(normalspath.size());
        v_normals.reserve(normalspath.size());
        if (config.ismuitlpenwidth)
        {
            planner.mergerLinesType(planner.allpath);
            planner.rangeLinespre();
            outfiles(planner.linesclassify, path_routes + modelName);
        }
        else
        {
            for (const auto& entry : normalspath)
            {
                path.push_back(entry.first);        // entry.first 是 K::Point_3 类型
                v_normals.push_back(entry.second);  // entry.second 是 K::Vector_3 类型
            }

            //getkeypoint(path);
            
            // routes
            CGAL::Point_set_3<K::Point_3> points;
            for (auto p : path) {
                if (p.x() == 0 && p.y() == 0 && p.z() == 0)
                {
                    continue;
                }
                points.insert(p);
            }

            if (is_local_test)
                CGAL::draw(points);

            // 导出坐标 Point
            std::string routes_fullpathname = path_routes + modelName + "P" + std::to_string(index) + "_P" + ".txt";
            std::ofstream foutroute(routes_fullpathname, std::ios::trunc);
            if (!foutroute.is_open()) {
                std::cerr << "未能创建路径文件!" << std::endl;
                return EXIT_FAILURE;
            }
            for (auto p : path) {
                if (p.x() == 0 && p.y() == 0 && p.z() == 0)
                {
                    continue;
                }
                foutroute << p << '\n';
            }
            foutroute.close();

            //normal
            std::string normal_fullpathname = path_routes + modelName + "P" + std::to_string(index) + "_N" + ".txt";
            std::ofstream foutnormal(normal_fullpathname, std::ios::trunc);
            if (!foutnormal.is_open()) {
                std::cerr << "未能创建路径文件!" << std::endl;
                return EXIT_FAILURE;
            }
            for (auto np : v_normals) {
                if (np.x() == 0 && np.y() == 0 && np.z() == 0)
                {
                    continue;
                }
                foutnormal << np << '\n';
            }
            foutnormal.close();
        }
        index++;
    }
    auto t2 = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "done.\n time cost all: " << duration.count() * 1e-3 << std::endl;
    return EXIT_SUCCESS;
}



